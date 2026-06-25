"""Typer CLI for ilma.

The CLI is intentionally framework-agnostic: it imports only ilma APIs/storage and
standard third-party CLI/runtime packages.  It does not depend on Hermes Agent.
"""

from __future__ import annotations

import csv
import inspect
import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Annotated, Any, cast, get_type_hints

import typer

from ilma import __version__
from ilma.api.mcp import IlmaConfigError, IlmaMcpService, _dsn_from_env, create_mcp_server
from ilma.config import IlmaConfig
from ilma.embeddings import EmbedderRegistry
from ilma.migration import migrate_hermes_config, migrate_hermes_v2_schema
from ilma.service import method_description, method_to_pydantic_model, tools_dict
from ilma.storage.postgres import PgBackend

try:  # psycopg is a core dependency, but keep import failure user-friendly.
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

app = typer.Typer(
    name="ilma",
    help="Framework-agnostic agent memory system — Postgres + pgvector + MCP.",
    no_args_is_help=True,
)

INIT_STEPS = (
    "Resolve Postgres DSN",
    "Check Postgres connection",
    "Create pgvector extension",
    "Create ilma schema",
    "Initialize generic backend table",
    "Initialize all surface schemas",
    "Initialize audit log",
    "Verify embedder reachability",
    "Print environment hints",
)

SURFACES = (
    "memory",
    "wiki",
    "journal",
    "skills",
    "kanban",
    "metrics",
    "observability",
    "sessions",
)


def _json_safe(value: Any) -> Any:
    """Convert common Python objects to JSON-safe structures."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _echo_json(payload: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


def _stringify_csv_value(value: Any) -> str:
    value = _json_safe(value)
    if isinstance(value, Mapping | list):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _echo_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "id",
        "operation_id",
        "tool_name",
        "surface",
        "action",
        "status",
        "created_at",
        "completed_at",
        "error_type",
        "error_message",
        "payload",
        "result",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _stringify_csv_value(row.get(field)) for field in fields})
    typer.echo(buffer.getvalue().rstrip("\n"))


def _call_repair(service: Any, *, force: bool) -> Mapping[str, Any]:
    try:
        return cast(Mapping[str, Any], service.ilma_repair(force=force))
    except TypeError:
        if force:
            raise
        return cast(Mapping[str, Any], service.ilma_repair())


def _call_audit(
    service: Any,
    *,
    tool: str | None,
    status: str | None,
    start: str | None,
    end: str | None,
    limit: int,
    offset: int,
) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        service.ilma_audit(
            tool=tool,
            status=status,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        ),
    )


def _error_message(result: Mapping[str, Any]) -> str:
    error = result.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        error_type = error.get("type")
        if error_type and message:
            return f"{error_type}: {message}"
        if message:
            return str(message)
    return "unknown error"


def _exit_if_failed(result: Mapping[str, Any], *, json_output: bool = False) -> None:
    if result.get("ok", False):
        return
    if json_output:
        _echo_json(result)
    else:
        typer.secho(f"Error: {_error_message(result)}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _service_from_env() -> IlmaMcpService:
    return IlmaMcpService.from_env()


def _print_step(index: int, message: str) -> None:
    typer.echo(f"[{index}/9] {message} ...", nl=False)


def _finish_step(ok: bool = True) -> None:
    typer.secho(" ok" if ok else " failed", fg=typer.colors.GREEN if ok else typer.colors.RED)


def _dsn_prompt(default: str | None = None) -> str:
    prompt = "Postgres DSN"
    if default:
        return cast(str, typer.prompt(prompt, default=default))
    return cast(str, typer.prompt(prompt))


def _resolve_init_dsn(dsn: str | None, *, yes: bool) -> str:
    if dsn:
        return dsn.strip()
    try:
        return _dsn_from_env()
    except IlmaConfigError:
        if yes:
            msg = "set ILMA_DSN or pass --dsn for non-interactive init"
            raise typer.BadParameter(msg) from None
        return _dsn_prompt()


def _connect_for_init(dsn: str) -> Any:
    if psycopg is None:  # pragma: no cover
        raise RuntimeError("psycopg is required for init")
    return psycopg.connect(dsn, autocommit=True)


def _verify_embedder() -> dict[str, Any]:
    registry = EmbedderRegistry.from_env()
    vector = registry.embed("ilma init embedder reachability check")
    return {"default_dim": registry.default_dim, "vector_length": len(vector)}


@app.callback()
def _version_callback(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print the installed ilma version and exit."),
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("init")
def init_command(
    dsn: Annotated[str | None, typer.Option("--dsn", help="Postgres DSN to configure.")] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", "-y", help="Run non-interactively; fail if required input is missing."
        ),
    ] = False,
    skip_embedder_check: Annotated[
        bool,
        typer.Option(
            "--skip-embedder-check",
            help="Skip the live embedding request during initialization.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Run the 9-step initialization wizard."""

    result: dict[str, Any] = {"ok": True, "steps": [], "surfaces": list(SURFACES)}
    try:
        if not json_output:
            typer.echo("ilma init — 9-step setup wizard")

        _print_step(1, INIT_STEPS[0]) if not json_output else None
        resolved_dsn = _resolve_init_dsn(dsn, yes=yes)
        os.environ["ILMA_DSN"] = resolved_dsn
        result["dsn_configured"] = True
        result["steps"].append({"step": 1, "name": INIT_STEPS[0], "ok": True})
        _finish_step() if not json_output else None

        _print_step(2, INIT_STEPS[1]) if not json_output else None
        with _connect_for_init(resolved_dsn) as connection:
            connection.execute("SELECT 1")
        result["steps"].append({"step": 2, "name": INIT_STEPS[1], "ok": True})
        _finish_step() if not json_output else None

        _print_step(3, INIT_STEPS[2]) if not json_output else None
        with _connect_for_init(resolved_dsn) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        result["steps"].append({"step": 3, "name": INIT_STEPS[2], "ok": True})
        _finish_step() if not json_output else None

        _print_step(4, INIT_STEPS[3]) if not json_output else None
        with _connect_for_init(resolved_dsn) as connection:
            connection.execute("CREATE SCHEMA IF NOT EXISTS ilma")
        result["steps"].append({"step": 4, "name": INIT_STEPS[3], "ok": True})
        _finish_step() if not json_output else None

        _print_step(5, INIT_STEPS[4]) if not json_output else None
        backend = PgBackend(resolved_dsn)
        backend.initialize_schema()
        result["steps"].append({"step": 5, "name": INIT_STEPS[4], "ok": True})
        _finish_step() if not json_output else None

        _print_step(6, INIT_STEPS[5]) if not json_output else None
        service = IlmaMcpService(backend)
        migrate_result = service.ilma_migrate()
        if not migrate_result.get("ok"):
            raise RuntimeError(_error_message(migrate_result))
        result["steps"].append({"step": 6, "name": INIT_STEPS[5], "ok": True})
        _finish_step() if not json_output else None

        _print_step(7, INIT_STEPS[6]) if not json_output else None
        service.audit.initialize_schema()
        result["steps"].append({"step": 7, "name": INIT_STEPS[6], "ok": True})
        _finish_step() if not json_output else None

        _print_step(8, INIT_STEPS[7]) if not json_output else None
        if skip_embedder_check:
            result["embedder"] = {"skipped": True}
        else:
            result["embedder"] = _verify_embedder()
        result["steps"].append({"step": 8, "name": INIT_STEPS[7], "ok": True, **result["embedder"]})
        _finish_step() if not json_output else None

        _print_step(9, INIT_STEPS[8]) if not json_output else None
        config = IlmaConfig.from_env()
        result["env"] = {
            "ILMA_DSN": "set",
            "ILMA_EMBED_PROVIDER": config.vectors.embedder,
            "ILMA_EMBED_MODEL": config.vectors.openai_model,
            "ILMA_EMBED_DIM": str(config.vectors.dim),
        }
        result["steps"].append({"step": 9, "name": INIT_STEPS[8], "ok": True})
        _finish_step() if not json_output else None
    except Exception as exc:
        result = {
            **result,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if json_output:
            _echo_json(result)
        else:
            typer.secho(
                f"\nInit failed: {type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True
            )
        raise typer.Exit(1) from exc

    if json_output:
        _echo_json(result)
    else:
        typer.secho("ilma initialized successfully", fg=typer.colors.GREEN)
        typer.echo("Set ILMA_DSN in your environment to use ilma from other shells.")


def status(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Show backend health and memory statistics."""

    result = _service_from_env().ilma_status()
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    backend = result.get("backend", {})
    memory = result.get("memory", {})
    typer.echo("ilma status")
    if isinstance(backend, Mapping):
        typer.echo(
            f"Backend: {'ok' if backend.get('ok') else 'unhealthy'}"
            f" database={backend.get('database', 'unknown')}"
            f" pgvector={backend.get('pgvector', 'unknown')}"
        )
    if isinstance(memory, Mapping):
        typer.echo(
            "Memory: "
            f"live={memory.get('live_memories', 0)} "
            f"total={memory.get('total_memories', 0)} "
            f"chunks={memory.get('total_chunks', 0)}"
        )
    typer.echo("Surfaces: " + ", ".join(str(s) for s in result.get("surfaces", SURFACES)))


def recall(
    query: Annotated[str, typer.Argument(help="Recall query.")],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=100)] = 10,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            min=0.0,
            max=1.0,
            help="Minimum similarity score (0.0 = no filter).",
        ),
    ] = 0.0,
    hybrid_text_weight: Annotated[
        float,
        typer.Option("--hybrid-text-weight", min=0.0, max=1.0),
    ] = 0.5,
    expand_graph: Annotated[
        bool,
        typer.Option(
            "--expand-graph",
            help="Expand recall hits with 1-hop graph neighbors.",
        ),
    ] = False,
    graph_hops: Annotated[
        int,
        typer.Option(
            "--graph-hops",
            min=1,
            max=3,
            help="Max traversal depth when --expand-graph is set.",
        ),
    ] = 1,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Recall memories relevant to a query. Canonical recall surface."""

    result = _service_from_env().ilma_recall(
        query,
        limit=limit,
        threshold=threshold,
        hybrid_text_weight=hybrid_text_weight,
        expand_graph=expand_graph,
        graph_hops=graph_hops,
    )
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    results = _json_safe(result.get("results", []))
    count = result.get("count", len(results))
    typer.echo(
        f"Recall: {count} result(s) for query={query!r} (limit={result.get('limit', limit)})"
    )
    if not results:
        typer.echo("No memories found.")
        return
    for item in results:
        if not isinstance(item, Mapping):
            typer.echo(str(item))
            continue
        memory_id = item.get("id", "?")
        content = str(item.get("content", "")).replace("\n", " ")
        tags = item.get("tags") or []
        category = item.get("category") or "uncategorized"
        typer.echo(f"[{memory_id}] {content}")
        typer.echo(f"    category={category} tags={', '.join(map(str, tags)) if tags else '-'}")

    if expand_graph:
        neighbors = _json_safe(result.get("graph_neighbors", []))
        if neighbors:
            typer.echo(f"\nGraph neighbors ({len(neighbors)}, max_hops={graph_hops}):")
            for n in neighbors:
                if not isinstance(n, Mapping):
                    continue
                nid = n.get("id", "?")
                kind = n.get("kind", "?")
                via = n.get("via_memory_id")
                props = n.get("properties") or {}
                category = props.get("category") or ""
                typer.echo(f"  [{nid}] kind={kind} via_memory={via} category={category or '-'}")


def remember(
    content: Annotated[str, typer.Argument(help="Memory content to store.")],
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "--tags", help="Tag to attach; may be repeated."),
    ] = None,
    category: Annotated[str | None, typer.Option("--category", "-c")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Store a memory."""

    result = _service_from_env().ilma_remember(content, tags or [], category, "cli")
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    memory_id = result.get("memory_id")
    if memory_id:
        typer.echo(f"Remembered memory {memory_id}.")
    else:
        typer.echo("Memory already exists; no duplicate stored.")


def forget(
    memory_id: Annotated[int, typer.Argument(help="Memory ID to soft-delete.", min=1)],
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Soft-delete a memory by ID."""

    result = _service_from_env().ilma_forget(memory_id)
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    if result.get("deleted"):
        typer.echo(f"Deleted memory {memory_id}.")
    else:
        typer.echo(f"Memory {memory_id} was not found or was already deleted.")


def list_memories(
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of memories to return (1-1000).",
        ),
    ] = 50,
    offset: Annotated[
        int,
        typer.Option(
            "--offset",
            help="Number of memories to skip (for pagination).",
        ),
    ] = 0,
    include_deleted: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Include soft-deleted memories (deleted_at IS NOT NULL).",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
    csv_output: Annotated[
        bool,
        typer.Option(
            "--csv", help="Emit CSV output (id, created_at, deleted, category, tags, content)."
        ),
    ] = False,
) -> None:
    """List memories in reverse chronological order (newest first).

    Default: live memories only. Pass --all to include soft-deleted rows
    (useful for debugging data-repair migrations and audit checks).
    Use --offset for paging through large sets.
    """

    if csv_output and json_output:
        typer.echo("--csv and --json are mutually exclusive.", err=True)
        raise typer.Exit(code=2)

    result = _service_from_env().ilma_list_memories(
        limit=limit,
        offset=offset,
        include_deleted=include_deleted,
    )
    _exit_if_failed(result, json_output=json_output)
    rows = _json_safe(result.get("results", []))

    if csv_output:
        fieldnames = ["id", "created_at", "deleted", "category", "tags", "content"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            tags = row.get("tags") or []
            writer.writerow(
                {
                    "id": row.get("id"),
                    "created_at": _stringify_csv_value(row.get("created_at")),
                    "deleted": row.get("deleted"),
                    "category": row.get("category") or "",
                    "tags": ",".join(map(str, tags)),
                    "content": _stringify_csv_value(row.get("content")),
                }
            )
        typer.echo(buf.getvalue().rstrip("\n"))
        return

    if json_output:
        _echo_json(result)
        return

    if not rows:
        typer.echo("No memories found.")
        return

    count = result.get("count", len(rows))
    showing = f"showing {count}"
    if include_deleted:
        showing += " (including deleted)"
    typer.echo(f"Memories: {showing} (limit={limit}, offset={offset})")
    for row in rows:
        memory_id = row.get("id", "?")
        created_at = row.get("created_at")
        created_str = ""
        if isinstance(created_at, datetime):
            created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        elif created_at is not None:
            created_str = str(created_at)
        deleted = bool(row.get("deleted"))
        category = row.get("category") or "uncategorized"
        tags = row.get("tags") or []
        tags_str = ",".join(map(str, tags)) if tags else "-"
        content = str(row.get("content", "")).replace("\n", " ")
        if len(content) > 200:
            content = content[:197] + "..."
        deleted_marker = " [DELETED]" if deleted else ""
        typer.echo(
            f"[{memory_id}]{deleted_marker} {created_str}  category={category}  tags={tags_str}"
        )
        typer.echo(f"    {content}")


def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Run backend, schema, and audit-log health checks."""

    result = _service_from_env().ilma_doctor()
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        healthy = bool(result.get("healthy"))
        typer.secho("ilma doctor: healthy" if healthy else "ilma doctor: unhealthy")
        checks = result.get("checks", {})
        if isinstance(checks, Mapping):
            for name, check in checks.items():
                if name == "surfaces" and isinstance(check, Mapping):
                    for surface, surface_check in check.items():
                        ok = isinstance(surface_check, Mapping) and surface_check.get("ok")
                        typer.echo(f"  surface.{surface}: {'ok' if ok else 'failed'}")
                elif isinstance(check, Mapping):
                    typer.echo(f"  {name}: {'ok' if check.get('ok', False) else 'failed'}")
                else:
                    typer.echo(f"  {name}: {check}")
        if not healthy:
            raise typer.Exit(1)


def repair(
    force: Annotated[
        bool,
        typer.Option("--force", help="Apply repairs (delete orphan chunks, reindex, vacuum)."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Inspect storage damage and optionally repair safe Postgres issues."""

    result = _call_repair(_service_from_env(), force=force)
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        typer.echo(str(result.get("message") or "Repair complete."))
        findings = result.get("findings", {})
        if isinstance(findings, Mapping):
            orphaned = findings.get("orphaned_chunks", {})
            duplicates = findings.get("duplicate_memories", {})
            fts = findings.get("fts_indexes", {})
            if isinstance(orphaned, Mapping):
                typer.echo(f"  orphaned_chunks={orphaned.get('count', 0)}")
            if isinstance(duplicates, Mapping):
                typer.echo(f"  duplicate_memory_groups={duplicates.get('count', 0)}")
            if isinstance(fts, Mapping):
                missing = fts.get("missing", [])
                typer.echo(
                    f"  missing_fts_indexes={len(missing) if isinstance(missing, list) else 0}"
                )


def audit(
    tool: Annotated[str | None, typer.Option("--tool", help="Filter by tool name.")] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by audit status: pending, succeeded, failed."),
    ] = None,
    start: Annotated[
        str | None,
        typer.Option("--start", help="Inclusive ISO-8601 created_at lower bound."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="Inclusive ISO-8601 created_at upper bound."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
    offset: Annotated[int, typer.Option("--offset", min=0)] = 0,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: table, json, or csv."),
    ] = "table",
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Query the write-ahead audit log."""

    result = _call_audit(
        _service_from_env(),
        tool=tool,
        status=status,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )
    _exit_if_failed(result, json_output=json_output or output_format == "json")
    rows = result.get("results", [])
    if not isinstance(rows, list):
        rows = []
    safe_rows = [row for row in rows if isinstance(row, Mapping)]
    fmt = output_format.strip().lower()
    if json_output or fmt == "json":
        _echo_json(result)
    elif fmt == "csv":
        _echo_csv(safe_rows)
    elif fmt == "table":
        if not safe_rows:
            typer.echo("No audit log rows found.")
            return
        for row in safe_rows:
            typer.echo(
                f"[{row.get('id', '?')}] {row.get('created_at', '')} "
                f"{row.get('tool_name', '?')} {row.get('status', '?')} "
                f"surface={row.get('surface', '?')} action={row.get('action', '?')}"
            )
            if row.get("error_message"):
                typer.echo(f"    error={row.get('error_type')}: {row.get('error_message')}")
    else:
        raise typer.BadParameter("--format must be one of: table, json, csv")


def migrate(
    dsn: Annotated[str | None, typer.Option("--dsn", help="Postgres DSN to migrate.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Inspect hermes-memory v2 data without writing ilma rows."),
    ] = False,
    reembed: Annotated[
        bool,
        typer.Option(
            "--reembed",
            help=(
                "After schema migration, post-process live memories: clean corrupt "
                "tags arrays, re-embed memories whose vector columns are zero "
                "(source hermes-memory v2 did not pre-compute embeddings), and "
                "chunk content that arrived with no chunks. Idempotent."
            ),
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Migrate hermes-memory v2 data into ilma, or run ilma schema migrations.

    Pass --reembed on a non-pristine source DB to repair known migration
    artifacts (corrupt tags, zero vectors, missing chunks). Safe to re-run.
    """

    resolved_dsn = dsn
    if resolved_dsn is None:
        try:
            resolved_dsn = _dsn_from_env()
        except IlmaConfigError:
            resolved_dsn = None

    if resolved_dsn is not None:
        progress = None if json_output else (lambda message: typer.echo(f"- {message}"))
        result = migrate_hermes_v2_schema(resolved_dsn, dry_run=dry_run, progress=progress)
        _exit_if_failed(result, json_output=json_output)
        if result.get("detected"):
            if json_output:
                _echo_json(result)
            else:
                typer.echo(
                    f"Migration complete: inserted={result.get('inserted', 0)} "
                    f"updated={result.get('updated', 0)} skipped={result.get('skipped', 0)} "
                    f"conflicts={result.get('conflicts', 0)}"
                )
            # Always also call ilma_migrate(reembed=True) when --reembed is set,
            # even when v2 schema was detected. The v2 → ilma migration is
            # content-hash-deduplicated, so re-running it on an already-migrated
            # DB is a no-op (inserted=0). The repair pass then runs the actual
            # fix-up logic (corrupt tags, zero vectors, missing chunks).
            if reembed:
                result2 = _service_from_env().ilma_migrate(reembed=True)
                _exit_if_failed(result2, json_output=json_output)
                if json_output:
                    _echo_json(result2)
                else:
                    repair_stats = result2.get("reembed") or {}
                    typer.echo(
                        f"Reembed pass: tags_cleaned={repair_stats.get('tags_cleaned', 0)} "
                        f"memories_embedded={repair_stats.get('memories_embedded', 0)} "
                        f"chunks_created={repair_stats.get('chunks_created', 0)} "
                        f"skipped={repair_stats.get('skipped', '')}"
                    )
            return
        if dry_run:
            if json_output:
                _echo_json(result)
            else:
                typer.echo(str(result.get("message") or "No hermes-memory v2 schema found."))
            return

    result = _service_from_env().ilma_migrate(reembed=reembed)
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        typer.echo(
            f"Migration complete: surfaces={result.get('surfaces', len(SURFACES))} "
            f"audit_log={result.get('audit_log', True)}" + ("  reembed applied" if reembed else "")
        )


_CLI_EXCLUDED = frozenset({"init", "mcp", "serve", "migrate-config"})
_CLI_TOOL_TO_COMMAND: Mapping[str, str] = {
    "ilma_status": "status",
    "ilma_recall": "recall",
    "ilma_remember": "remember",
    "ilma_forget": "forget",
    "ilma_list_memories": "list",
    "ilma_doctor": "doctor",
    "ilma_repair": "repair",
    "ilma_audit": "audit",
    "ilma_migrate": "migrate",
}


def _cli_tool_specs() -> dict[str, dict[str, Any]]:
    """Return service-derived specs for tools that have CLI command equivalents."""

    specs = dict(tools_dict(IlmaMcpService))
    if "ilma_audit" not in specs:
        audit_method = IlmaMcpService.ilma_audit
        specs["ilma_audit"] = {
            "description": method_description(audit_method),
            "input_model": method_to_pydantic_model(audit_method),
            "handler": None,
        }
    return {name: specs[name] for name in _CLI_TOOL_TO_COMMAND if name in specs}


def _make_auto_command(command_name: str, implementation: Any) -> Any:
    """Wrap an implementation function in an auto-registered Typer callback."""

    def command(**kwargs: Any) -> None:
        implementation(**kwargs)
        return None

    command.__name__ = command_name.replace("-", "_")
    command.__doc__ = implementation.__doc__
    type_hints = get_type_hints(implementation, include_extras=True)
    signature = inspect.signature(implementation)
    command.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[
            parameter.replace(annotation=type_hints.get(name, parameter.annotation))
            for name, parameter in signature.parameters.items()
        ]
    )
    command.__annotations__ = type_hints
    command.__ilma_auto_registered__ = True  # type: ignore[attr-defined]
    return command


def _register_service_commands(typer_app: typer.Typer) -> None:
    """Register CLI commands for service methods by walking ``tools_dict()``."""

    implementations = {
        "ilma_status": status,
        "ilma_recall": recall,
        "ilma_remember": remember,
        "ilma_forget": forget,
        "ilma_list_memories": list_memories,
        "ilma_doctor": doctor,
        "ilma_repair": repair,
        "ilma_audit": audit,
        "ilma_migrate": migrate,
    }
    for tool_name in _cli_tool_specs():
        command_name = _CLI_TOOL_TO_COMMAND[tool_name]
        if command_name in _CLI_EXCLUDED:
            continue
        implementation = implementations[tool_name]
        typer_app.command(command_name)(_make_auto_command(command_name, implementation))


_register_service_commands(app)


@app.command("migrate-config")
def migrate_config_command(
    config: Annotated[
        str | None,
        typer.Option(
            "--config", help="Path to Hermes config.yaml (default: ~/.hermes/config.yaml)."
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show changes without writing.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Update ~/.hermes/config.yaml from hermes-memory postgres to ilma."""

    try:
        result = migrate_hermes_config(config, dry_run=dry_run)
    except Exception as exc:
        result = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    if result.get("changed"):
        typer.echo(f"Updated {result.get('config_path')}")
        if result.get("backup_path"):
            typer.echo(f"Backup written to {result.get('backup_path')}")
    else:
        typer.echo("Config already uses ilma; no changes needed.")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535)] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Enable uvicorn reload.")] = False,
) -> None:
    """Start the HTTP API with uvicorn."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        typer.secho(
            "uvicorn is required for `ilma serve`; install ilma with the 'http' extra.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from exc
    uvicorn.run(
        "ilma.api.http:app_factory",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


@app.command("mcp")
def mcp_command() -> None:
    """Start the MCP server."""

    create_mcp_server().run()


# ---------------------------------------------------------------------------
# Graph subcommand
# ---------------------------------------------------------------------------


@app.command("graph")
def graph_command(
    action: Annotated[
        str,
        typer.Argument(
            help="Graph action.",
            case_sensitive=False,
        ),
    ] = "rebuild",
    min_shared_tags: Annotated[
        int,
        typer.Option(
            "--min-shared-tags",
            min=1,
            help="Minimum number of shared tags for a SHARES_TAG edge.",
        ),
    ] = 2,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            "-k",
            help="Vertex kind (for traverse).",
            case_sensitive=False,
        ),
    ] = "Memory",
    src_id: Annotated[
        int,
        typer.Option(
            "--src-id",
            help="Source vertex SQL id (for traverse).",
        ),
    ] = 0,
    max_hops: Annotated[
        int,
        typer.Option("--max-hops", min=0, max=3),
    ] = 2,
    edge_types: Annotated[
        list[str] | None,
        typer.Option(
            "--edge-type",
            help="Edge-type whitelist (repeatable).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=500),
    ] = 50,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Manage the ilma graph layer (Apache AGE).

    \b
    Actions:
      rebuild       Drop and rebuild the graph from current SQL state.
      traverse      Bounded BFS from a vertex (requires --kind and --src-id).
    """
    action_normalized = action.lower()
    if action_normalized == "rebuild":
        result = _service_from_env().ilma_graph_rebuild(min_shared_tags=min_shared_tags)
        _exit_if_failed(result, json_output=json_output)
        if json_output:
            _echo_json(result)
            return
        stats = result.get("stats") or {}
        typer.echo("Graph rebuilt:")
        for key in (
            "memory_vertices",
            "wiki_vertices",
            "skill_vertices",
            "shares_tag_edges",
            "co_occurs_edges",
            "references_wiki_edges",
            "uses_skill_edges",
        ):
            typer.echo(f"  {key}: {stats.get(key, 0)}")
        return
    if action_normalized == "traverse":
        if src_id <= 0:
            typer.secho(
                "--src-id is required for traverse (must be > 0).",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2)
        result = _service_from_env().ilma_traverse(
            kind=kind,
            src_id=src_id,
            max_hops=max_hops,
            edge_types=edge_types,
            limit=limit,
        )
        _exit_if_failed(result, json_output=json_output)
        if json_output:
            _echo_json(result)
            return
        sub = _json_safe(result.get("subgraph") or {})
        nodes = sub.get("nodes", [])
        edges = sub.get("edges", [])
        typer.echo(
            f"Traverse from {kind}#{src_id} (max_hops={max_hops}, "
            f"limit={limit}, edge_types={edge_types or 'all'}):"
        )
        typer.echo(f"  {len(nodes)} node(s), {len(edges)} edge(s)")
        for n in nodes[:limit]:
            if not isinstance(n, Mapping):
                continue
            typer.echo(
                f"  node: kind={n.get('kind')} src_id={n.get('src_id')} props={n.get('properties')}"
            )
        for e in edges[:limit]:
            if not isinstance(e, Mapping):
                continue
            typer.echo(
                f"  edge: {e.get('label')} {e.get('start_id')}->{e.get('end_id')} "
                f"props={e.get('properties')}"
            )
        return
    typer.secho(
        f"Unknown graph action: {action!r}. Use 'rebuild' or 'traverse'.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(2)


def main(args: Sequence[str] | None = None) -> None:
    """Console entry point for ``ilma``."""

    app(args=args)


__all__ = ["INIT_STEPS", "SURFACES", "app", "main"]
