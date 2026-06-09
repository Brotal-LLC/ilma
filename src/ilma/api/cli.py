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
        result["env"] = {
            "ILMA_DSN": "set",
            "ILMA_EMBED_PROVIDER": os.environ.get("ILMA_EMBED_PROVIDER", "ollama_local"),
            "ILMA_EMBED_MODEL": os.environ.get("ILMA_EMBED_MODEL", "bge-m3"),
            "ILMA_EMBED_DIM": os.environ.get("ILMA_EMBED_DIM", "1024"),
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


def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=500)] = 10,
    hybrid_text_weight: Annotated[
        float,
        typer.Option("--hybrid-text-weight", min=0.0, max=1.0),
    ] = 0.5,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Search memories."""

    result = _service_from_env().ilma_search(query, top_k, hybrid_text_weight)
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
        return
    results = _json_safe(result.get("results", []))
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
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Migrate hermes-memory v2 data into ilma, or run ilma schema migrations."""

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
            return
        if dry_run:
            if json_output:
                _echo_json(result)
            else:
                typer.echo(str(result.get("message") or "No hermes-memory v2 schema found."))
            return

    result = _service_from_env().ilma_migrate()
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        typer.echo(
            f"Migration complete: surfaces={result.get('surfaces', len(SURFACES))} "
            f"audit_log={result.get('audit_log', True)}"
        )


_CLI_EXCLUDED = frozenset({"init", "mcp", "serve", "migrate-config"})
_CLI_TOOL_TO_COMMAND: Mapping[str, str] = {
    "ilma_status": "status",
    "ilma_search": "search",
    "ilma_remember": "remember",
    "ilma_forget": "forget",
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
        return implementation(**kwargs)

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
        "ilma_search": search,
        "ilma_remember": remember,
        "ilma_forget": forget,
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


def main(args: Sequence[str] | None = None) -> None:
    """Console entry point for ``ilma``."""

    app(args=args)


__all__ = ["INIT_STEPS", "SURFACES", "app", "main"]
