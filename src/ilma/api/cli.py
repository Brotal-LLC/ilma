"""Typer CLI for ilma.

The CLI is intentionally framework-agnostic: it imports only ilma APIs/storage and
standard third-party CLI/runtime packages.  It does not depend on Hermes Agent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Annotated, Any, cast

import typer

from ilma import __version__
from ilma.api.mcp import IlmaConfigError, IlmaMcpService, _dsn_from_env, create_mcp_server
from ilma.embeddings import EmbedderRegistry
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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
def repair(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Repair/verify all schemas and audit logging."""

    result = _service_from_env().ilma_repair()
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        typer.echo(str(result.get("message") or "Repair complete."))


@app.command()
def migrate(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON output.")] = False,
) -> None:
    """Run idempotent schema migrations for all surfaces."""

    result = _service_from_env().ilma_migrate()
    _exit_if_failed(result, json_output=json_output)
    if json_output:
        _echo_json(result)
    else:
        typer.echo(
            f"Migration complete: surfaces={result.get('surfaces', len(SURFACES))} "
            f"audit_log={result.get('audit_log', True)}"
        )


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
