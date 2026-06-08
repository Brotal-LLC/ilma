"""Hermes Agent plugin adapter for ilma.

Thin wrapper that registers ilma's MCP service methods as Hermes tools
and overrides the built-in `memory` tool when `memory.provider=ilma`.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_ILMA = "ilma"
PROVIDER_POSTGRES = "postgres"
PROVIDER_LOCAL = "local"
PROVIDER_AUTO = "auto"


def _read_provider() -> str:
    """Read memory.provider from env (priority) or ~/.hermes/config.yaml."""
    env_val = os.environ.get("MEMORY_PROVIDER", "").strip().lower()
    if env_val:
        return env_val

    hermes_home = os.environ.get("HERMES_HOME", "").strip() or os.path.expanduser("~/.hermes")
    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.is_file():
        return PROVIDER_LOCAL

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return PROVIDER_LOCAL

    if not isinstance(data, dict):
        return PROVIDER_LOCAL
    memory_section = data.get("memory")
    if not isinstance(memory_section, dict):
        return PROVIDER_LOCAL
    cfg_val = memory_section.get("provider")
    if not isinstance(cfg_val, str):
        return PROVIDER_LOCAL
    cfg_val = cfg_val.strip().lower()
    if cfg_val in (PROVIDER_ILMA, PROVIDER_POSTGRES, PROVIDER_LOCAL, PROVIDER_AUTO):
        return cfg_val
    return PROVIDER_LOCAL


def _passthrough_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "additionalProperties": True},
    }


def _docstring_or_default(fn: Callable, name: str) -> str:
    doc = inspect.getdoc(fn)
    if doc:
        return doc.strip().split("\n", 1)[0]
    return f"ilma tool: {name}"


def _bind(fn: Callable) -> Callable:
    def _handler(args: dict[str, Any], **_: Any) -> Any:
        return fn(**args)

    _handler.__name__ = getattr(fn, "__name__", "tool_handler")
    _handler.__doc__ = getattr(fn, "__doc__", None)
    return _handler


def _try_build_service() -> Any | None:
    try:
        from ilma.api.mcp import IlmaMcpService
        from ilma.storage.postgres import PgBackend
    except ImportError as exc:
        logger.warning("ilma: import failed: %s", exc)
        return None

    dsn = (
        os.environ.get("ILMA_DSN")
        or os.environ.get("PG_MEM_DB_CONN_STR")
        or os.environ.get("HERMES_PG_CONN_STR")
    )
    if not dsn:
        logger.warning(
            "ilma: no DSN configured (ILMA_DSN / PG_MEM_DB_CONN_STR / HERMES_PG_CONN_STR)"
        )
        return None

    min_pool_size = int(os.environ.get("ILMA_PG_POOL_MIN", "1"))
    max_pool_size = int(os.environ.get("ILMA_PG_POOL_MAX", "8"))
    try:
        backend = PgBackend(dsn, min_pool_size=min_pool_size, max_pool_size=max_pool_size)
        return IlmaMcpService(backend)
    except Exception as exc:
        logger.warning("ilma: service construction failed: %s", exc)
        return None


def register(ctx) -> None:
    """Hermes Agent plugin entry point."""
    provider = _read_provider()
    service = _try_build_service() if provider in (PROVIDER_ILMA, PROVIDER_AUTO) else None

    # If provider is 'postgres', keep hermes-memory v2 behavior (don't register ilma tools).
    if provider == PROVIDER_POSTGRES:
        logger.info("ilma: memory.provider=%s; leaving hermes-memory v2 in control", provider)
        return

    if service is None and provider != PROVIDER_ILMA:
        logger.info("ilma: memory.provider=%s; no service constructed", provider)
        return

    # Register ilma tools that map to IlmaMcpService methods.
    _register_ilma_tools(ctx, service)

    # Override the built-in `memory` tool.
    _register_memory_override(ctx, service)

    # Hooks
    if hasattr(ctx, "register_hook"):

        def _on_session_end(**_: Any) -> None:
            try:
                obs = service.observability if service is not None else None
                if obs is not None and hasattr(obs, "flush"):
                    obs.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("ilma: on_session_end flush failed: %s", exc)

        ctx.register_hook("on_session_end", _on_session_end)

        def _pre_tool_call(tool_name: str, **_: Any) -> dict[str, Any] | None:
            if tool_name in ("system_prompt_refresh", "memory") and service is not None:
                from ilma.core.retrieval import build_memory_block

                return {"memory_block": build_memory_block(service.memory)}
            return None

        ctx.register_hook("pre_tool_call", _pre_tool_call)

    logger.info("ilma: registered tools + memory override (provider=%s)", provider)


def _register_ilma_tools(ctx: Any, service: Any | None) -> None:
    if service is None:
        return

    tool_methods = _collect_tool_methods(service)
    for name, fn in tool_methods.items():
        ctx.register_tool(
            name=name,
            toolset="ilma",
            schema=_passthrough_schema(name, _docstring_or_default(fn, name)),
            handler=_bind(fn),
        )


def _collect_tool_methods(service: Any) -> dict[str, Callable]:
    """Collect public ilma_* methods on the service as tool candidates."""
    methods: dict[str, Callable] = {}
    for name in dir(service):
        if name.startswith("_") or not name.startswith("ilma_"):
            continue
        fn = getattr(service, name)
        if callable(fn):
            methods[name] = fn
    return methods


def _register_memory_override(ctx: Any, service: Any | None) -> None:

    def _memory_tool_wrapper(**kwargs: Any) -> str:
        action = kwargs.pop("action", None)
        if action is None:
            return json.dumps({"error": "validation_error", "message": "action is required"})

        if service is None:
            return _route_local_memory(action, kwargs)

        # Route to ilma service methods where possible.
        if action == "add":
            content = kwargs.get("content")
            if not content:
                return json.dumps(
                    {"error": "validation_error", "message": "add requires 'content'"}
                )
            result = service.ilma_remember(
                content,
                tags=kwargs.get("tags") or [],
                category=kwargs.get("category"),
                source=kwargs.get("source", "hermes-memory"),
            )
            return json.dumps(result)
        if action == "search":
            query = kwargs.get("query")
            if not query:
                return json.dumps(
                    {"error": "validation_error", "message": "search requires 'query'"}
                )
            result = service.ilma_search(query, top_k=kwargs.get("top_k", 10))
            return json.dumps(result)
        if action == "remove":
            memory_id = kwargs.get("memory_id")
            if memory_id is None:
                return json.dumps(
                    {"error": "validation_error", "message": "remove requires 'memory_id'"}
                )
            result = service.ilma_forget(memory_id)
            return json.dumps(result)
        if action == "list":
            result = service.ilma_status()
            return json.dumps(result)
        if action == "replace":
            memory_id = kwargs.get("memory_id")
            content = kwargs.get("content")
            if memory_id is None or not content:
                return json.dumps(
                    {
                        "error": "validation_error",
                        "message": "replace requires 'memory_id' and 'content'",
                    }
                )
            service.ilma_forget(memory_id)
            result = service.ilma_remember(
                content,
                tags=kwargs.get("tags") or [],
                category=kwargs.get("category"),
                source=kwargs.get("source", "hermes-memory"),
            )
            return json.dumps(result)
        return json.dumps({"error": "invalid_action", "message": f"unknown action: {action}"})

    ctx.register_tool(
        name="memory",
        toolset="ilma",
        schema=_passthrough_schema(
            "memory",
            "Override of the built-in memory tool. Backed by ilma "
            "when memory.provider=ilma. Accepts action=add|replace|remove|search|list.",
        ),
        handler=_memory_tool_wrapper,
        override=True,
    )
    logger.info("ilma: 'memory' tool override registered")


def _route_local_memory(action: str, kwargs: dict[str, Any]) -> str:
    """Graceful fallback when ilma service is unavailable."""
    local_path = os.environ.get(
        "MEMORY_LOCAL_PATH",
        os.path.expanduser("~/.hermes/memories/MEMORY.md"),
    )
    if action == "add":
        content = kwargs.get("content")
        if not content:
            return json.dumps({"error": "validation_error", "message": "add requires 'content'"})
        try:
            with open(local_path, "a") as f:
                f.write(f"- {content}\n")
            return json.dumps({"status": "stored", "path": local_path, "mode": "local"})
        except OSError as exc:
            return json.dumps({"error": "io_error", "message": str(exc)})
    if action == "search":
        query = kwargs.get("query")
        if not query:
            return json.dumps({"error": "validation_error", "message": "search requires 'query'"})
        try:
            with open(local_path) as f:
                lines = [line.strip() for line in f if query.lower() in line.lower()]
            top_k = kwargs.get("top_k", 10)
            return json.dumps(
                {"query": query, "count": len(lines[:top_k]), "results": lines[:top_k]}
            )
        except FileNotFoundError:
            return json.dumps({"query": query, "count": 0, "results": []})
    if action == "list":
        try:
            with open(local_path) as f:
                return json.dumps({"path": local_path, "lines": f.read().splitlines()})
        except FileNotFoundError:
            return json.dumps({"path": local_path, "lines": []})
    return json.dumps(
        {
            "error": "not_implemented_in_local",
            "message": f"action {action!r} not supported in local fallback",
        }
    )
