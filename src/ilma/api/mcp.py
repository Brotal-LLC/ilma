"""MCP server for ilma.

This module is intentionally framework-agnostic: it imports only ilma core/storage
code and the official Python MCP SDK.  It exposes all persistent ilma surfaces as
MCP tools and returns structured success/error dictionaries so client tool calls do
not crash the server on expected repository errors.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any

from psycopg.types.json import Jsonb

from ilma.service import (
    TOOL_COUNT,
    WRITE_TOOLS,
    IlmaConfigError,
    _dsn_from_env,
    _json_safe,
    tools_dict,
)
from ilma.service import (
    IlmaService as IlmaMcpService,
)
from ilma.storage.postgres import PgBackend

try:  # Imported at module load so the entry point fails clearly if MCP is absent.
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised by packaging/import environments.
    raise RuntimeError(
        "The official Python MCP SDK is required. Install ilma with the 'mcp' extra "
        "or install the 'mcp' package."
    ) from exc


class PgAuditLogger:
    """Write-ahead audit logger stored in Postgres.

    A pending row is inserted before each write tool executes; it is then marked
    succeeded/failed after the write attempt.  If insertion fails, the write is not
    performed, preserving write-ahead semantics.
    """

    def __init__(self, backend: PgBackend) -> None:
        self._backend = backend

    def initialize_schema(self) -> None:
        with self._backend._pool.connection() as connection:  # noqa: SLF001 - backend owns pool.
            connection.execute("CREATE SCHEMA IF NOT EXISTS ilma")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ilma.audit_log (
                    id bigserial PRIMARY KEY,
                    operation_id text NOT NULL UNIQUE,
                    tool_name text NOT NULL,
                    surface text NOT NULL,
                    action text NOT NULL,
                    payload jsonb NOT NULL DEFAULT '{}',
                    status text NOT NULL DEFAULT 'pending',
                    error_type text,
                    error_message text,
                    result jsonb,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    completed_at timestamptz
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS audit_log_tool_created_idx "
                "ON ilma.audit_log(tool_name, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS audit_log_status_idx ON ilma.audit_log(status)"
            )

    def begin(self, tool_name: str, surface: str, action: str, payload: dict[str, Any]) -> str:
        self.initialize_schema()
        operation_id = str(uuid.uuid4())
        with self._backend._pool.connection() as connection:  # noqa: SLF001 - backend owns pool.
            connection.execute(
                """
                INSERT INTO ilma.audit_log (operation_id, tool_name, surface, action, payload, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                """,
                (operation_id, tool_name, surface, action, Jsonb(_json_safe(payload))),
            )
        return operation_id

    def finish(
        self,
        operation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self._backend._pool.connection() as connection:  # noqa: SLF001 - backend owns pool.
            connection.execute(
                """
                UPDATE ilma.audit_log
                SET status = %s,
                    result = %s,
                    error_type = %s,
                    error_message = %s,
                    completed_at = now()
                WHERE operation_id = %s
                """,
                (
                    status,
                    Jsonb(_json_safe(result or {})),
                    type(error).__name__ if error else None,
                    str(error) if error else None,
                    operation_id,
                ),
            )


class InMemoryAuditLogger:
    """Small audit logger used by unit tests and non-Postgres fakes."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def initialize_schema(self) -> None:
        return None

    def begin(self, tool_name: str, surface: str, action: str, payload: dict[str, Any]) -> str:
        operation_id = str(uuid.uuid4())
        self.records.append(
            {
                "operation_id": operation_id,
                "tool_name": tool_name,
                "surface": surface,
                "action": action,
                "payload": _json_safe(payload),
                "status": "pending",
            }
        )
        return operation_id

    def finish(
        self,
        operation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        for record in self.records:
            if record["operation_id"] == operation_id:
                record["status"] = status
                record["result"] = _json_safe(result or {})
                record["error_type"] = type(error).__name__ if error else None
                record["error_message"] = str(error) if error else None
                return


_SERVICE: IlmaMcpService | None = None


def get_service() -> IlmaMcpService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = IlmaMcpService.from_env()
    return _SERVICE


def set_service(service: IlmaMcpService | None) -> None:
    """Override the process service for tests/embedding."""

    global _SERVICE
    _SERVICE = service


def create_mcp_server(service: IlmaMcpService | None = None) -> FastMCP:
    """Create and register the official MCP SDK server.

    All tool registrations are driven by ``tools_dict(service)``, which
    introspects the service class. To add a new MCP tool, add a public
    ``ilma_*`` method to ``IlmaService`` — registration is automatic.

    The dispatch glue is a tiny function built via ``exec()`` whose
    signature matches the service method's. FastMCP's
    ``func_metadata`` inspects that signature and produces a correct
    Pydantic input model (one field per parameter, ``required`` set
    from non-defaulted parameters). The function body forwards the
    call to ``spec['handler']``, which is the audit + metrics
    wrapper produced by ``_make_tool_handler``.
    """
    if service is not None:
        set_service(service)

    active_service = service if service is not None else get_service()
    server = FastMCP("ilma")

    for tool_name, spec in tools_dict(active_service).items():
        method = getattr(active_service, tool_name)
        sig = inspect.signature(method)
        # Drop ``self`` — the service is bound to ``active_service`` and
        # the underlying method does not take a self argument at call
        # time. Including it would force the Pydantic arg model to
        # declare a ``self`` field, which would surprise callers.
        params = [p for p in sig.parameters.values() if p.name != "self"]
        # Build the dispatch wrapper at runtime with the exact signature
        # of the service method, so FastMCP's func_metadata produces a
        # matching Pydantic input model. Annotations on the service
        # method are stored as strings (the file uses
        # ``from __future__ import annotations``); we paste them as
        # forward-reference strings into the generated function, and
        # pre-populate the exec namespace with the typing names the
        # strings reference so ``eval_str=True`` can resolve them.
        #
        # Critically, defaults must be embedded in the *signature* of
        # the generated function (not just the body), because
        # ``func_metadata`` reads parameter defaults from the
        # signature to decide which Pydantic fields are required.
        params_src = ", ".join(_param_decl(p) for p in params)
        ns: dict[str, Any] = {"_handler": spec["handler"]}
        for p in params:
            if p.annotation is inspect.Parameter.empty or not isinstance(p.annotation, str):
                continue
            for name in _collect_typing_names(p.annotation):
                # Skip builtin names — they're already in __builtins__
                # and pre-populating them with ``None`` (typing has no
                # ``str``/``int``/``float``/``bool``/``bytes``) would
                # shadow the real types and break PEP 604 unions.
                if name in _BUILTIN_NAMES:
                    continue
                ns.setdefault(name, _lookup_typing_name(name))
        body_src = (
            "def _dispatch(" + params_src + "):\n"
            "    return _handler(" + ", ".join(f"{p.name}={p.name}" for p in params) + ")\n"
        )
        exec(body_src, ns)
        dispatch_fn = ns["_dispatch"]
        dispatch_fn.__name__ = tool_name
        dispatch_fn.__doc__ = spec["description"]
        # The exec'd function already has the right __annotations__ from
        # the source — don't reassign, or we risk re-introducing string
        # annotations that re-trigger the eval_str path incorrectly.
        server.add_tool(
            dispatch_fn,
            name=tool_name,
            description=spec["description"],
        )

    return server


# Names that the typing module does NOT export as attributes (they're
# builtins). Pre-populating these into the exec namespace with ``None``
# would shadow the real types and break annotation evaluation.
_BUILTIN_NAMES = frozenset(
    {"str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple", "frozenset", "type"}
)


def _collect_typing_names(ann_str: str) -> set[str]:
    """Pull capitalized identifiers out of an annotation string.

    Recognizes ``Any``, ``Optional``, ``list``, ``dict``, ``str``, ``int``,
    ``float``, ``bool``, ``bytes``, ``None``, plus anything wrapped in
    ``typing.`` prefix.  These are the names the exec'd function needs
    in its namespace to evaluate annotations with ``eval_str=True``.
    """
    candidates = {
        "Any",
        "Optional",
        "Union",
        "List",
        "Dict",
        "Tuple",
        "Set",
        "FrozenSet",
        "Sequence",
        "Mapping",
        "Iterable",
        "Iterator",
        "Callable",
        "Type",
        "ClassVar",
        "Literal",
        "Final",
        "Annotated",
        "TypeVar",
        "Generic",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "int",
        "float",
        "bool",
        "str",
        "bytes",
        # Note: "None" is intentionally excluded — it's a built-in literal,
        # not a typing object, and pre-populating it would shadow the
        # built-in and break annotation evaluation (Optional[None]).
    }
    found: set[str] = set()
    for name in candidates:
        # Word-boundary match so 'str' doesn't match inside 'strong'.
        import re

        if re.search(rf"\b{re.escape(name)}\b", ann_str):
            found.add(name)
    return found


def _lookup_typing_name(name: str) -> Any:
    """Resolve a typing name to the actual object for the exec namespace."""
    import typing

    return getattr(typing, name, None)


def _kwargs_src(sig: inspect.Signature) -> str:
    """Build the keyword-arg expansion source for a method's signature.

    Includes defaults so the constructed call matches the method's
    declared defaults. Example: 'query, top_k=top_k, hybrid_text_weight=hybrid_text_weight'.
    """
    parts: list[str] = []
    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={p.name}")
    return ", ".join(parts)


def _param_decl(p: inspect.Parameter) -> str:
    """Render a single ``inspect.Parameter`` as a function-parameter declaration.

    Includes the annotation (as a forward-reference string when the
    service method uses ``from __future__ import annotations``) and
    the default if one was declared. Without the default in the
    generated signature, FastMCP's ``func_metadata`` would mark every
    parameter as required, breaking optional-arg tool calls.
    """
    if p.annotation is inspect.Parameter.empty:
        ann = ""
    elif isinstance(p.annotation, str):
        ann = f": {p.annotation}"
    else:
        ann = f": {p.annotation!s}"
    if p.default is inspect.Parameter.empty:
        return f"{p.name}{ann}"
    # Repr the default so strings stay quoted and booleans render as
    # ``True``/``False`` rather than ``1``/``0``.  ``None`` and numbers
    # round-trip fine.
    return f"{p.name}{ann}={p.default!r}"


def main() -> None:
    """Console entry point for ``ilma-mcp``."""

    create_mcp_server().run()


__all__ = [
    "IlmaConfigError",
    "IlmaMcpService",
    "InMemoryAuditLogger",
    "PgAuditLogger",
    "TOOL_COUNT",
    "WRITE_TOOLS",
    "_dsn_from_env",
    "create_mcp_server",
    "get_service",
    "main",
    "set_service",
]
