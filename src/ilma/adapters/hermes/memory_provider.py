"""Hermes MemoryProvider adapter for ilma.

This module exposes ilma through Hermes Agent's pluggable memory provider
interface.  The provider is intentionally thin: all actual storage, recall,
and tool behavior is delegated to :class:`ilma.service.IlmaService`.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover - exercised in a real Hermes runtime.
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - keeps ilma's standalone tests importable.

    class MemoryProvider:  # type: ignore[no-redef]
        """Small compatibility shim when Hermes Agent is not importable."""

        @property
        def name(self) -> str:
            raise NotImplementedError


from ilma.service import IlmaService, tools_dict
from ilma.storage.postgres import PgBackend

logger = logging.getLogger(__name__)

_DSN_ENV_VARS = ("ILMA_DSN", "PG_MEM_DB_CONN_STR", "HERMES_PG_CONN_STR")
_NON_PRIMARY_CONTEXTS = {"cron", "flush"}


class IlmaMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider backed by ilma's Postgres + vector service."""

    def __init__(self) -> None:
        self._service: Any | None = None
        self._hermes_home = ""
        self._platform = ""
        self._agent_context = "primary"
        self._session_id = ""

    @property
    def name(self) -> str:
        return "ilma"

    def is_available(self) -> bool:
        """Return True when ilma can be imported and a Postgres DSN is configured."""

        try:
            _ = IlmaService
            _ = PgBackend
        except Exception as exc:  # pragma: no cover - imports are module-level in tests.
            logger.debug("ilma memory provider unavailable: import failed: %s", exc)
            return False
        return bool(_resolve_dsn(self._hermes_home or None))

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        """Initialize the provider for a Hermes session."""

        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._platform = str(kwargs.get("platform") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary").lower()

        dsn = str(kwargs.get("dsn") or "").strip() or _resolve_dsn(self._hermes_home or None)
        self._service = _build_service(dsn=dsn, hermes_home=self._hermes_home or None)

    def system_prompt_block(self) -> str:
        return (
            "Memory is backed by ilma: persistent Postgres + vector recall is available "
            "through the `ilma_recall` and `ilma_remember` tools. Use recall for relevant "
            "past context and remember for durable facts or conversation turns."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn."""

        service = self._require_service()
        recall = getattr(service, "ilma_recall", None)
        if not callable(recall):
            return ""
        result = recall(query=query, limit=5)
        return _format_prefetch_context(result)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op; ilma recall is fast enough to run synchronously for now."""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a completed Hermes turn to ilma unless running in a non-primary context."""

        if self._agent_context in _NON_PRIMARY_CONTEXTS:
            return
        service = self._require_service()
        remember = service.ilma_remember
        source_session = session_id or self._session_id
        remember(
            content=f"User: {user_content}\nAssistant: {assistant_content}",
            category="turn",
            source=f"hermes:{source_session}",
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """No-op lifecycle hook."""

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """No-op lifecycle hook."""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Track the active Hermes session id for subsequent turn writes."""

        self._session_id = new_session_id

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all ilma tools."""

        service = self._require_service()
        schemas: list[dict[str, Any]] = []
        for name, spec in _service_tools(service).items():
            input_model = spec.get("input_schema") or spec.get("input_model")
            parameters = _model_json_schema(input_model)
            schemas.append(
                {
                    "name": name,
                    "description": str(spec.get("description") or f"ilma tool: {name}"),
                    "parameters": parameters,
                }
            )
        return schemas

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        """Dispatch an ilma_* tool call to the underlying IlmaService."""

        if not tool_name.startswith("ilma_"):
            return json.dumps(
                {"error": "invalid_tool", "message": f"Ilma cannot handle tool {tool_name!r}"}
            )
        service = self._require_service()
        method = getattr(service, tool_name, None)
        if not callable(method):
            return json.dumps(
                {"error": "unknown_tool", "message": f"Unknown ilma tool: {tool_name}"}
            )
        try:
            result = method(**(args or {}))
        except TypeError:
            # Some service registries expose handlers instead of direct methods;
            # fall back to the introspected handler so validation errors are structured.
            spec = _service_tools(service).get(tool_name)
            handler = spec.get("handler") if spec else None
            if not callable(handler):
                raise
            result = handler(**(args or {}))
        return json.dumps(_json_safe(result))

    def shutdown(self) -> None:
        """Release the service reference; ilma's shared pg pool owns DB cleanup."""

        self._service = None

    def _require_service(self) -> Any:
        if self._service is None:
            self.initialize(session_id=self._session_id, hermes_home=self._hermes_home)
        if self._service is None:  # pragma: no cover - defensive after initialize failure.
            raise RuntimeError("IlmaMemoryProvider is not initialized")
        return self._service


def _build_service(*, dsn: str, hermes_home: str | None = None) -> IlmaService:
    """Construct IlmaService across old and new constructor shapes."""

    signature = inspect.signature(IlmaService)
    kwargs: dict[str, Any] = {}
    if "dsn" in signature.parameters:
        kwargs["dsn"] = dsn or None
    if "hermes_home" in signature.parameters:
        kwargs["hermes_home"] = hermes_home
    if kwargs:
        return IlmaService(**kwargs)  # type: ignore[arg-type]

    if not dsn:
        msg = "set ILMA_DSN, PG_MEM_DB_CONN_STR, HERMES_PG_CONN_STR, or ilma config.yaml dsn"
        raise RuntimeError(msg)
    min_pool_size = int(os.environ.get("ILMA_PG_POOL_MIN", "1"))
    max_pool_size = int(os.environ.get("ILMA_PG_POOL_MAX", "8"))
    backend = PgBackend(dsn, min_pool_size=min_pool_size, max_pool_size=max_pool_size)
    return IlmaService(backend)


def _service_tools(service: Any) -> dict[str, dict[str, Any]]:
    service_tools = getattr(service, "tools_dict", None)
    if callable(service_tools):
        return service_tools()
    return tools_dict(service)


def _model_json_schema(input_model: Any) -> dict[str, Any]:
    if input_model is None:
        return {"type": "object", "properties": {}, "required": []}
    if hasattr(input_model, "model_json_schema"):
        schema = input_model.model_json_schema()
    elif hasattr(input_model, "schema"):
        schema = input_model.schema()
    elif isinstance(input_model, dict):
        schema = dict(input_model)
    else:
        schema = {"type": "object", "properties": {}, "required": []}
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return schema


def _resolve_dsn(hermes_home: str | None = None) -> str:
    for env_var in _DSN_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value

    for path in _config_paths(hermes_home):
        value = _dsn_from_config(path)
        if value:
            return value
    return ""


def _config_paths(hermes_home: str | None = None) -> list[Path]:
    paths: list[Path] = []
    if hermes_home:
        paths.append(Path(hermes_home).expanduser() / "ilma" / "config.yaml")
        paths.append(Path(hermes_home).expanduser() / "config.yaml")
    paths.append(Path("~/.config/ilma/config.yaml").expanduser())
    return paths


def _dsn_from_config(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - config parse failure should mean unavailable.
        logger.debug("failed reading ilma config %s: %s", path, exc)
        return ""
    if not isinstance(data, dict):
        return ""
    candidates = (
        data.get("dsn"),
        data.get("database_url"),
        data.get("postgres_dsn"),
        data.get("connection_string"),
    )
    postgres = data.get("postgres")
    if isinstance(postgres, dict):
        candidates = (
            *candidates,
            postgres.get("dsn"),
            postgres.get("database_url"),
            postgres.get("connection_string"),
        )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _format_prefetch_context(result: Any) -> str:
    if not result:
        return ""
    items = _extract_results(result)
    if not items:
        return ""
    lines = ["ilma recalled memory context:"]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {_format_item(item)}")
    return "\n".join(lines)


def _extract_results(result: Any) -> list[Any]:
    if isinstance(result, dict):
        raw = result.get("results") or result.get("memories") or result.get("matches")
        if raw is None and result.get("ok") is not False:
            raw = result.get("memory") or result.get("result")
    else:
        raw = result
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _format_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "document", "body", "summary"):
            value = item.get(key)
            if value:
                return str(value)
        return json.dumps(_json_safe(item), sort_keys=True)
    content = getattr(item, "content", None)
    if content:
        return str(content)
    text = getattr(item, "text", None)
    if text:
        return str(text)
    return str(item)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            return _json_safe(value.model_dump())
        if hasattr(value, "__dict__"):
            return _json_safe(vars(value))
        return str(value)
