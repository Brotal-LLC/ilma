"""Shared service layer and introspection helpers for ilma entry points."""

from __future__ import annotations

import ast
import inspect
import os
import re
import textwrap
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, cast, get_type_hints

from psycopg.rows import dict_row
from pydantic import BaseModel, Field, ValidationError, create_model

from ilma.api.hardening import METRICS, log_observation
from ilma.embeddings import SUPPORTED_DIMS
from ilma.storage.postgres import PgBackend

TOOLS_ATTR_PREFIX = "ilma_"
EXCLUDED_TOOL_NAMES = frozenset({"ilma_audit", "ilma_health"})


def method_description(method: Callable[..., Any]) -> str:
    """Return the first docstring line for a service method."""

    doc = inspect.getdoc(method)
    if not doc:
        return f"ilma: {method.__name__}"
    return doc.strip().split("\n", 1)[0]


def method_to_pydantic_model(method: Callable[..., Any]) -> type[BaseModel]:
    """Build a Pydantic argument model from a service method signature."""

    signature = inspect.signature(method)
    type_hints = get_type_hints(method)
    field_descriptions = _field_descriptions(method)
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        annotation = type_hints.get(name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = parameter.default
        if default is inspect.Parameter.empty:
            default = ...
        fields[name] = (
            annotation,
            Field(default=default, description=field_descriptions.get(name, "")),
        )
    model = create_model(f"{method.__name__}_Args", **fields)
    model.__doc__ = method_description(method)
    return model


def tools_dict(service: Any) -> dict[str, dict[str, Any]]:
    """Build a registry for public MCP tools exposed by a service instance."""

    result: dict[str, dict[str, Any]] = {}
    for name in dir(service):
        if name in EXCLUDED_TOOL_NAMES or not name.startswith(TOOLS_ATTR_PREFIX):
            continue
        method = getattr(service, name)
        if not callable(method):
            continue
        model = method_to_pydantic_model(method)
        result[name] = {
            "description": method_description(method),
            "input_model": model,
            "handler": _make_tool_handler(service, name, method, model),
        }
    return result


def _make_tool_handler(
    service: Any, name: str, method: Callable[..., dict[str, Any]], model: type[BaseModel]
) -> Callable[..., dict[str, Any]]:
    """Build a tool-call handler for a service method.

    The returned callable accepts keyword arguments (matching the
    method's parameter names) and returns a result dict. Validation
    against the Pydantic model happens at the boundary so callers
    get a structured error rather than a 500.

    The handler does **not** wrap the dispatch in ``service.call`` —
    the service method is responsible for its own audit + metrics
    wrapping (see e.g. ``IlmaService.ilma_remember`` which already
    calls ``self.call(...)``). Wrapping here would double-fire the
    audit log.
    """

    def handler(**kwargs: Any) -> dict[str, Any]:
        try:
            model_instance = model(**kwargs)
        except ValidationError as exc:
            return {"error": "validation_error", "details": exc.errors()}
        payload = model_instance.model_dump()
        try:
            result = method(**payload)
        except Exception as exc:  # noqa: BLE001 - dispatch must not crash MCP.
            return {"error": "dispatch_error", "message": str(exc)}
        return cast(dict[str, Any], _json_safe(result))

    return handler


def _field_descriptions(method: Callable[..., Any]) -> dict[str, str]:
    doc = inspect.getdoc(method) or ""
    if not doc:
        return {}
    descriptions: dict[str, str] = {}
    descriptions.update(_parse_google_args(doc))
    descriptions.update(_parse_sphinx_params(doc))
    return descriptions


def _parse_google_args(doc: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    lines = doc.splitlines()
    in_args = False
    current_name: str | None = None
    current_indent = 0
    for line in lines:
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            current_name = None
            continue
        if not in_args:
            continue
        if stripped and not line.startswith((" ", "\t")):
            break
        match = re.match(
            r"^(?P<indent>\s+)(?P<name>[A-Za-z_]\w*)(?:\s*\([^)]*\))?:\s*(?P<desc>.*)$", line
        )
        if match:
            current_name = match.group("name")
            current_indent = len(match.group("indent"))
            descriptions[current_name] = match.group("desc").strip()
            continue
        if current_name and stripped and len(line) - len(line.lstrip()) > current_indent:
            descriptions[current_name] = f"{descriptions[current_name]} {stripped}".strip()
    return descriptions


def _parse_sphinx_params(doc: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    current_name: str | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        match = re.match(r"^:param\s+(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<desc>.*)$", stripped)
        if match:
            current_name = match.group("name")
            descriptions[current_name] = match.group("desc").strip()
            continue
        if current_name and stripped and not stripped.startswith(":"):
            descriptions[current_name] = f"{descriptions[current_name]} {stripped}".strip()
        elif stripped.startswith(":"):
            current_name = None
    return descriptions


SURFACE_TABLES: Mapping[str, tuple[str, ...]] = {
    "memory": ("memories", "memory_chunks"),
    "wiki": ("wiki_docs", "wiki_chunks"),
    "journal": ("journal_entries",),
    "skills": ("skills",),
    "kanban": ("kanban_tasks",),
    "metrics": ("metrics",),
    "observability": ("observations",),
    "sessions": ("sessions", "session_messages"),
}

FTS_INDEXES = (
    "memories_content_tsv_idx",
    "memory_chunks_content_tsv_idx",
    "wiki_docs_content_tsv_idx",
    "wiki_chunks_content_tsv_idx",
    "journal_entries_content_tsv_idx",
    "skills_body_tsv_idx",
    "kanban_tasks_content_tsv_idx",
    "session_messages_content_tsv_idx",
)


class IlmaConfigError(RuntimeError):
    """Raised when the MCP server cannot determine its Postgres configuration."""


def _dsn_from_env(env: Mapping[str, str] | None = None) -> str:
    """Resolve the Postgres DSN from environment variables.

    ILMA_DSN is canonical. PG_MEM_DB_CONN_STR is supported for migration
    compatibility with older installs.
    """

    source = env if env is not None else os.environ
    dsn = (source.get("ILMA_DSN") or source.get("PG_MEM_DB_CONN_STR") or "").strip()
    if not dsn:
        raise IlmaConfigError("set ILMA_DSN or PG_MEM_DB_CONN_STR to a Postgres DSN")
    return dsn


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _json_safe(value: Any) -> Any:
    """Convert dataclasses, datetimes, tuples, and mappings to JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple | list | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **_json_safe(payload)}


def _err(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _limit(value: int, *, default: int, maximum: int = 500) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def _offset(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


class IlmaService:
    """Tool implementation layer used by the MCP SDK registration and tests."""

    def __init__(self, backend: Any, audit: Any | None = None) -> None:
        self.backend = backend
        if audit is None:
            from ilma.api.mcp import PgAuditLogger

            audit = PgAuditLogger(backend)
        self.audit = audit
        self.memory = backend.memory_repo()
        self.wiki = backend.wiki_repo()
        self.journal = backend.journal_repo()
        self.skills = backend.skills_repo()
        self.kanban = backend.kanban_repo()
        self.metrics = backend.metrics_repo()
        self.observability = backend.observability_repo()
        self.sessions = backend.sessions_repo()

    @classmethod
    def from_env(cls) -> IlmaService:
        dsn = _dsn_from_env()
        min_pool_size = int(os.environ.get("ILMA_PG_POOL_MIN", "1"))
        max_pool_size = int(os.environ.get("ILMA_PG_POOL_MAX", "8"))
        return cls(PgBackend(dsn, min_pool_size=min_pool_size, max_pool_size=max_pool_size))

    def call(
        self, tool_name: str, fn: Callable[[], dict[str, Any]], payload: dict[str, Any]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        operation_id: str | None = None
        success = False
        error: BaseException | None = None
        try:
            if tool_name in WRITE_TOOLS:
                surface, action = WRITE_TOOLS[tool_name]
                try:
                    operation_id = self.audit.begin(tool_name, surface, action, payload)
                except Exception as exc:  # audit is write-ahead: do not execute write if it fails.
                    error = exc
                    return _err(exc)
            try:
                result = fn()
            except Exception as exc:  # Never crash MCP server for repository/tool errors.
                error = exc
                if operation_id is not None:
                    self._safe_audit_finish(operation_id, status="failed", error=exc)
                return _err(exc)
            if operation_id is not None:
                self._safe_audit_finish(operation_id, status="succeeded", result=result)
            success = True
            return cast(dict[str, Any], _json_safe(result))
        finally:
            self._record_tool_call(
                tool_name,
                duration_seconds=time.perf_counter() - started,
                success=success,
                error=error,
            )

    def _record_tool_call(
        self,
        tool_name: str,
        *,
        duration_seconds: float,
        success: bool,
        error: BaseException | None = None,
    ) -> None:
        labels = {"tool_name": tool_name, "success": str(success).lower()}
        METRICS.increment("tool_call_count", labels)
        METRICS.observe("tool_call_duration", duration_seconds, labels)
        if tool_name == "ilma_recall":
            METRICS.observe(
                "memory_search_latency", duration_seconds, {"success": str(success).lower()}
            )
        log_observation(
            self.observability,
            level="info" if success else "error",
            message="mcp tool call",
            source="mcp.tool",
            context={
                "tool_name": tool_name,
                "duration_ms": round(duration_seconds * 1000, 3),
                "success": success,
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
            },
        )

    def _safe_audit_finish(
        self,
        operation_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        try:
            self.audit.finish(operation_id, status=status, result=result, error=error)
        except Exception:
            traceback.print_exc()

    def ilma_health(self) -> dict[str, Any]:
        """Run structured HTTP health checks without registering another MCP tool."""

        checks: dict[str, Any] = {}
        backend_health: dict[str, Any]
        try:
            backend_health = self.backend.health()
            checks["postgres"] = {
                "ok": bool(backend_health.get("ok", False)),
                "database": backend_health.get("database"),
            }
            checks["pgvector"] = {"ok": bool(backend_health.get("pgvector", False))}
        except Exception as exc:
            backend_health = {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            checks["postgres"] = backend_health
            checks["pgvector"] = {"ok": False, "skipped": "postgres check failed"}

        checks["embedder"] = self._check_embedder()
        ok = all(bool(check.get("ok", False)) for check in checks.values())
        return _ok(
            ok=ok, status="healthy" if ok else "unhealthy", backend=backend_health, checks=checks
        )

    def _check_embedder(self) -> dict[str, Any]:
        embedder_registry = getattr(self.memory, "_embedders", None)
        if embedder_registry is None:
            embedder_registry = getattr(self.wiki, "_embedders", None)
        if embedder_registry is None:
            return {
                "ok": True,
                "configured": False,
                "skipped": "embedder unavailable on repository",
            }
        dim = int(
            getattr(embedder_registry, "default_dim", getattr(self.memory, "default_dim", 0) or 0)
        )
        try:
            embed = embedder_registry.embed
            if dim > 0:
                vector = embed("ilma health embedder reachability check", dim=dim)
            else:
                vector = embed("ilma health embedder reachability check")
            return {
                "ok": True,
                "configured": True,
                "default_dim": dim,
                "vector_length": len(vector),
            }
        except Exception as exc:
            return {
                "ok": False,
                "configured": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    # Memory surface -----------------------------------------------------
    def ilma_status(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            backend_health: dict[str, Any]
            try:
                backend_health = self.backend.health()
            except Exception as exc:
                backend_health = {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            return _ok(
                backend=backend_health,
                memory=self.memory.status(),
                surfaces=[
                    "memory",
                    "wiki",
                    "journal",
                    "skills",
                    "metrics",
                    "kanban",
                    "observability",
                    "sessions",
                ],
                tool_count=TOOL_COUNT,
            )

        return self.call("ilma_status", run, {})

    def ilma_recall(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        hybrid_text_weight: float = 0.5,
    ) -> dict[str, Any]:
        """Recall memories relevant to a query.

        Canonical surface for memory recall. Returns ``{"results": [...]}``
        where each result is a memory row ordered by relevance.

        Parameters
        ----------
        query:
            Natural-language query to match against stored memories.
        limit:
            Maximum number of results to return (1-100, default 10).
        threshold:
            Minimum cosine similarity for a result to be included
            (0.0 = no filter, default 0.0). Filters are applied client-side
            after the underlying hybrid search.
        hybrid_text_weight:
            Weight given to keyword/lexical matches vs vector matches
            (0.0 = pure vector, 1.0 = pure keyword, default 0.5).
        """

        def run() -> dict[str, Any]:
            capped_limit = _limit(limit, default=10)
            raw = self.memory.search(
                query,
                top_k=capped_limit,
                hybrid_text_weight=hybrid_text_weight,
            )
            results = self._filter_by_threshold(raw, threshold)
            return _ok(results=results, count=len(results), query=query, limit=capped_limit)

        return self.call(
            "ilma_recall",
            run,
            {
                "query": query,
                "limit": limit,
                "threshold": threshold,
                "hybrid_text_weight": hybrid_text_weight,
            },
        )

    def ilma_recent(self, limit: int = 10) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            repo_recent = getattr(self.memory, "recent", None)
            if callable(repo_recent):
                return _ok(results=repo_recent(limit=_limit(limit, default=10)))
            return _ok(results=self._memory_rows(limit=_limit(limit, default=10), offset_value=0))

        return self.call("ilma_recent", run, {"limit": limit})

    def ilma_get_memory(self, memory_id: int) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            repo_get = getattr(self.memory, "get", None)
            if callable(repo_get):
                memory = repo_get(int(memory_id))
                return _ok(memory=memory)
            rows = self._memory_rows(memory_id=int(memory_id), include_deleted=True)
            return _ok(memory=rows[0] if rows else None)

        return self.call("ilma_get_memory", run, {"memory_id": memory_id})

    def ilma_list_memories(
        self, limit: int = 50, offset: int = 0, include_deleted: bool = False
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            repo_list = getattr(self.memory, "list", None)
            if callable(repo_list):
                return _ok(
                    results=repo_list(
                        limit=_limit(limit, default=50),
                        offset=_offset(offset),
                        include_deleted=bool(include_deleted),
                    )
                )
            return _ok(
                results=self._memory_rows(
                    limit=_limit(limit, default=50),
                    offset_value=_offset(offset),
                    include_deleted=bool(include_deleted),
                )
            )

        return self.call(
            "ilma_list_memories",
            run,
            {"limit": limit, "offset": offset, "include_deleted": include_deleted},
        )

    def ilma_remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = "mcp",
    ) -> dict[str, Any]:
        return self.call(
            "ilma_remember",
            lambda: _ok(
                memory_id=self.memory.remember(
                    content, tags=tags or [], category=category, source=source
                )
            ),
            {"content": content, "tags": tags or [], "category": category, "source": source},
        )

    def ilma_forget(self, memory_id: int) -> dict[str, Any]:
        return self.call(
            "ilma_forget",
            lambda: _ok(deleted=self.memory.forget(int(memory_id))),
            {"memory_id": memory_id},
        )

    # Wiki surface -------------------------------------------------------
    def ilma_get_wiki(self, slug: str) -> dict[str, Any]:
        return self.call("ilma_get_wiki", lambda: _ok(document=self.wiki.get(slug)), {"slug": slug})

    def ilma_wiki_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self.call(
            "ilma_wiki_search",
            lambda: _ok(results=self.wiki.search(query, top_k=_limit(top_k, default=5))),
            {"query": query, "top_k": top_k},
        )

    def ilma_list_wiki(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            repo_list = getattr(self.wiki, "list", None)
            if callable(repo_list):
                return _ok(
                    results=repo_list(limit=_limit(limit, default=50), offset=_offset(offset))
                )
            return _ok(
                results=self._wiki_rows(
                    limit=_limit(limit, default=50), offset_value=_offset(offset)
                )
            )

        return self.call("ilma_list_wiki", run, {"limit": limit, "offset": offset})

    def ilma_wiki_create(
        self,
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        return self.call(
            "ilma_wiki_create",
            lambda: _ok(
                **self.wiki.ingest(
                    slug, title, body_md, category=category, tags=tags or [], source_uri=source_uri
                )
            ),
            {
                "slug": slug,
                "title": title,
                "body_md": body_md,
                "category": category,
                "tags": tags or [],
                "source_uri": source_uri,
            },
        )

    def ilma_wiki_update(
        self,
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        # The repository's ingest method is upsert/versioned, so update shares implementation.
        return self.call(
            "ilma_wiki_update",
            lambda: _ok(
                **self.wiki.ingest(
                    slug, title, body_md, category=category, tags=tags or [], source_uri=source_uri
                )
            ),
            {
                "slug": slug,
                "title": title,
                "body_md": body_md,
                "category": category,
                "tags": tags or [],
                "source_uri": source_uri,
            },
        )

    # Journal surface ----------------------------------------------------
    def ilma_journal_search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return self.call(
            "ilma_journal_search",
            lambda: _ok(results=self.journal.search(query, top_k=_limit(top_k, default=10))),
            {"query": query, "top_k": top_k},
        )

    def ilma_journal_recent(self, limit: int = 10) -> dict[str, Any]:
        return self.call(
            "ilma_journal_recent",
            lambda: _ok(results=self.journal.recent(limit=_limit(limit, default=10))),
            {"limit": limit},
        )

    # Skills surface -----------------------------------------------------
    def ilma_skills_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self.call(
            "ilma_skills_search",
            lambda: _ok(results=self.skills.search(query, top_k=_limit(top_k, default=5))),
            {"query": query, "top_k": top_k},
        )

    def ilma_skills_get(self, name: str) -> dict[str, Any]:
        return self.call(
            "ilma_skills_get", lambda: _ok(skill=self.skills.get(name)), {"name": name}
        )

    # Kanban surface -----------------------------------------------------
    def ilma_kanban_list(self, status: str = "todo", limit: int = 50) -> dict[str, Any]:
        return self.call(
            "ilma_kanban_list",
            lambda: _ok(
                results=self.kanban.list_by_status(status, limit=_limit(limit, default=50))
            ),
            {"status": status, "limit": limit},
        )

    def ilma_kanban_get(self, task_id: int) -> dict[str, Any]:
        return self.call(
            "ilma_kanban_get",
            lambda: _ok(task=self.kanban.get(int(task_id))),
            {"task_id": task_id},
        )

    def ilma_kanban_create(
        self,
        title: str,
        description: str = "",
        status: str = "todo",
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        return self.call(
            "ilma_kanban_create",
            lambda: _ok(
                task_id=self.kanban.create(
                    title,
                    description=description,
                    status=status,
                    priority=int(priority),
                    tags=tags or [],
                    parent_id=parent_id,
                )
            ),
            {
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "tags": tags or [],
                "parent_id": parent_id,
            },
        )

    def ilma_kanban_update(
        self,
        task_id: int,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        tags: list[str] | None = None,
        parent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            updates: dict[str, Any] = {}
            for key, value in {
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "tags": tags,
                "parent_id": parent_id,
                "metadata": metadata,
            }.items():
                if value is not None:
                    updates[key] = value
            return _ok(updated=self.kanban.update(int(task_id), **updates))

        return self.call(
            "ilma_kanban_update",
            run,
            {
                "task_id": task_id,
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "tags": tags,
                "parent_id": parent_id,
                "metadata": metadata,
            },
        )

    def ilma_kanban_complete(self, task_id: int) -> dict[str, Any]:
        return self.call(
            "ilma_kanban_complete",
            lambda: _ok(completed=self.kanban.complete(int(task_id))),
            {"task_id": task_id},
        )

    # Metrics surface ----------------------------------------------------
    def ilma_metrics_record(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return self.call(
            "ilma_metrics_record",
            lambda: _ok(metric_id=self.metrics.record(name, float(value), labels=labels or {})),
            {"name": name, "value": value, "labels": labels or {}},
        )

    def ilma_metrics_query(
        self,
        name: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        aggregate_window: str | None = None,
    ) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            if aggregate_window:
                return _ok(
                    results=self.metrics.aggregate(name, window=aggregate_window), aggregate=True
                )
            return _ok(
                results=self.metrics.query(
                    name,
                    start=_parse_datetime(start),
                    end=_parse_datetime(end),
                    limit=_limit(limit, default=100),
                ),
                aggregate=False,
            )

        return self.call(
            "ilma_metrics_query",
            run,
            {
                "name": name,
                "start": start,
                "end": end,
                "limit": limit,
                "aggregate_window": aggregate_window,
            },
        )

    # Observability surface ---------------------------------------------
    def ilma_obs_log(
        self,
        level: str,
        message: str,
        source: str | None = "mcp",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.call(
            "ilma_obs_log",
            lambda: _ok(
                observation_id=self.observability.log(
                    level, message, source=source, context=context or {}
                )
            ),
            {"level": level, "message": message, "source": source, "context": context or {}},
        )

    def ilma_obs_query(
        self,
        level: str | None = None,
        source: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self.call(
            "ilma_obs_query",
            lambda: _ok(
                results=self.observability.query(
                    level=level,
                    source=source,
                    start=_parse_datetime(start),
                    end=_parse_datetime(end),
                    limit=_limit(limit, default=100),
                )
            ),
            {"level": level, "source": source, "start": start, "end": end, "limit": limit},
        )

    # Sessions surface ---------------------------------------------------
    def ilma_session_search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        return self.call(
            "ilma_session_search",
            lambda: _ok(results=self.sessions.search(query, top_k=_limit(top_k, default=10))),
            {"query": query, "top_k": top_k},
        )

    def ilma_session_get(self, session_id: str, limit: int = 100) -> dict[str, Any]:
        return self.call(
            "ilma_session_get",
            lambda: _ok(
                messages=self.sessions.get_session(session_id, limit=_limit(limit, default=100))
            ),
            {"session_id": session_id, "limit": limit},
        )

    # Maintenance --------------------------------------------------------
    def ilma_repair(self, force: bool = False) -> dict[str, Any]:
        """Inspect storage damage and optionally repair safe Postgres issues."""

        def run() -> dict[str, Any]:
            if not force:
                findings = self._repair_findings()
                return _ok(
                    repaired=False,
                    force=False,
                    findings=findings,
                    message="repair dry-run complete; rerun with --force to rebuild indexes/vacuum",
                )

            self._initialize_all()
            findings = self._repair_findings()
            actions = self._repair_apply(findings)
            post_findings = self._repair_findings()
            return _ok(
                repaired=True,
                force=True,
                findings=post_findings,
                pre_repair_findings=findings,
                actions=actions,
                message="repair complete",
            )

        return self.call("ilma_repair", run, {"force": force})

    def ilma_doctor(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            checks: dict[str, Any] = {
                "dsn_configured": True,
                "backend": self._doctor_backend(),
                "pgvector": self._doctor_pgvector(),
                "pool": self._doctor_pool(),
                "embedding_dimensions": self._doctor_embedding_dimensions(),
                "surfaces": self._doctor_surfaces(),
                "audit_log": self._doctor_audit_log(),
            }
            overall = all(
                bool(check.get("ok", False))
                for check in (
                    checks["backend"],
                    checks["pgvector"],
                    checks["pool"],
                    checks["embedding_dimensions"],
                    checks["audit_log"],
                    *checks["surfaces"].values(),
                )
            )
            return _ok(healthy=overall, checks=checks)

        return self.call("ilma_doctor", run, {})

    def ilma_audit(
        self,
        *,
        tool: str | None = None,
        status: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query the audit log, with Postgres and in-memory audit support."""

        return self.call(
            "ilma_audit",
            lambda: _ok(
                results=self._audit_rows(
                    tool=tool,
                    status=status,
                    start=_parse_datetime(start),
                    end=_parse_datetime(end),
                    limit=_limit(limit, default=100, maximum=1000),
                    offset_value=_offset(offset),
                )
            ),
            {
                "tool": tool,
                "status": status,
                "start": start,
                "end": end,
                "limit": limit,
                "offset": offset,
            },
        )

    def ilma_migrate(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._initialize_all()
            return _ok(migrated=True, surfaces=8, audit_log=True)

        return self.call("ilma_migrate", run, {})

    def _has_pg_pool(self) -> bool:
        return hasattr(self.backend, "_pool")

    def _repair_findings(self) -> dict[str, Any]:
        if not self._has_pg_pool():
            return {
                "postgres": {"ok": False, "skipped": "backend does not expose a Postgres pool"},
                "orphaned_chunks": {"count": 0, "rows": [], "skipped": True},
                "duplicate_memories": {"count": 0, "groups": [], "skipped": True},
                "fts_indexes": {"existing": [], "missing": list(FTS_INDEXES), "skipped": True},
                "vacuum_analyze": {"tables": ["memories", "memory_chunks"], "pending": True},
            }

        findings: dict[str, Any] = {"postgres": {"ok": True}}
        try:
            with (
                self.backend._pool.connection() as connection,  # noqa: SLF001 - maintenance helper.
                connection.cursor(row_factory=dict_row) as cursor,
            ):
                cursor.execute(
                    """
                    SELECT c.id, c.memory_id, c.chunk_index
                    FROM ilma.memory_chunks c
                    LEFT JOIN ilma.memories m ON m.id = c.memory_id
                    WHERE m.id IS NULL
                    ORDER BY c.id
                    LIMIT 100
                    """
                )
                orphan_rows = [dict(row) for row in cursor.fetchall()]
                cursor.execute(
                    """
                    SELECT count(*) AS count
                    FROM ilma.memory_chunks c
                    LEFT JOIN ilma.memories m ON m.id = c.memory_id
                    WHERE m.id IS NULL
                    """
                )
                orphan_count = int(cursor.fetchone()["count"])
                findings["orphaned_chunks"] = {"count": orphan_count, "rows": orphan_rows}

                cursor.execute(
                    """
                    SELECT md5(content) AS content_hash,
                           count(*) AS count,
                           array_agg(id ORDER BY id) AS memory_ids
                    FROM ilma.memories
                    WHERE deleted_at IS NULL
                    GROUP BY md5(content)
                    HAVING count(*) > 1
                    ORDER BY count(*) DESC, min(id)
                    LIMIT 100
                    """
                )
                duplicate_groups = [dict(row) for row in cursor.fetchall()]
                findings["duplicate_memories"] = {
                    "count": len(duplicate_groups),
                    "groups": duplicate_groups,
                }

                cursor.execute(
                    """
                    SELECT c.relname
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'ilma' AND c.relname = ANY(%s)
                    """,
                    (list(FTS_INDEXES),),
                )
                existing = sorted(str(row["relname"]) for row in cursor.fetchall())
                findings["fts_indexes"] = {
                    "existing": existing,
                    "missing": [name for name in FTS_INDEXES if name not in existing],
                }
                findings["vacuum_analyze"] = {
                    "tables": ["memories", "memory_chunks"],
                    "pending": True,
                }
        except Exception as exc:
            findings["postgres"] = {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        return findings

    def _repair_apply(self, findings: Mapping[str, Any]) -> dict[str, Any]:
        if not self._has_pg_pool():
            return {"skipped": "backend does not expose a Postgres pool"}

        actions: dict[str, Any] = {
            "deleted_orphaned_chunks": 0,
            "reindexed": [],
            "vacuum_analyze": [],
        }
        with self.backend._pool.connection() as connection:  # noqa: SLF001 - maintenance helper.
            orphaned = findings.get("orphaned_chunks", {})
            if isinstance(orphaned, Mapping) and int(orphaned.get("count", 0) or 0) > 0:
                row = connection.execute(
                    """
                    DELETE FROM ilma.memory_chunks c
                    WHERE NOT EXISTS (SELECT 1 FROM ilma.memories m WHERE m.id = c.memory_id)
                    RETURNING c.id
                    """
                ).fetchall()
                actions["deleted_orphaned_chunks"] = len(row)

            fts = findings.get("fts_indexes", {})
            existing_indexes = fts.get("existing", []) if isinstance(fts, Mapping) else []
            for index_name in existing_indexes:
                if index_name not in FTS_INDEXES:
                    continue
                connection.execute(f"REINDEX INDEX ilma.{index_name}")
                actions["reindexed"].append(index_name)

        with self.backend._pool.connection() as connection:  # noqa: SLF001 - maintenance helper.
            previous_autocommit = connection.autocommit
            connection.autocommit = True
            try:
                for table_name in ("memories", "memory_chunks"):
                    connection.execute(f"VACUUM (ANALYZE) ilma.{table_name}")
                    actions["vacuum_analyze"].append(table_name)
            finally:
                connection.autocommit = previous_autocommit
        return actions

    def _doctor_backend(self) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], self.backend.health())
        except Exception as exc:
            return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}

    def _doctor_pgvector(self) -> dict[str, Any]:
        if self._has_pg_pool():
            try:
                with self.backend._pool.connection() as connection:  # noqa: SLF001 - health helper.
                    row = connection.execute(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    ).fetchone()
                return {"ok": bool(row and row[0]), "installed": bool(row and row[0])}
            except Exception as exc:
                return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
        backend = self._doctor_backend()
        installed = bool(backend.get("pgvector", False))
        return {"ok": installed, "installed": installed, "source": "backend.health"}

    def _doctor_pool(self) -> dict[str, Any]:
        if not self._has_pg_pool():
            return {"ok": True, "configured": False, "skipped": "backend has no Postgres pool"}
        try:
            pool = self.backend._pool  # noqa: SLF001 - health helper.
            with pool.connection() as connection:
                row = connection.execute("SELECT 1").fetchone()
            stats_fn = getattr(pool, "get_stats", None)
            stats = stats_fn() if callable(stats_fn) else {}
            return {"ok": row is not None and row[0] == 1, "configured": True, "stats": stats}
        except Exception as exc:
            return {
                "ok": False,
                "configured": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    def _doctor_embedding_dimensions(self) -> dict[str, Any]:
        dims: dict[str, int] = {}
        for surface, repo in {"memory": self.memory, "wiki": self.wiki}.items():
            dim = getattr(repo, "default_dim", None)
            if dim is not None:
                dims[surface] = int(dim)
        unsupported = {surface: dim for surface, dim in dims.items() if dim not in SUPPORTED_DIMS}
        columns: dict[str, bool] = {}
        if self._has_pg_pool() and dims:
            try:
                expected_columns = [f"vector_{dim}" for dim in sorted(set(dims.values()))]
                with (
                    self.backend._pool.connection() as connection,  # noqa: SLF001 - health helper.
                    connection.cursor(row_factory=dict_row) as cursor,
                ):
                    cursor.execute(
                        """
                        SELECT table_name, column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'ilma'
                          AND table_name = ANY(%s)
                          AND column_name = ANY(%s)
                        """,
                        (
                            ["memories", "memory_chunks", "wiki_docs", "wiki_chunks"],
                            expected_columns,
                        ),
                    )
                    present = {(row["table_name"], row["column_name"]) for row in cursor.fetchall()}
                for table_name in ("memories", "memory_chunks"):
                    if "memory" in dims:
                        columns[f"{table_name}.vector_{dims['memory']}"] = (
                            table_name,
                            f"vector_{dims['memory']}",
                        ) in present
                for table_name in ("wiki_chunks",):
                    if "wiki" in dims:
                        columns[f"{table_name}.vector_{dims['wiki']}"] = (
                            table_name,
                            f"vector_{dims['wiki']}",
                        ) in present
            except Exception as exc:
                return {
                    "ok": False,
                    "dimensions": dims,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
        ok = not unsupported and all(columns.values()) if columns else not unsupported
        return {
            "ok": ok,
            "dimensions": dims,
            "supported_dimensions": list(SUPPORTED_DIMS),
            "unsupported": unsupported,
            "columns": columns,
        }

    def _doctor_surfaces(self) -> dict[str, Any]:
        if not self._has_pg_pool():
            checks: dict[str, Any] = {}
            for name, repo in self._surface_repos().items():
                try:
                    initializer = getattr(repo, "initialize_schema", None)
                    if callable(initializer):
                        initializer()
                    checks[name] = {"ok": True, "checked": "initializer"}
                except Exception as exc:
                    checks[name] = {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            return checks

        try:
            table_names = sorted({table for tables in SURFACE_TABLES.values() for table in tables})
            with (
                self.backend._pool.connection() as connection,  # noqa: SLF001 - health helper.
                connection.cursor(row_factory=dict_row) as cursor,
            ):
                cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'ilma' AND table_name = ANY(%s)
                    """,
                    (table_names,),
                )
                present = {str(row["table_name"]) for row in cursor.fetchall()}
            return {
                surface: {
                    "ok": all(table in present for table in tables),
                    "tables": {table: table in present for table in tables},
                }
                for surface, tables in SURFACE_TABLES.items()
            }
        except Exception as exc:
            return {
                surface: {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                for surface in SURFACE_TABLES
            }

    def _doctor_audit_log(self) -> dict[str, Any]:
        if self.audit.__class__.__name__ == "InMemoryAuditLogger":
            return {"ok": True, "type": "in_memory"}
        if not self._has_pg_pool():
            return {"ok": True, "skipped": "backend has no Postgres pool"}
        try:
            with self.backend._pool.connection() as connection:  # noqa: SLF001 - health helper.
                row = connection.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'ilma' AND table_name = 'audit_log'
                    )
                    """
                ).fetchone()
            return {"ok": bool(row and row[0]), "table_exists": bool(row and row[0])}
        except Exception as exc:
            return {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}

    def _surface_repos(self) -> Mapping[str, Any]:
        return {
            "memory": self.memory,
            "wiki": self.wiki,
            "journal": self.journal,
            "skills": self.skills,
            "kanban": self.kanban,
            "metrics": self.metrics,
            "observability": self.observability,
            "sessions": self.sessions,
        }

    def _audit_rows(
        self,
        *,
        tool: str | None,
        status: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        offset_value: int,
    ) -> list[dict[str, Any]]:
        records = getattr(self.audit, "records", None)
        if isinstance(records, list):
            rows = [dict(record) for record in records]
            if tool:
                rows = [row for row in rows if row.get("tool_name") == tool]
            if status:
                rows = [row for row in rows if row.get("status") == status]
            # In-memory audit records in tests do not have timestamps; date filters
            # only apply to rows that carry created_at.
            if start:
                rows = [
                    row
                    for row in rows
                    if not isinstance(row.get("created_at"), datetime) or row["created_at"] >= start
                ]
            if end:
                rows = [
                    row
                    for row in rows
                    if not isinstance(row.get("created_at"), datetime) or row["created_at"] <= end
                ]
            return rows[offset_value : offset_value + limit]

        if not self._has_pg_pool():
            return []
        self.audit.initialize_schema()
        sql = [
            "SELECT id, operation_id, tool_name, surface, action, payload, status, ",
            "error_type, error_message, result, created_at, completed_at ",
            "FROM ilma.audit_log WHERE true ",
        ]
        params: list[Any] = []
        if tool:
            sql.append("AND tool_name = %s ")
            params.append(tool)
        if status:
            sql.append("AND status = %s ")
            params.append(status)
        if start:
            sql.append("AND created_at >= %s ")
            params.append(start)
        if end:
            sql.append("AND created_at <= %s ")
            params.append(end)
        sql.append("ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s")
        params.extend([limit, offset_value])
        with (
            self.backend._pool.connection() as connection,  # noqa: SLF001 - audit query helper.
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [dict(row) for row in cursor.fetchall()]

    def _initialize_all(self) -> None:
        backend_init = getattr(self.backend, "initialize_schema", None)
        if callable(backend_init):
            backend_init()
        for repo in (
            self.memory,
            self.wiki,
            self.journal,
            self.skills,
            self.metrics,
            self.kanban,
            self.observability,
            self.sessions,
        ):
            initializer = getattr(repo, "initialize_schema", None)
            if callable(initializer):
                initializer()
        self.audit.initialize_schema()

    # SQL fallback helpers for list/get tools not present in core Protocols.
    @staticmethod
    def _filter_by_threshold(results: list[Any], threshold: float) -> list[Any]:
        """Filter search results by minimum similarity score.

        The current ``ilma.memory.search`` does not return per-row similarity
        scores — the hybrid search sorts results by combined score internally
        and discards the score before returning. This helper therefore acts
        as a no-op pass-through today, returning the input list unchanged
        when ``threshold > 0``.

        Once ``Memory`` rows carry a ``score`` attribute (planned), this
        filter will start enforcing ``score >= threshold`` without callers
        needing to change. The threshold parameter is kept in the public
        API so existing callers don't need to migrate later.
        """
        if threshold <= 0:
            return results
        # Look for a numeric `score` attribute on each row. If present, filter;
        # otherwise preserve all results (backward-compatible with rows that
        # don't expose a score).
        filtered: list[Any] = []
        for row in results:
            score = getattr(row, "score", None)
            if score is None and isinstance(row, dict):
                score = row.get("score")
            if score is None or score >= threshold:
                filtered.append(row)
        return filtered

    def _memory_rows(
        self,
        *,
        limit: int = 50,
        offset_value: int = 0,
        memory_id: int | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        sql = [
            "SELECT id, content, tags, category, source, metadata, ",
            "deleted_at IS NOT NULL AS deleted, created_at ",
            "FROM ilma.memories WHERE true ",
        ]
        params: list[Any] = []
        if memory_id is not None:
            sql.append("AND id = %s ")
            params.append(memory_id)
        if not include_deleted:
            sql.append("AND deleted_at IS NULL ")
        sql.append("ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s")
        params.extend([limit, offset_value])
        with (
            self.backend._pool.connection() as connection,  # noqa: SLF001 - PG fallback helper.
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute("".join(sql), params)
            return [dict(row) for row in cursor.fetchall()]

    def _wiki_rows(self, *, limit: int = 50, offset_value: int = 0) -> list[dict[str, Any]]:
        with (
            self.backend._pool.connection() as connection,  # noqa: SLF001 - PG fallback helper.
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """
                SELECT id, slug, title, category, tags, source_uri, version,
                       created_at, updated_at, metadata
                FROM ilma.wiki_docs
                ORDER BY updated_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset_value),
            )
            return [dict(row) for row in cursor.fetchall()]


_READ_TOOL_ACTIONS = frozenset(
    {
        "audit",
        "doctor",
        "get",
        "health",
        "list",
        "query",
        "recall",
        "recent",
        "status",
    }
)
_READ_TOOL_SUFFIXES = frozenset({"search"})
_MAINTENANCE_ACTIONS = frozenset({"migrate", "repair"})


def _derive_write_tools(service_cls: type[Any] = IlmaService) -> dict[str, tuple[str, str]]:
    """Derive audit write-tool metadata by inspecting IlmaService methods.

    A write tool is a public ``ilma_*`` method that dispatches through
    ``self.call(...)`` and whose tool-name action is not read-only. The
    surface/action pair is inferred from the tool name and, when available,
    from the repository surface used in the method body.
    """

    discovered: list[tuple[int, str, tuple[str, str]]] = []
    for index, (name, member) in enumerate(vars(service_cls).items()):
        if name in EXCLUDED_TOOL_NAMES or not name.startswith(TOOLS_ATTR_PREFIX):
            continue
        if not inspect.isfunction(member) or not _method_calls_self_call(member):
            continue
        surface_action = _infer_write_surface_action(name, member)
        if surface_action is None:
            continue
        discovered.append((index, name, surface_action))
    discovered.sort(key=_write_tool_order_key)
    return {name: surface_action for _, name, surface_action in discovered}


def _write_tool_order_key(item: tuple[int, str, tuple[str, str]]) -> tuple[int, str]:
    index, _, (surface, action) = item
    surface_order = {surface_name: order for order, surface_name in enumerate(SURFACE_TABLES)}
    surface_order["maintenance"] = len(surface_order)
    order = surface_order.get(surface, len(surface_order) + 1)
    secondary = action if surface == "maintenance" else f"{index:06d}"
    return order, secondary


def _method_calls_self_call(method: Callable[..., Any]) -> bool:
    tree = _method_ast(method)
    if tree is None:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        for node in ast.walk(tree)
    )


def _infer_write_surface_action(
    tool_name: str, method: Callable[..., Any]
) -> tuple[str, str] | None:
    suffix = tool_name.removeprefix(TOOLS_ATTR_PREFIX)
    parts = suffix.split("_")
    if not suffix or _is_read_tool_action(parts[0]):
        return None

    if len(parts) == 1:
        action = parts[0]
        if _is_read_tool_action(action):
            return None
        surface = _single_surface_from_method(method) or _surface_for_single_action(action)
        if surface is None:
            return None
        return surface, action

    action = "_".join(parts[1:])
    if _is_read_tool_action(action):
        return None
    surface = _single_surface_from_method(method) or parts[0]
    return surface, action


def _is_read_tool_action(action: str) -> bool:
    return action in _READ_TOOL_ACTIONS or action in _READ_TOOL_SUFFIXES


def _single_surface_from_method(method: Callable[..., Any]) -> str | None:
    tree = _method_ast(method)
    if tree is None:
        return None
    surfaces = {
        node.func.value.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr in SURFACE_TABLES
    }
    if len(surfaces) == 1:
        return next(iter(surfaces))
    return None


def _surface_for_single_action(action: str) -> str | None:
    if action in _MAINTENANCE_ACTIONS:
        return "maintenance"
    return None


def _method_ast(method: Callable[..., Any]) -> ast.Module | None:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return None
    try:
        return ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None


WRITE_TOOLS: Mapping[str, tuple[str, str]] = _derive_write_tools()


def _count_tools(service_cls: type[Any] = IlmaService) -> int:
    """Return the count of public ``ilma_*`` methods on the service.

    This is the runtime source of truth for the tool surface size and is
    consumed by tests, the CLI banner, and the MCP server's registration
    assertion. Keeping it derived (instead of a hand-maintained constant)
    ensures it stays in sync as new tools are added.
    """

    return sum(
        1
        for name, member in vars(service_cls).items()
        if name.startswith(TOOLS_ATTR_PREFIX)
        and name not in EXCLUDED_TOOL_NAMES
        and inspect.isfunction(member)
    )


TOOL_COUNT: int = _count_tools()
