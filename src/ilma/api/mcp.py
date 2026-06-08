"""MCP server for ilma.

This module is intentionally framework-agnostic: it imports only ilma core/storage
code and the official Python MCP SDK.  It exposes all persistent ilma surfaces as
MCP tools and returns structured success/error dictionaries so client tool calls do
not crash the server on expected repository errors.
"""

from __future__ import annotations

import os
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ilma.storage.postgres import PgBackend

try:  # Imported at module load so the entry point fails clearly if MCP is absent.
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised by packaging/import environments.
    raise RuntimeError(
        "The official Python MCP SDK is required. Install ilma with the 'mcp' extra "
        "or install the 'mcp' package."
    ) from exc

TOOL_COUNT = 30
WRITE_TOOLS: Mapping[str, tuple[str, str]] = {
    "ilma_remember": ("memory", "remember"),
    "ilma_forget": ("memory", "forget"),
    "ilma_wiki_create": ("wiki", "create"),
    "ilma_wiki_update": ("wiki", "update"),
    "ilma_kanban_create": ("kanban", "create"),
    "ilma_kanban_update": ("kanban", "update"),
    "ilma_kanban_complete": ("kanban", "complete"),
    "ilma_metrics_record": ("metrics", "record"),
    "ilma_obs_log": ("observability", "log"),
    "ilma_migrate": ("maintenance", "migrate"),
    "ilma_repair": ("maintenance", "repair"),
}


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


class IlmaMcpService:
    """Tool implementation layer used by the MCP SDK registration and tests."""

    def __init__(self, backend: Any, audit: Any | None = None) -> None:
        self.backend = backend
        self.audit = audit or PgAuditLogger(backend)
        self.memory = backend.memory_repo()
        self.wiki = backend.wiki_repo()
        self.journal = backend.journal_repo()
        self.skills = backend.skills_repo()
        self.kanban = backend.kanban_repo()
        self.metrics = backend.metrics_repo()
        self.observability = backend.observability_repo()
        self.sessions = backend.sessions_repo()

    @classmethod
    def from_env(cls) -> IlmaMcpService:
        dsn = _dsn_from_env()
        min_pool_size = int(os.environ.get("ILMA_PG_POOL_MIN", "1"))
        max_pool_size = int(os.environ.get("ILMA_PG_POOL_MAX", "8"))
        return cls(PgBackend(dsn, min_pool_size=min_pool_size, max_pool_size=max_pool_size))

    def call(
        self, tool_name: str, fn: Callable[[], dict[str, Any]], payload: dict[str, Any]
    ) -> dict[str, Any]:
        operation_id: str | None = None
        if tool_name in WRITE_TOOLS:
            surface, action = WRITE_TOOLS[tool_name]
            try:
                operation_id = self.audit.begin(tool_name, surface, action, payload)
            except Exception as exc:  # audit is write-ahead: do not execute write if it fails.
                return _err(exc)
        try:
            result = fn()
        except Exception as exc:  # Never crash MCP server for repository/tool errors.
            if operation_id is not None:
                self._safe_audit_finish(operation_id, status="failed", error=exc)
            return _err(exc)
        if operation_id is not None:
            self._safe_audit_finish(operation_id, status="succeeded", result=result)
        return _json_safe(result)

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

    def ilma_search(
        self, query: str, top_k: int = 10, hybrid_text_weight: float = 0.5
    ) -> dict[str, Any]:
        return self.call(
            "ilma_search",
            lambda: _ok(
                results=self.memory.search(
                    query, top_k=_limit(top_k, default=10), hybrid_text_weight=hybrid_text_weight
                )
            ),
            {"query": query, "top_k": top_k, "hybrid_text_weight": hybrid_text_weight},
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

    def ilma_search_wiki(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self.call(
            "ilma_search_wiki",
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
    def ilma_repair(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._initialize_all()
            return _ok(repaired=True, message="schemas and audit log verified")

        return self.call("ilma_repair", run, {})

    def ilma_doctor(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            checks: dict[str, Any] = {"dsn_configured": True, "surfaces": {}}
            try:
                checks["backend"] = self.backend.health()
            except Exception as exc:
                checks["backend"] = {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            for name, repo in {
                "memory": self.memory,
                "wiki": self.wiki,
                "journal": self.journal,
                "skills": self.skills,
                "metrics": self.metrics,
                "kanban": self.kanban,
                "observability": self.observability,
                "sessions": self.sessions,
            }.items():
                try:
                    initializer = getattr(repo, "initialize_schema", None)
                    if callable(initializer):
                        initializer()
                    checks["surfaces"][name] = {"ok": True}
                except Exception as exc:
                    checks["surfaces"][name] = {
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
            try:
                self.audit.initialize_schema()
                checks["audit_log"] = {"ok": True}
            except Exception as exc:
                checks["audit_log"] = {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            overall = all(
                check.get("ok", False)
                for check in [checks.get("backend", {})]
                + list(checks.get("surfaces", {}).values())
                + [checks.get("audit_log", {})]
            )
            return _ok(healthy=overall, checks=checks)

        return self.call("ilma_doctor", run, {})

    def ilma_migrate(self) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            self._initialize_all()
            return _ok(migrated=True, surfaces=8, audit_log=True)

        return self.call("ilma_migrate", run, {})

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
    """Create and register the official MCP SDK server."""

    if service is not None:
        set_service(service)
    server = FastMCP("ilma")

    @server.tool()
    def ilma_status() -> dict[str, Any]:
        """Return backend and repository health/status for all ilma surfaces."""
        return get_service().ilma_status()

    @server.tool()
    def ilma_search(query: str, top_k: int = 10, hybrid_text_weight: float = 0.5) -> dict[str, Any]:
        """Search memories with hybrid full-text/vector retrieval."""
        return get_service().ilma_search(query, top_k, hybrid_text_weight)

    @server.tool()
    def ilma_recent(limit: int = 10) -> dict[str, Any]:
        """Return recent memories."""
        return get_service().ilma_recent(limit)

    @server.tool()
    def ilma_get_memory(memory_id: int) -> dict[str, Any]:
        """Fetch a memory by id."""
        return get_service().ilma_get_memory(memory_id)

    @server.tool()
    def ilma_list_memories(
        limit: int = 50, offset: int = 0, include_deleted: bool = False
    ) -> dict[str, Any]:
        """List memories with pagination."""
        return get_service().ilma_list_memories(limit, offset, include_deleted)

    @server.tool()
    def ilma_remember(
        content: str,
        tags: list[str] | None = None,
        category: str | None = None,
        source: str | None = "mcp",
    ) -> dict[str, Any]:
        """Store a short durable memory."""
        return get_service().ilma_remember(content, tags, category, source)

    @server.tool()
    def ilma_forget(memory_id: int) -> dict[str, Any]:
        """Soft-delete a memory by id."""
        return get_service().ilma_forget(memory_id)

    @server.tool()
    def ilma_get_wiki(slug: str) -> dict[str, Any]:
        """Fetch a wiki document by slug."""
        return get_service().ilma_get_wiki(slug)

    @server.tool()
    def ilma_search_wiki(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search wiki documents."""
        return get_service().ilma_search_wiki(query, top_k)

    @server.tool()
    def ilma_list_wiki(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List wiki documents."""
        return get_service().ilma_list_wiki(limit, offset)

    @server.tool()
    def ilma_wiki_create(
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        """Create or ingest a wiki document."""
        return get_service().ilma_wiki_create(slug, title, body_md, category, tags, source_uri)

    @server.tool()
    def ilma_wiki_update(
        slug: str,
        title: str,
        body_md: str,
        category: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        """Update a wiki document, creating a new version."""
        return get_service().ilma_wiki_update(slug, title, body_md, category, tags, source_uri)

    @server.tool()
    def ilma_journal_search(query: str, top_k: int = 10) -> dict[str, Any]:
        """Search journal entries."""
        return get_service().ilma_journal_search(query, top_k)

    @server.tool()
    def ilma_journal_recent(limit: int = 10) -> dict[str, Any]:
        """Return recent journal entries."""
        return get_service().ilma_journal_recent(limit)

    @server.tool()
    def ilma_skills_search(query: str, top_k: int = 5) -> dict[str, Any]:
        """Search stored skills."""
        return get_service().ilma_skills_search(query, top_k)

    @server.tool()
    def ilma_skills_get(name: str) -> dict[str, Any]:
        """Fetch a stored skill by name."""
        return get_service().ilma_skills_get(name)

    @server.tool()
    def ilma_kanban_list(status: str = "todo", limit: int = 50) -> dict[str, Any]:
        """List kanban tasks by status."""
        return get_service().ilma_kanban_list(status, limit)

    @server.tool()
    def ilma_kanban_get(task_id: int) -> dict[str, Any]:
        """Fetch a kanban task by id."""
        return get_service().ilma_kanban_get(task_id)

    @server.tool()
    def ilma_kanban_create(
        title: str,
        description: str = "",
        status: str = "todo",
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        """Create a kanban task."""
        return get_service().ilma_kanban_create(
            title, description, status, priority, tags, parent_id
        )

    @server.tool()
    def ilma_kanban_update(
        task_id: int,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        tags: list[str] | None = None,
        parent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update kanban task fields."""
        return get_service().ilma_kanban_update(
            task_id, title, description, status, priority, tags, parent_id, metadata
        )

    @server.tool()
    def ilma_kanban_complete(task_id: int) -> dict[str, Any]:
        """Mark a kanban task complete."""
        return get_service().ilma_kanban_complete(task_id)

    @server.tool()
    def ilma_metrics_record(
        name: str, value: float, labels: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Record a metric point."""
        return get_service().ilma_metrics_record(name, value, labels)

    @server.tool()
    def ilma_metrics_query(
        name: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        aggregate_window: str | None = None,
    ) -> dict[str, Any]:
        """Query metric points or aggregates."""
        return get_service().ilma_metrics_query(name, start, end, limit, aggregate_window)

    @server.tool()
    def ilma_obs_log(
        level: str,
        message: str,
        source: str | None = "mcp",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an observability event."""
        return get_service().ilma_obs_log(level, message, source, context)

    @server.tool()
    def ilma_obs_query(
        level: str | None = None,
        source: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query observability events."""
        return get_service().ilma_obs_query(level, source, start, end, limit)

    @server.tool()
    def ilma_session_search(query: str, top_k: int = 10) -> dict[str, Any]:
        """Search session messages."""
        return get_service().ilma_session_search(query, top_k)

    @server.tool()
    def ilma_session_get(session_id: str, limit: int = 100) -> dict[str, Any]:
        """Fetch session messages by session id."""
        return get_service().ilma_session_get(session_id, limit)

    @server.tool()
    def ilma_repair() -> dict[str, Any]:
        """Repair/verify schemas for all surfaces and audit logging."""
        return get_service().ilma_repair()

    @server.tool()
    def ilma_doctor() -> dict[str, Any]:
        """Run health checks for backend, surfaces, and audit log."""
        return get_service().ilma_doctor()

    @server.tool()
    def ilma_migrate() -> dict[str, Any]:
        """Run idempotent schema migrations for all surfaces."""
        return get_service().ilma_migrate()

    return server


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
    "create_mcp_server",
    "get_service",
    "main",
    "set_service",
]
