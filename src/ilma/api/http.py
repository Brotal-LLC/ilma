"""FastAPI HTTP API for ilma.

The HTTP layer intentionally reuses :class:`ilma.api.mcp.IlmaMcpService` so REST
and MCP expose the same behavior while keeping framework concerns isolated to
this module.  No Hermes-specific imports belong here.
"""

from __future__ import annotations

import os
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ilma.api.hardening import (
    METRICS,
    SlidingWindowRateLimiter,
    log_observation,
    pool_size_from_backend,
)
from ilma.api.mcp import IlmaMcpService, get_service, set_service

try:  # pragma: no cover - exercised only when optional HTTP dependencies are absent.
    from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required for the HTTP API. Install ilma with the 'http' extra "
        "or install 'fastapi' and 'uvicorn'."
    ) from exc


SURFACES = [
    "memory",
    "wiki",
    "journal",
    "skills",
    "kanban",
    "metrics",
    "observability",
    "sessions",
]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=500)
    hybrid_text_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class RememberRequest(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    source: str | None = "http"


class ForgetRequest(BaseModel):
    memory_id: int = Field(ge=1)


class WikiSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=500)


class WikiWriteRequest(BaseModel):
    slug: str
    title: str
    body_md: str
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_uri: str | None = None


class JournalSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=500)


class SkillsSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=500)


class KanbanCreateRequest(BaseModel):
    title: str
    description: str = ""
    status: str = "todo"
    priority: int = 0
    tags: list[str] = Field(default_factory=list)
    parent_id: int | None = None


class KanbanUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: int | None = None
    tags: list[str] | None = None
    parent_id: int | None = None
    metadata: dict[str, Any] | None = None


class MetricRecordRequest(BaseModel):
    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


class MetricQueryRequest(BaseModel):
    name: str
    start: str | None = None
    end: str | None = None
    limit: int = Field(default=100, ge=1, le=500)
    aggregate_window: str | None = None


class ObservationRecordRequest(BaseModel):
    level: str
    message: str
    source: str | None = "http"
    context: dict[str, Any] = Field(default_factory=dict)


class ObservationQueryRequest(BaseModel):
    level: str | None = None
    source: str | None = None
    start: str | None = None
    end: str | None = None
    limit: int = Field(default=100, ge=1, le=500)


class SessionSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=500)


def service_dependency() -> IlmaMcpService:
    """FastAPI dependency that resolves the configured ilma service lazily."""

    return get_service()


SERVICE_DEPENDENCY = Depends(service_dependency)

AUTH_EXEMPT_PATHS = {"/health", "/openapi.json"}
RATE_LIMIT_EXEMPT_PATHS = {"/health"}


def _api_key_from_env() -> str | None:
    value = os.environ.get("ILMA_API_KEY", "").strip()
    return value or None


def _is_api_key_authorized(request: Request) -> bool:
    configured_api_key = _api_key_from_env()
    if not configured_api_key or request.url.path in AUTH_EXEMPT_PATHS:
        return True
    return request.headers.get("x-api-key", "") == configured_api_key


def api_key_dependency(request: Request) -> None:
    """FastAPI dependency enforcing ILMA_API_KEY via the X-API-Key header."""

    if not _is_api_key_authorized(request):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _rate_limit_from_env() -> float:
    value = os.environ.get("ILMA_RATE_LIMIT_RPS", "30").strip()
    try:
        parsed = float(value)
    except ValueError:
        return 30.0
    return parsed if parsed >= 0 else 30.0


def _cors_origins_from_env() -> list[str]:
    value = os.environ.get("ILMA_CORS_ORIGINS", "*").strip()
    if not value:
        return ["*"]
    origins = [origin.strip() for origin in value.split(",") if origin.strip()]
    return origins or ["*"]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)


def _log_http_access(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
    client_ip: str,
) -> None:
    try:
        service = get_service()
    except Exception:
        return
    observability = getattr(service, "observability", None)
    log_observation(
        observability,
        level="info" if status_code < 500 else "error",
        message="http request",
        source="http.access",
        context={
            "method": method,
            "path": path,
            "status": status_code,
            "duration_ms": round(duration_seconds * 1000, 3),
            "client_ip": client_ip,
        },
    )


def _refresh_runtime_gauges(service: IlmaMcpService) -> None:
    METRICS.set_gauge(
        "db_connection_pool_size", pool_size_from_backend(getattr(service, "backend", None))
    )


def _health_payload(service: IlmaMcpService) -> dict[str, Any]:
    health = getattr(service, "ilma_health", None)
    if callable(health):
        payload = health()
        if isinstance(payload, dict):
            return payload
        return {
            "ok": False,
            "error": {
                "type": "InvalidHealthPayload",
                "message": "health payload must be an object",
            },
        }
    status = service.ilma_status()
    backend = status.get("backend", {}) if status.get("ok") else {}
    return {
        "ok": bool(status.get("ok") and backend.get("ok", True)),
        "backend": backend,
    }


def create_app(service: IlmaMcpService | None = None) -> FastAPI:
    """Create the FastAPI app for the ilma REST API.

    Args:
        service: Optional pre-built service, primarily for embedding and tests. If
            omitted, the app resolves a process-wide service from environment on
            first request via :func:`ilma.api.mcp.get_service`.
    """

    if service is not None:
        set_service(service)

    app = FastAPI(
        title="ilma HTTP API",
        description="REST API exposing all persistent ilma surfaces.",
        version="0.1.0",
    )

    cors_origins = _cors_origins_from_env()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    rate_limiter = SlidingWindowRateLimiter(requests_per_second=_rate_limit_from_env())

    @app.middleware("http")
    async def hardening_middleware(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        client_ip = _client_ip(request)
        status_code = 500
        response: Response
        try:
            if request.url.path not in RATE_LIMIT_EXEMPT_PATHS and not rate_limiter.allow(
                client_ip
            ):
                response = JSONResponse(
                    status_code=429,
                    content={
                        "ok": False,
                        "error": {"type": "RateLimitExceeded", "message": "rate limit exceeded"},
                    },
                )
            else:
                if not _is_api_key_authorized(request):
                    response = JSONResponse(
                        status_code=401,
                        content={
                            "ok": False,
                            "error": {
                                "type": "Unauthorized",
                                "message": "invalid or missing API key",
                            },
                        },
                    )
                else:
                    response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_seconds = time.perf_counter() - started
            path = _route_path(request)
            labels = {"method": request.method, "path": path, "status": str(status_code)}
            METRICS.increment("request_count", labels)
            METRICS.observe("request_duration", duration_seconds, labels)
            _log_http_access(
                method=request.method,
                path=path,
                status_code=status_code,
                duration_seconds=duration_seconds,
                client_ip=client_ip,
            )

    @app.get("/health", tags=["system"])
    def health(service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        """Return structured health checks for Postgres, pgvector, and embedders."""

        return _health_payload(service)

    @app.get("/metrics", tags=["system"], response_class=PlainTextResponse)
    def scrape_metrics(service: IlmaMcpService = SERVICE_DEPENDENCY) -> str:
        """Return Prometheus-style in-memory runtime metrics."""

        _refresh_runtime_gauges(service)
        return METRICS.render_prometheus()

    @app.get("/status", tags=["system"])
    def status(service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        """Return backend and surface status."""

        return service.ilma_status()

    # Memory surface -----------------------------------------------------
    @app.post("/search", tags=["memory"])
    def search_memories(
        request: SearchRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_search(request.query, request.top_k, request.hybrid_text_weight)

    @app.post("/remember", tags=["memory"])
    def remember(
        request: RememberRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_remember(
            request.content, request.tags, request.category, request.source
        )

    @app.post("/forget", tags=["memory"])
    def forget(
        request: ForgetRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_forget(request.memory_id)

    @app.get("/memories/{memory_id}", tags=["memory"])
    def get_memory(memory_id: int, service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_get_memory(memory_id)

    @app.get("/memories", tags=["memory"])
    def list_memories(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        include_deleted: bool = False,
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_list_memories(limit, offset, include_deleted)

    # Wiki surface -------------------------------------------------------
    @app.post("/wiki/search", tags=["wiki"])
    def search_wiki(
        request: WikiSearchRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_search_wiki(request.query, request.top_k)

    @app.get("/wiki", tags=["wiki"])
    def list_wiki(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_list_wiki(limit, offset)

    @app.get("/wiki/{slug}", tags=["wiki"])
    def get_wiki(slug: str, service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_get_wiki(slug)

    @app.post("/wiki", tags=["wiki"])
    def create_wiki(
        request: WikiWriteRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_wiki_create(
            request.slug,
            request.title,
            request.body_md,
            request.category,
            request.tags,
            request.source_uri,
        )

    @app.patch("/wiki/{slug}", tags=["wiki"])
    def update_wiki(
        slug: str,
        request: WikiWriteRequest,
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_wiki_update(
            slug,
            request.title,
            request.body_md,
            request.category,
            request.tags,
            request.source_uri,
        )

    # Journal surface ----------------------------------------------------
    @app.post("/journal/search", tags=["journal"])
    def search_journal(
        request: JournalSearchRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_journal_search(request.query, request.top_k)

    @app.get("/journal/recent", tags=["journal"])
    def recent_journal(
        limit: int = Query(default=10, ge=1, le=500),
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_journal_recent(limit)

    # Skills surface -----------------------------------------------------
    @app.post("/skills/search", tags=["skills"])
    def search_skills(
        request: SkillsSearchRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_skills_search(request.query, request.top_k)

    @app.get("/skills/{name}", tags=["skills"])
    def get_skill(name: str, service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_skills_get(name)

    # Kanban surface -----------------------------------------------------
    @app.get("/kanban", tags=["kanban"])
    def list_kanban(
        status: str = "todo",
        limit: int = Query(default=50, ge=1, le=500),
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_kanban_list(status, limit)

    @app.get("/kanban/{task_id}", tags=["kanban"])
    def get_kanban(task_id: int, service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_kanban_get(task_id)

    @app.post("/kanban", tags=["kanban"])
    def create_kanban(
        request: KanbanCreateRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_kanban_create(
            request.title,
            request.description,
            request.status,
            request.priority,
            request.tags,
            request.parent_id,
        )

    @app.patch("/kanban/{task_id}", tags=["kanban"])
    def update_kanban(
        task_id: int,
        request: KanbanUpdateRequest,
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_kanban_update(
            task_id,
            request.title,
            request.description,
            request.status,
            request.priority,
            request.tags,
            request.parent_id,
            request.metadata,
        )

    @app.post("/kanban/{task_id}/complete", tags=["kanban"])
    def complete_kanban(
        task_id: int, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_kanban_complete(task_id)

    # Metrics surface ----------------------------------------------------
    @app.post("/metrics", tags=["metrics"])
    def record_metric(
        request: MetricRecordRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_metrics_record(request.name, request.value, request.labels)

    @app.post("/metrics/query", tags=["metrics"])
    def query_metrics(
        request: MetricQueryRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_metrics_query(
            request.name,
            request.start,
            request.end,
            request.limit,
            request.aggregate_window,
        )

    # Observability surface ---------------------------------------------
    @app.post("/observations", tags=["observability"])
    def record_observation(
        request: ObservationRecordRequest,
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_obs_log(request.level, request.message, request.source, request.context)

    @app.post("/observations/query", tags=["observability"])
    def query_observations(
        request: ObservationQueryRequest,
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_obs_query(
            request.level, request.source, request.start, request.end, request.limit
        )

    # Sessions surface ---------------------------------------------------
    @app.post("/sessions/search", tags=["sessions"])
    def search_sessions(
        request: SessionSearchRequest, service: IlmaMcpService = SERVICE_DEPENDENCY
    ) -> dict[str, Any]:
        return service.ilma_session_search(request.query, request.top_k)

    @app.get("/sessions/{session_id}", tags=["sessions"])
    def get_session(
        session_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        service: IlmaMcpService = SERVICE_DEPENDENCY,
    ) -> dict[str, Any]:
        return service.ilma_session_get(session_id, limit)

    # Maintenance --------------------------------------------------------
    @app.post("/repair", tags=["maintenance"])
    def repair(service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_repair()

    @app.post("/doctor", tags=["maintenance"])
    def doctor(service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_doctor()

    @app.post("/migrate", tags=["maintenance"])
    def migrate(service: IlmaMcpService = SERVICE_DEPENDENCY) -> dict[str, Any]:
        return service.ilma_migrate()

    return app


def app_factory() -> FastAPI:
    """Uvicorn-friendly zero-argument app factory."""

    return create_app()


app = create_app()


__all__ = [
    "SURFACES",
    "api_key_dependency",
    "app",
    "app_factory",
    "create_app",
    "service_dependency",
]
