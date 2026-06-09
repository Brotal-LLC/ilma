"""FastAPI HTTP API for ilma.

The HTTP layer intentionally reuses :class:`ilma.api.mcp.IlmaMcpService` so REST
and MCP expose the same behavior while keeping framework concerns isolated to
this module.  No Hermes-specific imports belong here.
"""

from __future__ import annotations

import inspect
import re
import time
from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model

from ilma.api.hardening import (
    METRICS,
    SlidingWindowRateLimiter,
    log_observation,
    pool_size_from_backend,
)
from ilma.api.mcp import IlmaMcpService, get_service, set_service
from ilma.config import IlmaConfig
from ilma.service import tools_dict

try:  # pragma: no cover - exercised only when optional HTTP dependencies are absent.
    from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query, Request, Response
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


_TOOL_TO_ROUTE: dict[str, tuple[str, str]] = {
    "ilma_status": ("/status", "GET"),
    "ilma_recall": ("/recall", "POST"),
    "ilma_remember": ("/remember", "POST"),
    "ilma_forget": ("/forget", "POST"),
    "ilma_get_memory": ("/memories/{memory_id}", "GET"),
    "ilma_list_memories": ("/memories", "GET"),
    "ilma_get_wiki": ("/wiki/{slug}", "GET"),
    "ilma_wiki_search": ("/wiki/search", "POST"),
    "ilma_list_wiki": ("/wiki", "GET"),
    "ilma_wiki_create": ("/wiki", "POST"),
    "ilma_wiki_update": ("/wiki/{slug}", "PATCH"),
    "ilma_journal_search": ("/journal/search", "POST"),
    "ilma_journal_recent": ("/journal/recent", "GET"),
    "ilma_skills_search": ("/skills/search", "POST"),
    "ilma_skills_get": ("/skills/{name}", "GET"),
    "ilma_kanban_list": ("/kanban", "GET"),
    "ilma_kanban_get": ("/kanban/{task_id}", "GET"),
    "ilma_kanban_create": ("/kanban", "POST"),
    "ilma_kanban_update": ("/kanban/{task_id}", "PATCH"),
    "ilma_kanban_complete": ("/kanban/{task_id}/complete", "POST"),
    "ilma_metrics_record": ("/metrics", "POST"),
    "ilma_metrics_query": ("/metrics/query", "POST"),
    "ilma_obs_log": ("/observations", "POST"),
    "ilma_obs_query": ("/observations/query", "POST"),
    "ilma_session_search": ("/sessions/search", "POST"),
    "ilma_session_get": ("/sessions/{session_id}", "GET"),
    "ilma_repair": ("/repair", "POST"),
    "ilma_doctor": ("/doctor", "POST"),
    # Existing HTTP API route. Not in the original R-008 table, but preserved
    # here to keep the public HTTP surface and existing tests unchanged.
    "ilma_migrate": ("/migrate", "POST"),
}

_HTTP_EXCLUDED = frozenset(
    {
        "ilma_status",  # /status is an infrastructure route with custom docs/tagging.
        "ilma_recent",  # MCP/CLI-only; no existing HTTP route.
        "ilma_audit",  # CLI-only.
    }
)

_PATH_PARAM_RE = re.compile(r"{(?P<name>[A-Za-z_]\w*)}")


def service_dependency() -> IlmaMcpService:
    """FastAPI dependency that resolves the configured ilma service lazily."""

    return get_service()


SERVICE_DEPENDENCY = Depends(service_dependency)

AUTH_EXEMPT_PATHS = {"/health", "/openapi.json"}
RATE_LIMIT_EXEMPT_PATHS = {"/health"}


def _api_key_from_env() -> str | None:
    value = (IlmaConfig.from_env().api.api_key or "").strip()
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
    parsed = IlmaConfig.from_env().api.rate_limit_rps
    return parsed if parsed >= 0 else 30.0


def _cors_origins_from_env() -> list[str]:
    origins = [
        origin.strip() for origin in IlmaConfig.from_env().api.cors_origins if origin.strip()
    ]
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


def _registration_service(service: IlmaMcpService | None) -> IlmaMcpService:
    """Return an object suitable for tools_dict() without opening a backend."""

    if service is not None:
        return service
    return IlmaMcpService.__new__(IlmaMcpService)


def _path_param_names(path: str) -> set[str]:
    return {match.group("name") for match in _PATH_PARAM_RE.finditer(path)}


def _route_tag(path: str) -> str:
    if path in {"/recall", "/remember", "/forget"} or path.startswith("/memories"):
        return "memory"
    if path.startswith("/observations"):
        return "observability"
    if path in {"/repair", "/doctor", "/migrate"}:
        return "maintenance"
    segment = path.strip("/").split("/", 1)[0]
    return segment or "system"


def _field_default(model: type[BaseModel], name: str) -> Any:
    field = model.model_fields[name]
    if field.is_required():
        return ...
    return field.default


def _field_annotation(model: type[BaseModel], name: str) -> Any:
    return model.model_fields[name].annotation or Any


def _body_model_for_route(
    tool_name: str, input_model: type[BaseModel], path_params: set[str]
) -> type[BaseModel] | None:
    body_fields = {
        name: (_field_annotation(input_model, name), _field_default(input_model, name))
        for name in input_model.model_fields
        if name not in path_params
    }
    if not body_fields:
        return None
    return create_model(f"{tool_name}_HttpBody", **body_fields)


def _route_signature(
    *, input_model: type[BaseModel], path: str, method: str, body_model: type[BaseModel] | None
) -> inspect.Signature:
    path_params = _path_param_names(path)
    parameters: list[inspect.Parameter] = [
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Request),
        inspect.Parameter(
            "service",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=SERVICE_DEPENDENCY,
            annotation=IlmaMcpService,
        ),
    ]
    for name in sorted(path_params):
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=Path(...),
                annotation=_field_annotation(input_model, name),
            )
        )
    if method == "GET":
        for name in input_model.model_fields:
            if name in path_params:
                continue
            default = _field_default(input_model, name)
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=Query(default),
                    annotation=_field_annotation(input_model, name),
                )
            )
    elif body_model is not None:
        required = any(field.is_required() for field in body_model.model_fields.values())
        parameters.append(
            inspect.Parameter(
                "body",
                inspect.Parameter.KEYWORD_ONLY,
                default=Body(...) if required else Body(None),
                annotation=body_model,
            )
        )
    return inspect.Signature(parameters, return_annotation=dict[str, Any])


def _make_service_route_handler(
    tool_name: str, input_model: type[BaseModel], path: str, method: str
) -> Any:
    path_params = _path_param_names(path)
    body_model = (
        None if method == "GET" else _body_model_for_route(tool_name, input_model, path_params)
    )

    async def service_route(
        request: Request, service: IlmaMcpService = SERVICE_DEPENDENCY, **route_values: Any
    ) -> dict[str, Any]:
        body = route_values.pop("body", None)
        payload: dict[str, Any] = body.model_dump() if isinstance(body, BaseModel) else {}
        payload.update(route_values)
        try:
            model = input_model(**payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        handler = getattr(service, tool_name)
        return handler(**model.model_dump())

    service_route.__name__ = f"http_{tool_name}"
    cast(Any, service_route).__signature__ = _route_signature(
        input_model=input_model, path=path, method=method, body_model=body_model
    )
    return service_route


def _register_service_routes(app: FastAPI, service: IlmaMcpService | None = None) -> int:
    """Register FastAPI routes for service tools declared in _TOOL_TO_ROUTE."""

    tools = tools_dict(_registration_service(service))
    registered = 0
    for tool_name, spec in sorted(tools.items()):
        if tool_name in _HTTP_EXCLUDED:
            continue
        route = _TOOL_TO_ROUTE.get(tool_name)
        if route is None:
            continue
        path, method = route
        app.add_api_route(
            path,
            _make_service_route_handler(tool_name, spec["input_model"], path, method),
            methods=[method],
            tags=[_route_tag(path)],
        )
        registered += 1
    return registered


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

    _register_service_routes(app, service)

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
