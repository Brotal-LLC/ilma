from __future__ import annotations

from typing import Any

import pytest
from fastapi.routing import APIRoute

from ilma.api.http import _HTTP_EXCLUDED, _TOOL_TO_ROUTE, create_app
from ilma.service import tools_dict

from .test_http import FakeHttpService


def _registered_routes(app: Any) -> dict[tuple[str, str], APIRoute]:
    routes: dict[tuple[str, str], APIRoute] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or set():
            if method == "HEAD":
                continue
            routes[(route.path, method)] = route
    return routes


@pytest.mark.parametrize("tool_name,route", sorted(_TOOL_TO_ROUTE.items()))
def test_tool_route_table_entries_are_registered(tool_name: str, route: tuple[str, str]) -> None:
    app = create_app(FakeHttpService())
    path, method = route

    registered_routes = _registered_routes(app)

    assert (path, method) in registered_routes, tool_name
    status_code = registered_routes[(path, method)].status_code or 200
    assert status_code == 200


def test_http_route_table_covers_service_tools() -> None:
    service = FakeHttpService()
    tool_names = set(tools_dict(service))

    assert set(_TOOL_TO_ROUTE) <= tool_names
    assert tool_names <= set(_TOOL_TO_ROUTE) | set(_HTTP_EXCLUDED)
