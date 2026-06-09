"""Tests for the canonical ilma_recall tool.

These tests focus on the public surface contract:

- ``ilma_recall(query, limit=10, threshold=0.0, hybrid_text_weight=0.5)`` returns
  a ``{"ok": True, "results": [...], "count": N, "query": ..., "limit": N}`` dict
  where the ``count`` field equals ``len(results)`` and the ``limit`` field
  reflects the post-cap value.
- Calls go through the audit log via ``self.call()`` so the audit pipeline sees
  every recall call.
- The CLI and HTTP surfaces expose the same canonical recall operation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ilma.service import (
    WRITE_TOOLS,
    IlmaService,
    _count_tools,
    _derive_write_tools,
    tools_dict,
)


class _FakeMemoryResult:
    """Duck-typed memory row returned by ``IlmaService.memory.search``."""

    def __init__(
        self,
        memory_id: int,
        content: str,
        score: float | None = None,
    ) -> None:
        self.id = memory_id
        self.content = content
        self.tags: tuple[str, ...] = ()
        self.category: str | None = None
        self.source: str | None = None
        self.embedding_dim: int = 1536
        self.deleted: bool = False
        self.created_at: Any = None
        self.metadata: dict[str, Any] = {}
        self.score = score


class _FakeMemory:
    """In-memory stub for ``IlmaService.memory`` recall."""

    def __init__(self, rows: list[_FakeMemoryResult]) -> None:
        self._rows = rows
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        hybrid_text_weight: float = 0.5,
    ) -> list[_FakeMemoryResult]:
        self.search_calls.append(
            {"query": query, "top_k": top_k, "hybrid_text_weight": hybrid_text_weight}
        )
        return list(self._rows[:top_k])


class _FakeService:
    """Minimal stub matching the surface ``IlmaService`` exposes to its tools."""

    def __init__(self, rows: list[_FakeMemoryResult]) -> None:
        self.memory = _FakeMemory(rows)
        self.audit = MagicMock()
        self.calls: list[dict[str, Any]] = []
        self._record_id = 0

    def call(self, tool_name: str, fn: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self._record_id += 1
        record_id = str(self._record_id)
        self.calls.append({"id": record_id, "tool": tool_name, "payload": payload})
        return fn()

    @staticmethod
    def _filter_by_threshold(
        rows: list[_FakeMemoryResult], threshold: float
    ) -> list[_FakeMemoryResult]:
        return IlmaService._filter_by_threshold(rows, threshold)


@pytest.fixture
def fake_service() -> _FakeService:
    rows = [
        _FakeMemoryResult(1, "User prefers dark mode", score=0.92),
        _FakeMemoryResult(2, "Project uses Postgres + pgvector", score=0.81),
        _FakeMemoryResult(3, "Hermes home is ~/.hermes", score=0.74),
    ]
    return _FakeService(rows)


# ---------------------------------------------------------------------------
# ilma_recall direct method tests
# ---------------------------------------------------------------------------


def test_ilma_recall_returns_results_count_and_query_metadata(
    fake_service: _FakeService,
) -> None:
    """ilma_recall returns a structured envelope with results, count, query, and limit."""

    response = IlmaService.ilma_recall(fake_service, query="dark mode", limit=5)

    assert response["ok"] is True
    assert response["query"] == "dark mode"
    assert response["limit"] == 5
    assert response["count"] == 3
    assert len(response["results"]) == 3
    assert response["results"][0].content == "User prefers dark mode"


def test_ilma_recall_caps_limit_at_500() -> None:
    """ilma_recall applies the standard _limit cap (default 10, max 500)."""

    svc = _FakeService([])
    IlmaService.ilma_recall(svc, query="x", limit=10000)
    assert svc.memory.search_calls[-1]["top_k"] <= 500


def test_ilma_recall_threshold_filters_when_score_attribute_present(
    fake_service: _FakeService,
) -> None:
    """When Memory rows expose a score, threshold filters them out."""

    response = IlmaService.ilma_recall(fake_service, query="anything", threshold=0.85)

    assert response["count"] == 1
    assert response["results"][0].id == 1


def test_ilma_recall_threshold_zero_is_passthrough(fake_service: _FakeService) -> None:
    """threshold=0.0 means no filter; all results pass through."""

    response = IlmaService.ilma_recall(fake_service, query="anything", threshold=0.0)

    assert response["count"] == 3


def test_ilma_recall_threshold_passes_through_when_no_score() -> None:
    """Rows that don't expose a score pass through the threshold filter."""

    rows = [
        _FakeMemoryResult(1, "no-score-1"),
        _FakeMemoryResult(2, "no-score-2"),
    ]
    svc = _FakeService(rows)
    response = IlmaService.ilma_recall(svc, query="x", threshold=0.9)
    assert response["count"] == 2


def test_ilma_recall_passes_hybrid_weight_to_memory_repo(fake_service: _FakeService) -> None:
    """The canonical recall method preserves the hybrid-search tuning knob."""

    IlmaService.ilma_recall(fake_service, query="dark", hybrid_text_weight=0.25)

    assert fake_service.memory.search_calls[-1] == {
        "query": "dark",
        "top_k": 10,
        "hybrid_text_weight": 0.25,
    }


def test_ilma_recall_audit_records_payload(fake_service: _FakeService) -> None:
    """ilma_recall goes through self.call so the audit logger sees it."""

    IlmaService.ilma_recall(fake_service, query="dark", limit=3, threshold=0.5)

    assert len(fake_service.calls) == 1
    record = fake_service.calls[0]
    assert record["tool"] == "ilma_recall"
    assert record["payload"] == {
        "query": "dark",
        "limit": 3,
        "threshold": 0.5,
        "hybrid_text_weight": 0.5,
    }


# ---------------------------------------------------------------------------
# Auto-registration / wiring tests
# ---------------------------------------------------------------------------


def test_recall_command_registered_in_cli_table() -> None:
    """The CLI maps the canonical recall tool to the recall command."""

    from ilma.api.cli import _CLI_TOOL_TO_COMMAND

    assert _CLI_TOOL_TO_COMMAND["ilma_recall"] == "recall"


def test_recall_route_registered_in_http_table() -> None:
    """FastAPI maps ilma_recall -> POST /recall."""

    from ilma.api.http import _TOOL_TO_ROUTE

    assert _TOOL_TO_ROUTE["ilma_recall"] == ("/recall", "POST")


def test_tools_dict_includes_ilma_recall() -> None:
    """The tools registry exposes ilma_recall as a first-class tool."""

    registry = tools_dict(IlmaService)
    assert "ilma_recall" in registry

    input_model = registry["ilma_recall"]["input_model"]
    field_names = set(input_model.model_fields.keys())
    assert field_names == {"query", "limit", "threshold", "hybrid_text_weight"}


def test_recall_is_a_read_tool_not_a_write_tool() -> None:
    """ilma_recall is a read action; must NOT be in WRITE_TOOLS."""

    assert "ilma_recall" not in WRITE_TOOLS
    assert "ilma_recall" not in _derive_write_tools(IlmaService)


def test_tool_count_matches_actual_method_count() -> None:
    """_count_tools is the runtime source of truth for the tool surface size."""

    assert _count_tools() == len(tools_dict(IlmaService))
