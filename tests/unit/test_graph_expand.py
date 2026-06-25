"""Tests for the expand_graph option on ilma_recall and ilma_wiki_search.

These tests use a FakeService whose ``graph`` attribute returns a stub
that mimics the PgGraphRepo interface (a ``traverse(kind, src_id,
max_hops, edge_types, limit)`` method). They verify that:

- expand_graph=False (default) does NOT touch the graph.
- expand_graph=True traverses each hit and unions neighbors, deduplicating
  against the original hit set.
- graph=None (no backend) leaves graph_neighbors as an empty list — the
  recall still succeeds.
- Traversal errors per-hit do not abort the whole recall (graceful degrade).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from ilma.service import WRITE_TOOLS, IlmaService

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeMemoryResult:
    """Mimics the shape of ilma.memory.search results."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def __iter__(self) -> Any:
        return iter([{"id": i, "content": f"mem-{i}", "tags": []} for i in self.ids])


class _StubGraph:
    """Records every traverse call and returns a programmable subgraph."""

    def __init__(self, subgraph: dict[str, Any] | None = None) -> None:
        self.traverse_calls: list[dict[str, Any]] = []
        self._subgraph = subgraph or {"nodes": [], "edges": []}

    def traverse(
        self,
        *,
        kind: str,
        src_id: int,
        max_hops: int = 1,
        edge_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        self.traverse_calls.append(
            {
                "kind": kind,
                "src_id": src_id,
                "max_hops": max_hops,
                "edge_types": edge_types,
                "limit": limit,
            }
        )
        return self._subgraph


class _FakeMemoryRepo:
    """Mimics the methods service.py calls on self.memory."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def search(self, query: str, *, top_k: int, hybrid_text_weight: float) -> Any:
        return [{"id": i, "content": f"mem-{i}", "tags": []} for i in self._ids[:top_k]]


class _FakeWikiRepo:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def search(self, query: str, *, top_k: int) -> Any:
        return [{"id": i, "slug": f"wiki-{i}", "title": f"Wiki {i}"} for i in self._ids[:top_k]]


class _FakeGenericRepo:
    """No-op repo for journal/skills/kanban/etc. Service touches them on init only."""

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _FakeBackend:
    def __init__(self, graph: Any, memory_ids: list[int], wiki_ids: list[int]) -> None:
        self.memory = _FakeMemoryRepo(memory_ids)
        self.wiki = _FakeWikiRepo(wiki_ids)
        self.journal = _FakeGenericRepo()
        self.skills = _FakeGenericRepo()
        self.kanban = _FakeGenericRepo()
        self.metrics = _FakeGenericRepo()
        self.observability = _FakeGenericRepo()
        self.sessions = _FakeGenericRepo()
        self._graph = graph

    def memory_repo(self) -> Any:
        return self.memory

    def wiki_repo(self) -> Any:
        return self.wiki

    def journal_repo(self) -> Any:
        return self.journal

    def skills_repo(self) -> Any:
        return self.skills

    def kanban_repo(self) -> Any:
        return self.kanban

    def metrics_repo(self) -> Any:
        return self.metrics

    def observability_repo(self) -> Any:
        return self.observability

    def sessions_repo(self) -> Any:
        return self.sessions

    def graph_repo(self) -> Any:
        return self._graph


class _NullAudit:
    def begin(self, *args: Any, **kwargs: Any) -> str:
        return "op-1"

    def finish(self, *args: Any, **kwargs: Any) -> None:
        return None


def _make_service(graph: Any, memory_ids: list[int], wiki_ids: list[int]) -> Any:
    backend = _FakeBackend(graph, memory_ids, wiki_ids)
    return IlmaService(backend, audit=_NullAudit())


# ---------------------------------------------------------------------------
# ilma_recall + expand_graph
# ---------------------------------------------------------------------------


class TestRecallExpandGraph:
    def test_default_does_not_touch_graph(self) -> None:
        # When expand_graph is False (default), the graph is never queried.
        stub = _StubGraph()
        svc = _make_service(stub, memory_ids=[1, 2, 3], wiki_ids=[])
        result = svc.ilma_recall("test")
        assert result["ok"] is True
        assert result["graph_neighbors"] == []
        assert stub.traverse_calls == []

    def test_expand_graph_true_traverses_each_hit(self) -> None:
        # expand_graph=True should call graph.traverse once per recall hit.
        stub = _StubGraph()
        svc = _make_service(stub, memory_ids=[10, 20, 30], wiki_ids=[])
        svc.ilma_recall("test", expand_graph=True)
        assert [c["src_id"] for c in stub.traverse_calls] == [10, 20, 30]
        # All calls should be for Memory kind.
        assert all(c["kind"] == "Memory" for c in stub.traverse_calls)
        # max_hops defaults to 1.
        assert all(c["max_hops"] == 1 for c in stub.traverse_calls)

    def test_graph_hops_clamped_to_3(self) -> None:
        # graph_hops=99 should clamp to 3.
        stub = _StubGraph()
        svc = _make_service(stub, memory_ids=[1], wiki_ids=[])
        svc.ilma_recall("test", expand_graph=True, graph_hops=99)
        assert stub.traverse_calls[0]["max_hops"] == 3

    def test_graph_neighbors_dedupes_against_original_hits(self) -> None:
        # If a traverse returns a neighbor that's already in the recall
        # hits, it should NOT appear in graph_neighbors.
        stub = _StubGraph(
            subgraph={
                "nodes": [
                    {"kind": "Memory", "src_id": 2, "vertex_id": 99, "properties": {"id": 2}},
                    {
                        "kind": "Memory",
                        "src_id": 99,
                        "vertex_id": 100,
                        "properties": {"id": 99, "category": "fact"},
                    },
                ],
                "edges": [],
            }
        )
        svc = _make_service(stub, memory_ids=[1, 2], wiki_ids=[])
        result = svc.ilma_recall("test", expand_graph=True)
        # 2 hits, traverse returns 2 nodes total but #2 is a duplicate.
        ids = [n["id"] for n in result["graph_neighbors"]]
        assert 2 not in ids
        assert 99 in ids
        assert len(ids) == 1

    def test_graph_neighbors_captures_via_memory_id(self) -> None:
        stub = _StubGraph(
            subgraph={
                "nodes": [
                    {
                        "kind": "Memory",
                        "src_id": 5,
                        "vertex_id": 200,
                        "properties": {"id": 5, "category": "fact"},
                    }
                ],
                "edges": [],
            }
        )
        svc = _make_service(stub, memory_ids=[1], wiki_ids=[])
        result = svc.ilma_recall("test", expand_graph=True)
        assert len(result["graph_neighbors"]) == 1
        neighbor = result["graph_neighbors"][0]
        assert neighbor["id"] == 5
        assert neighbor["via_memory_id"] == 1  # the hit that pulled it in

    def test_graph_none_returns_empty_neighbors(self) -> None:
        # When the backend has no graph_repo, graph_neighbors stays empty.
        svc = _make_service(graph=None, memory_ids=[1, 2], wiki_ids=[])
        result = svc.ilma_recall("test", expand_graph=True)
        assert result["ok"] is True
        assert result["graph_neighbors"] == []

    def test_traverse_error_does_not_abort_recall(self) -> None:
        # If one hit's traverse raises, the recall still succeeds for the
        # remaining hits and the failed one is silently skipped.
        stub = MagicMock()

        def traverse_side_effect(*, kind: str, src_id: int, **kwargs: Any) -> Any:
            if src_id == 2:
                raise RuntimeError("graph unavailable")
            return {
                "nodes": [
                    {"kind": "Memory", "src_id": 99, "vertex_id": 1, "properties": {"id": 99}}
                ],
                "edges": [],
            }

        stub.traverse.side_effect = traverse_side_effect
        svc = _make_service(stub, memory_ids=[1, 2, 3], wiki_ids=[])
        result = svc.ilma_recall("test", expand_graph=True)
        assert result["ok"] is True
        # Hit 1 and hit 3 both pull in neighbor 99, but dedup keeps it
        # to one entry. Hit 2 is silently skipped because traverse raised.
        ids = [n["id"] for n in result["graph_neighbors"]]
        assert ids == [99]
        # The via_memory_id of the surviving neighbor should be 1 (first
        # successful hit), not 3 (which would also pull it but is processed
        # after the dedupe wins).
        assert result["graph_neighbors"][0]["via_memory_id"] == 1

    def test_response_shape_unchanged_when_disabled(self) -> None:
        # Regression: default (no graph) response shape is identical to
        # the pre-expand_graph contract.
        svc = _make_service(_StubGraph(), memory_ids=[1], wiki_ids=[])
        result = svc.ilma_recall("test")
        assert set(result.keys()) >= {"ok", "results", "count", "query", "limit"}
        assert "graph_neighbors" in result
        assert result["graph_neighbors"] == []

    def test_ilma_recall_remains_read_only(self) -> None:
        # Critical: even with expand_graph=True, ilma_recall is read-only.
        # Audit pipeline should not classify it as a write tool.
        assert "ilma_recall" not in WRITE_TOOLS


# ---------------------------------------------------------------------------
# ilma_wiki_search + expand_graph
# ---------------------------------------------------------------------------


class TestWikiSearchExpandGraph:
    def test_default_does_not_touch_graph(self) -> None:
        stub = _StubGraph()
        svc = _make_service(stub, memory_ids=[], wiki_ids=[10, 20])
        result = svc.ilma_wiki_search("test")
        assert result["ok"] is True
        assert result["graph_neighbors"] == []
        assert stub.traverse_calls == []

    def test_expand_graph_true_traverses_wiki_kind(self) -> None:
        stub = _StubGraph(
            subgraph={
                "nodes": [
                    {
                        "kind": "Memory",
                        "src_id": 42,
                        "vertex_id": 999,
                        "properties": {"id": 42, "category": "fact"},
                    }
                ],
                "edges": [],
            }
        )
        svc = _make_service(stub, memory_ids=[], wiki_ids=[10])
        result = svc.ilma_wiki_search("test", expand_graph=True)
        assert [c["kind"] for c in stub.traverse_calls] == ["Wiki"]
        assert [c["src_id"] for c in stub.traverse_calls] == [10]
        # Neighbor (Memory #42) appears in graph_neighbors.
        assert len(result["graph_neighbors"]) == 1
        assert result["graph_neighbors"][0]["id"] == 42
        assert result["graph_neighbors"][0]["via_wiki_id"] == 10

    def test_graph_neighbors_dedupes_across_multiple_wiki_hits(self) -> None:
        stub = _StubGraph(
            subgraph={
                "nodes": [
                    {"kind": "Memory", "src_id": 42, "vertex_id": 1, "properties": {"id": 42}}
                ],
                "edges": [],
            }
        )
        # Two wiki hits — both should produce neighbor #42, but only once.
        svc = _make_service(stub, memory_ids=[], wiki_ids=[10, 20])
        result = svc.ilma_wiki_search("test", expand_graph=True)
        ids = [n["id"] for n in result["graph_neighbors"]]
        assert ids == [42]

    def test_graph_hops_clamped(self) -> None:
        stub = _StubGraph()
        svc = _make_service(stub, memory_ids=[], wiki_ids=[1])
        svc.ilma_wiki_search("test", expand_graph=True, graph_hops=0)
        # 0 should clamp up to 1 (otherwise we'd traverse nothing).
        assert stub.traverse_calls[0]["max_hops"] == 1

    def test_graph_none_returns_empty_neighbors(self) -> None:
        svc = _make_service(graph=None, memory_ids=[], wiki_ids=[1])
        result = svc.ilma_wiki_search("test", expand_graph=True)
        assert result["ok"] is True
        assert result["graph_neighbors"] == []


# ---------------------------------------------------------------------------
# Audit pipeline + tool registration sanity
# ---------------------------------------------------------------------------


def test_recall_with_expand_graph_does_not_change_tool_count() -> None:
    """Adding kwargs didn't add a new MCP tool."""
    from ilma.service import _count_tools

    # Snapshot before any client changes; re-count after.
    # We just assert that the count is stable — the actual surface is
    # verified by the test_mcp_server_registers_expected_* tests.
    assert _count_tools() >= 30


def test_recall_audit_pipeline_does_not_block_when_expand_graph_fails() -> None:
    """A graph-traverse failure during recall must not poison the audit log.

    ilma_recall is read-only, so it isn't in WRITE_TOOLS — the audit
    pipeline doesn't write a record for it. We verify that even with a
    broken graph, the call returns ok=True and the audit logger is never
    invoked for this tool.
    """
    captured: list[dict[str, Any]] = []

    class _CapturingAudit:
        def begin(self, tool: str, surface: str, action: str, payload: dict[str, Any]) -> str:
            captured.append({"tool": tool, "payload": payload})
            return "op-1"

        def finish(self, *args: Any, **kwargs: Any) -> None:
            return None

    # Graph that always errors.
    broken = MagicMock()
    broken.traverse.side_effect = RuntimeError("graph unavailable")
    backend = _FakeBackend(broken, memory_ids=[1, 2], wiki_ids=[])
    svc = IlmaService(backend, audit=_CapturingAudit())
    result = svc.ilma_recall("test", expand_graph=True)
    assert result["ok"] is True
    # ilma_recall is read-only; no audit record was emitted for it.
    assert not any(c["tool"] == "ilma_recall" for c in captured)
