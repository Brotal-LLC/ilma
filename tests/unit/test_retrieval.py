"""Tests for ilma.core.retrieval — injection layer and query sanitizer."""

from __future__ import annotations

from typing import Any

import pytest

from ilma.core.memory import Memory, MemoryRepo
from ilma.core.retrieval import (
    InjectionLayer,
    _format_bullet,
    _sanitize_query,
    _score_memory,
    build_memory_block,
)


class FakeMemoryRepo(MemoryRepo):
    """In-memory fake for injection layer tests."""

    default_dim: int = 1024

    def __init__(self, memories: list[Memory] | None = None) -> None:
        self._memories = memories or []
        self._next_id = max((m.id for m in self._memories), default=0) + 1

    def remember(self, content: str, *, tags=None, category=None, source=None) -> int:
        mid = self._next_id
        self._next_id += 1
        self._memories.append(
            Memory(
                id=mid,
                content=content,
                tags=tuple(tags or ()),
                category=category,
                source=source,
            )
        )
        return mid

    def search(
        self, query: str, *, top_k: int = 10, hybrid_text_weight: float = 0.5
    ) -> list[Memory]:
        # Match if ANY word in the query appears in content, tags, or category
        words = [w for w in query.lower().split() if len(w) > 2]
        if not words:
            return list(self._memories)[:top_k]
        results = []
        for m in self._memories:
            text = m.content.lower()
            tags = [t.lower() for t in m.tags]
            cat = (m.category or "").lower()
            if any(w in text or w in cat or any(w in t for t in tags) for w in words):
                results.append(m)
        return results[:top_k]

    def forget(self, memory_id: int) -> bool:
        for m in self._memories:
            if m.id == memory_id:
                m.mark_deleted()
                return True
        return False

    def status(self) -> dict[str, Any]:
        return {"live_memories": len([m for m in self._memories if not m.deleted])}


# ---------------------------------------------------------------------------
# Query sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("system: user prefers dark mode", "user prefers dark mode"),
        ("developer - fix the bug", "fix the bug"),
        ("<tool>search</tool> results", "search results"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("clean query", "clean query"),
    ],
)
def test_sanitize_query(raw: str, expected: str) -> None:
    assert _sanitize_query(raw) == expected


# ---------------------------------------------------------------------------
# Memory scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "category", "expected_min"),
    [
        (("user", "preference"), "identity", 3.5),
        (("user",), None, 1.5),
        (("project",), None, 1.0),
        ((), None, 0.0),
    ],
)
def test_score_memory(tags: tuple[str, ...], category: str | None, expected_min: float) -> None:
    m = Memory(id=1, content="test", tags=tags, category=category)
    assert _score_memory(m) >= expected_min


# ---------------------------------------------------------------------------
# Bullet formatting
# ---------------------------------------------------------------------------


def test_format_bullet_short() -> None:
    m = Memory(id=1, content="short", tags=("a", "b"))
    assert _format_bullet(m) == "• [a b] short"


def test_format_bullet_long_truncation() -> None:
    m = Memory(id=1, content="x" * 200, tags=())
    bullet = _format_bullet(m)
    assert bullet.endswith("…")
    assert len(bullet) <= 145  # "• " + 140 chars + "…"


# ---------------------------------------------------------------------------
# InjectionLayer.render
# ---------------------------------------------------------------------------


def test_render_empty_repo() -> None:
    repo = FakeMemoryRepo()
    layer = InjectionLayer(char_limit=500)
    block = layer.render(repo)
    assert "no memories yet" in block
    assert "live: 0" in block


def test_render_single_memory() -> None:
    repo = FakeMemoryRepo()
    repo.remember("User prefers dark mode", tags=["user", "preference"], category="identity")
    layer = InjectionLayer(char_limit=500)
    block = layer.render(repo)
    assert "dark mode" in block
    assert "live: 1" in block


def test_render_priority_ordering() -> None:
    repo = FakeMemoryRepo()
    repo.remember("Zzz low priority", tags=[])
    repo.remember("User prefers dark mode", tags=["user", "preference"], category="identity")
    repo.remember("Project uses Postgres", tags=["project"])
    layer = InjectionLayer(char_limit=1000)
    block = layer.render(repo)
    # Identity + preference should appear before plain project
    lines = [ln for ln in block.splitlines() if ln.startswith("•")]
    assert "dark mode" in lines[0]


def test_render_char_limit_truncation() -> None:
    repo = FakeMemoryRepo()
    for i in range(50):
        repo.remember(f"Memory {i}: " + "x" * 100, tags=["user"])
    layer = InjectionLayer(char_limit=800)
    block = layer.render(repo)
    assert len(block) <= 800 + 50  # small tolerance for header variance
    assert "more memories available" in block or len(block) < 800


def test_render_no_repo() -> None:
    layer = InjectionLayer()
    assert "no memory store wired" in layer.render(None)


# ---------------------------------------------------------------------------
# build_memory_block wrapper
# ---------------------------------------------------------------------------


def test_build_memory_block_signature() -> None:
    repo = FakeMemoryRepo()
    repo.remember("test user context", tags=[])
    block = build_memory_block(repo, char_limit=500)
    assert "test" in block
    assert "MEMORY (your personal notes)" in block
