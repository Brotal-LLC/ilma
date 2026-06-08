"""Memory repository interface.

The canonical storage contract for agent memories. Implementations:
- PgMemoryRepo (Postgres + pgvector)
- InMemoryRepo (tests, ephemeral)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

#: Hard cap on raw memory content (32 KB). Longer content routes to wiki.
MEMORY_MAX_CHARS = 32 * 1024


@dataclass
class Memory:
    """A single stored memory."""

    id: int
    content: str
    tags: tuple[str, ...] = ()
    category: str | None = None
    source: str | None = None
    embedding_dim: int = 1024
    deleted: bool = False
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_deleted(self) -> None:
        self.deleted = True


class MemoryNotFoundError(Exception):
    """Raised when forget() targets a non-existent memory."""


class RoutingRuleViolationError(Exception):
    """Content exceeds MEMORY_MAX_CHARS; route to wiki instead."""


def routing_rule_message(*, size_bytes: int, cap: int) -> str:
    """Canonical routing-rule error message."""
    return (
        f"Memory size {size_bytes:,} chars exceeds the {cap:,}-char cap "
        f"(MEMORY_MAX_CHARS).\n\n"
        f"Routing rule:\n"
        f"  • MEMORY  — short, durable facts (< 1 screen). Stored via\n"
        f"             memory_remember. Surface: system prompt + searches.\n"
        f"  • WIKI    — long-form, structured, multi-paragraph. Stored via\n"
        f"             wiki_create. Surface: explicit reads, cross-linked.\n"
        f"  • SESSION — never persist; use session_search.\n\n"
        f'Did you mean: wiki_create with category="projects."<name>"?'
    )


class MemoryRepo(Protocol):
    """Abstract memory repository."""

    default_dim: int = 1024

    def remember(
        self,
        content: str,
        *,
        tags: Sequence[str] | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> int:
        """Store a memory. Returns id, or 0 if duplicate."""
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        hybrid_text_weight: float = 0.5,
    ) -> list[Memory]:
        """Hybrid search. Returns matching memories."""
        ...

    def forget(self, memory_id: int) -> bool:
        """Soft-delete a memory."""
        ...

    def status(self) -> dict[str, Any]:
        """Repository stats."""
        ...
