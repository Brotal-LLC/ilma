"""Journal repository interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class JournalEntry:
    id: int
    content: str
    tags: tuple[str, ...] = ()
    created_at: datetime | None = None


class JournalRepo(Protocol):
    def append(self, content: str, *, tags: list[str] | None = None) -> int:
        """Add a journal entry. Returns entry id."""
        ...

    def search(self, query: str, *, top_k: int = 10) -> list[JournalEntry]:
        """Search journal entries."""
        ...

    def recent(self, *, limit: int = 10) -> list[JournalEntry]:
        """Recent entries."""
        ...
