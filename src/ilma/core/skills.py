"""Skills repository interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Skill:
    id: int
    name: str
    content: str
    category: str | None = None
    tags: tuple[str, ...] = ()
    updated_at: datetime | None = None


class SkillsRepo(Protocol):
    def upsert(
        self, name: str, content: str, *, category: str | None = None, tags: list[str] | None = None
    ) -> int:
        """Store or update a skill. Returns skill id."""
        ...

    def get(self, name: str) -> Skill | None:
        """Fetch a skill by name."""
        ...

    def search(self, query: str, *, top_k: int = 5) -> list[Skill]:
        """Search skills."""
        ...
