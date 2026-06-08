"""Kanban repository interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: str = "todo"  # todo | in_progress | done
    priority: int = 0
    tags: tuple[str, ...] = ()
    parent_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class KanbanRepo(Protocol):
    def create(
        self,
        title: str,
        *,
        description: str = "",
        status: str = "todo",
        priority: int = 0,
        tags: list[str] | None = None,
        parent_id: int | None = None,
    ) -> int:
        """Create a task. Returns task id."""
        ...

    def get(self, task_id: int) -> Task | None:
        """Fetch a task."""
        ...

    def update(self, task_id: int, **kwargs: Any) -> bool:
        """Update task fields."""
        ...

    def complete(self, task_id: int) -> bool:
        """Mark task as done."""
        ...

    def list_by_status(self, status: str, *, limit: int = 50) -> list[Task]:
        """List tasks by status."""
        ...

    def search(self, query: str, *, top_k: int = 10) -> list[Task]:
        """Search tasks."""
        ...
