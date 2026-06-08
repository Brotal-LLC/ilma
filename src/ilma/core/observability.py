"""Observability repository interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


@dataclass
class Observation:
    id: int
    level: str  # debug | info | warn | error
    message: str
    source: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime | None = None


class ObservabilityRepo(Protocol):
    def log(self, level: str, message: str, *, source: str | None = None, context: dict[str, Any] | None = None) -> int:
        """Log an observation. Returns observation id."""
        ...

    def query(self, *, level: str | None = None, source: str | None = None, start: datetime | None = None, end: datetime | None = None, limit: int = 100) -> list[Observation]:
        """Query observations."""
        ...
