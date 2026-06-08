"""Metrics repository interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class Metric:
    id: int
    name: str
    value: float
    labels: dict[str, str]
    recorded_at: datetime


class MetricsRepo(Protocol):
    def record(self, name: str, value: float, *, labels: dict[str, str] | None = None) -> int:
        """Record a metric point. Returns point id."""
        ...

    def query(
        self,
        name: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[Metric]:
        """Query metric history."""
        ...

    def aggregate(self, name: str, *, window: str = "1 hour") -> list[dict[str, Any]]:
        """Aggregate metrics over time windows."""
        ...
