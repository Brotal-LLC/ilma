"""Shared HTTP/MCP hardening helpers.

This module intentionally stays framework-light so the MCP service can reuse the
metrics and observability helpers without depending on FastAPI.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def now_monotonic() -> float:
    """Return a monotonic timestamp for durations/rate limiting."""

    return time.perf_counter()


class SlidingWindowRateLimiter:
    """In-memory per-key sliding-window rate limiter.

    The configured limit is requests per one-second window.  This is deliberately
    process-local and dependency-free; deployments needing global limits should put
    a shared limiter in front of the service.
    """

    def __init__(self, *, requests_per_second: float = 30.0, window_seconds: float = 1.0) -> None:
        self.requests_per_second = float(requests_per_second)
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Return True when ``key`` is still within the configured rate."""

        if self.requests_per_second <= 0:
            return True
        limit = max(1, int(self.requests_per_second))
        current = now_monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(current)
            return True


@dataclass
class _Histogram:
    buckets: tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS
    values: list[float] = field(default_factory=list)


def _labels_key(labels: Mapping[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def _format_labels(labels: Iterable[tuple[str, str]]) -> str:
    parts = list(labels)
    if not parts:
        return ""
    escaped = [
        f'{key}="{value.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34))}"'
        for key, value in parts
    ]
    return "{" + ",".join(escaped) + "}"


class MetricsRegistry:
    """Small in-memory Prometheus-style metrics registry."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._lock = threading.Lock()

    def increment(
        self, name: str, labels: Mapping[str, str] | None = None, amount: int = 1
    ) -> None:
        with self._lock:
            self._counters[(name, _labels_key(labels))] += int(amount)

    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = 0.0
        key = (name, _labels_key(labels))
        with self._lock:
            histogram = self._histograms.setdefault(key, _Histogram())
            histogram.values.append(max(0.0, numeric))

    def set_gauge(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = 0.0
        with self._lock:
            self._gauges[(name, _labels_key(labels))] = numeric

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""

        with self._lock:
            counters = dict(self._counters)
            histograms = {
                key: _Histogram(value.buckets, list(value.values))
                for key, value in self._histograms.items()
            }
            gauges = dict(self._gauges)

        lines = [
            "# TYPE request_count counter",
            "# TYPE tool_call_count counter",
            "# TYPE request_duration histogram",
            "# TYPE tool_call_duration histogram",
            "# TYPE memory_search_latency histogram",
            "# TYPE db_connection_pool_size gauge",
        ]
        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}{_format_labels(labels)} {value}")
        for (name, labels), gauge_value in sorted(gauges.items()):
            lines.append(f"{name}{_format_labels(labels)} {gauge_value:g}")
        for (name, labels), histogram in sorted(histograms.items()):
            values = histogram.values
            for bucket in histogram.buckets:
                count = sum(1 for item in values if item <= bucket)
                bucket_labels = (*labels, ("le", f"{bucket:g}"))
                lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {count}")
            inf_labels = (*labels, ("le", "+Inf"))
            lines.append(f"{name}_bucket{_format_labels(inf_labels)} {len(values)}")
            lines.append(f"{name}_count{_format_labels(labels)} {len(values)}")
            lines.append(f"{name}_sum{_format_labels(labels)} {sum(values):g}")
        return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()


def log_observation(
    observability: Any,
    *,
    level: str,
    message: str,
    source: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Best-effort structured log through an ObservabilityRepo-like object."""

    logger = getattr(observability, "log", None)
    if not callable(logger):
        return
    try:
        logger(level, message, source=source, context=dict(context or {}))
    except Exception:
        return


def pool_size_from_backend(backend: Any) -> int:
    """Best-effort current Postgres pool size for metrics exposition."""

    pool = getattr(backend, "_pool", None)
    if pool is None:
        return 0
    get_stats = getattr(pool, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats()
            if isinstance(stats, Mapping):
                for key in ("pool_size", "pool_available", "connections_num"):
                    if key in stats:
                        return int(stats[key])
        except Exception:
            pass
    for attr in ("max_size", "_max_size", "min_size", "_min_size"):
        value = getattr(pool, attr, None)
        if isinstance(value, int):
            return value
    return 0
