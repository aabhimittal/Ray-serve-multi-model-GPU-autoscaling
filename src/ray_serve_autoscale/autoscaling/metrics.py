"""Latency bookkeeping shared by the deployments and the autoscaler.

Each model replica records its per-request service time into a sliding
:class:`LatencyWindow`. The custom autoscaler reads aggregated percentiles from
these windows (surfaced over a Serve handle) to decide whether the p95 latency
justifies adding or removing GPU replicas.

The window is intentionally dependency-free (pure stdlib) so it is cheap to
carry inside every replica and trivial to unit-test.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySnapshot:
    """Immutable view of a window's state at one instant."""

    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rps: float

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "rps": round(self.rps, 3),
        }


class LatencyWindow:
    """Thread-safe sliding window of recent request latencies.

    Samples older than ``window_s`` are evicted lazily on read/write so the
    percentiles always reflect a rolling recent horizon rather than the life of
    the process.
    """

    def __init__(self, window_s: float = 30.0, max_samples: int = 4096) -> None:
        self._window_s = window_s
        self._max_samples = max_samples
        # Each entry is (timestamp, latency_ms).
        self._samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, *, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._samples.append((ts, latency_ms))
            self._evict(ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        samples = self._samples
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def snapshot(self, *, now: float | None = None) -> LatencySnapshot:
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._evict(ts)
            latencies = sorted(v for _, v in self._samples)
            count = len(latencies)
            if count == 0:
                return LatencySnapshot(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rps = count / self._window_s
            return LatencySnapshot(
                count=count,
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                p99_ms=_percentile(latencies, 99),
                max_ms=latencies[-1],
                rps=rps,
            )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    # Nearest-rank: rank = ceil(pct/100 * N), 1-indexed.
    import math

    rank = math.ceil((pct / 100.0) * len(sorted_values))
    idx = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[idx]
