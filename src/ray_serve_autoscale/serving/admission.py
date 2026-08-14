"""Admission control: load shedding and circuit breaking.

Autoscaling is not instant. Between the moment demand spikes and the moment new
GPU replicas are warm there is a window where the fleet simply cannot serve
everything arriving. What happens in that window decides whether the service
degrades gracefully or collapses.

The failure mode to avoid is *congestive collapse*: every request is accepted,
queues grow without bound, and by the time a request reaches a GPU its caller
has already timed out. The system then spends 100% of its GPU time computing
answers nobody is waiting for -- throughput of useful work goes to zero exactly
when it is needed most.

Two mechanisms prevent that:

**Load shedding.** Estimate the wait a new request would face from the queue
depth and the observed service rate. If that wait already exceeds the SLO by
``shed_at_slo_fraction``, the request is doomed -- reject it *immediately* with
a 503 and ``Retry-After`` instead of burning GPU time on it. Rejecting fast
keeps the admitted traffic inside its SLO; a fast, honest failure is worth far
more to a caller than a slow one.

**Circuit breaking.** When a model's replicas start failing (GPU OOM, a wedged
CUDA context, a crash-looping deployment) the breaker opens and requests fail
instantly rather than piling onto a broken backend. After a cooldown it admits
a few probes; success closes it, failure re-opens it.

Both are pure state machines driven by an injected clock, so every transition
is deterministically testable.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from ray_serve_autoscale.autoscaling.signals import sanitize
from ray_serve_autoscale.settings import AdmissionConfig

# Circuit breaker states.
CLOSED = "closed"  # healthy: traffic flows
OPEN = "open"  # failing: reject immediately
HALF_OPEN = "half_open"  # probing: admit a trickle to test recovery

CircuitState = str


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of an admission check for a single request."""

    admitted: bool
    reason: str = "ok"
    retry_after_s: float = 0.0
    estimated_wait_ms: float = 0.0

    @property
    def status_code(self) -> int:
        return 200 if self.admitted else 503


class CircuitBreaker:
    """Per-model breaker over a rolling window of request outcomes.

    Uses an error *rate* over a minimum sample count rather than a raw
    consecutive-failure count: at high concurrency a handful of consecutive
    failures is noise, while a sustained 50% error rate is an outage.
    """

    def __init__(
        self,
        cfg: AdmissionConfig,
        clock: Callable[[], float],
        window_size: int = 200,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        # Each entry is (timestamp, ok?).
        self._outcomes: deque[tuple[float, bool]] = deque(maxlen=window_size)
        self._state: CircuitState = CLOSED
        self._opened_at = 0.0
        self._probes_left = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Promote OPEN -> HALF_OPEN once the cooldown has elapsed."""
        if self._state == OPEN:
            if self._clock() - self._opened_at >= self._cfg.breaker_open_s:
                self._state = HALF_OPEN
                self._probes_left = max(self._cfg.breaker_half_open_probes, 1)

    def allow(self) -> bool:
        """Whether a request may proceed under the current breaker state."""
        with self._lock:
            self._maybe_half_open()
            if self._state == CLOSED:
                return True
            if self._state == OPEN:
                return False
            # HALF_OPEN: admit a bounded number of probes.
            if self._probes_left > 0:
                self._probes_left -= 1
                return True
            return False

    def record(self, ok: bool) -> None:
        """Feed a request outcome back into the breaker."""
        with self._lock:
            now = self._clock()
            self._outcomes.append((now, ok))

            if self._state == HALF_OPEN:
                # A probe decides recovery outright -- no averaging needed.
                if ok:
                    self._state = CLOSED
                    self._outcomes.clear()
                else:
                    self._state = OPEN
                    self._opened_at = now
                return

            if self._state == CLOSED and self._should_open():
                self._state = OPEN
                self._opened_at = now

    def _should_open(self) -> bool:
        total = len(self._outcomes)
        if total < max(self._cfg.breaker_min_requests, 1):
            return False
        failures = sum(1 for _, ok in self._outcomes if not ok)
        return (failures / total) >= self._cfg.error_rate_threshold

    def error_rate(self) -> float:
        with self._lock:
            if not self._outcomes:
                return 0.0
            failures = sum(1 for _, ok in self._outcomes if not ok)
            return failures / len(self._outcomes)


def estimate_wait_ms(
    queue_depth: int,
    replicas: int,
    service_time_ms: float,
    max_concurrent_per_replica: int = 1,
) -> float:
    """Estimate how long a newly-arrived request would wait before service.

    A queue of depth ``Q`` drains at ``replicas x concurrency / service_time``,
    so the wait is ``Q x service_time / (replicas x concurrency)``. With no
    serving replicas the wait is unbounded -- reported as ``inf`` so callers
    shed rather than divide by zero.
    """
    queue_depth = max(int(sanitize(queue_depth)), 0)
    replicas = max(int(sanitize(replicas)), 0)
    service_time_ms = max(sanitize(service_time_ms), 0.0)
    concurrency = max(int(sanitize(max_concurrent_per_replica)), 1)

    if queue_depth <= 0:
        return 0.0
    if replicas <= 0:
        return math.inf
    return (queue_depth * service_time_ms) / (replicas * concurrency)


class AdmissionController:
    """Combines queue-wait shedding with a circuit breaker for one model."""

    def __init__(
        self,
        model_name: str,
        slo_ms: float,
        cfg: AdmissionConfig,
        shed_at_slo_fraction: float = 1.5,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        import time

        self.model_name = model_name
        self._slo_ms = max(sanitize(slo_ms), 0.0)
        self._cfg = cfg
        self._shed_fraction = max(sanitize(shed_at_slo_fraction), 1.0)
        self._clock = clock or time.monotonic
        self.breaker = CircuitBreaker(cfg, self._clock)
        self.shed_count = 0
        self.admitted_count = 0

    def check(
        self,
        *,
        queue_depth: int,
        replicas: int,
        service_time_ms: float,
        max_concurrent_per_replica: int = 1,
    ) -> AdmissionDecision:
        """Decide whether to admit one request."""
        if not self._cfg.enabled:
            self.admitted_count += 1
            return AdmissionDecision(True, "admission control disabled")

        if not self.breaker.allow():
            self.shed_count += 1
            return AdmissionDecision(
                False,
                f"circuit breaker {self.breaker.state}",
                retry_after_s=self._cfg.breaker_open_s,
            )

        # Hard cap: an unbounded queue is a memory leak as much as a latency
        # problem, so it is enforced independently of the wait estimate.
        cap = self._cfg.max_queue_depth_per_replica * max(replicas, 1)
        if queue_depth > cap:
            self.shed_count += 1
            return AdmissionDecision(
                False,
                f"queue depth {queue_depth} over cap {cap}",
                retry_after_s=1.0,
            )

        wait = estimate_wait_ms(
            queue_depth, replicas, service_time_ms, max_concurrent_per_replica
        )
        budget = self._slo_ms * self._shed_fraction
        if self._slo_ms > 0 and wait > budget:
            self.shed_count += 1
            retry = 1.0 if math.isinf(wait) else min(max(wait / 1000.0, 0.1), 30.0)
            return AdmissionDecision(
                False,
                f"projected wait {wait:.0f}ms exceeds {budget:.0f}ms budget",
                retry_after_s=retry,
                estimated_wait_ms=wait,
            )

        self.admitted_count += 1
        return AdmissionDecision(True, "ok", estimated_wait_ms=wait)

    def record_result(self, ok: bool) -> None:
        self.breaker.record(ok)

    def stats(self) -> dict:
        total = self.admitted_count + self.shed_count
        return {
            "model": self.model_name,
            "admitted": self.admitted_count,
            "shed": self.shed_count,
            "shed_rate": round(self.shed_count / total, 4) if total else 0.0,
            "breaker_state": self.breaker.state,
            "error_rate": round(self.breaker.error_rate(), 4),
        }
