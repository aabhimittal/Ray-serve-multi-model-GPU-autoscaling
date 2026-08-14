"""Signal conditioning for the autoscaling control loop.

A control loop is only as good as the signal driving it. In production the raw
p95 feed misbehaves in ways that make a naive controller dangerous:

* a single 10-second GC pause or a cold cache produces a spike that is gone
  before new replicas finish booting -- reacting to it wastes GPUs;
* a metrics endpoint that silently freezes keeps returning the *last* value,
  so the controller happily "holds" while the fleet melts down;
* alternating scale-out/scale-in decisions (flapping) cost more than either
  choice, because every GPU replica pays a cold start on the way in;
* a NaN, an inf, or a negative duration from a clock that jumped will happily
  propagate through arithmetic and produce a nonsense replica count.

This module turns the raw feed into a signal a controller can trust:
:class:`SignalTracker` per model, smoothing with EWMA, estimating the latency
*trend* (so the controller can look ahead over the cold-start horizon),
detecting staleness, and damping flap. Everything here is pure stdlib and
deterministic, so the nasty cases are unit-testable without a cluster.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ray_serve_autoscale.settings import SignalConfig


def sanitize(value: object, *, default: float = 0.0, allow_negative: bool = False) -> float:
    """Coerce an arbitrary observed value into a finite, usable float.

    Metrics arrive over HTTP from another process; anything can show up. NaN
    and +/-inf poison every downstream comparison (``nan >= x`` is False, so a
    NaN latency silently reads as "healthy"), and a negative duration means a
    clock moved backwards. All of them collapse to ``default``.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    if out < 0 and not allow_negative:
        return default
    return out


def finite_or_none(value: object) -> Optional[float]:
    """Return ``value`` as a finite non-negative float, or None if unusable.

    Distinct from :func:`sanitize` on purpose. Coercing a bad *latency* to 0.0
    would read as "perfectly healthy" and could trigger a scale-in during an
    outage -- the worst possible response. For signal ingestion we drop the
    sample instead, so the estimator simply keeps its previous belief.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out) or out < 0:
        return None
    return out


class Ewma:
    """Exponentially weighted moving average with a bias-corrected warmup.

    Plain EWMA initialised at zero reads far too low for the first several
    samples -- long enough for a controller to miss a real ramp. We divide by
    the accumulated weight so the very first sample reads as itself.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        # Clamp: alpha<=0 would freeze the estimate, alpha>1 would oscillate.
        self._alpha = min(max(alpha, 1e-3), 1.0)
        self._raw = 0.0
        self._weight = 0.0

    def update(self, value: object) -> float:
        clean = finite_or_none(value)
        if clean is None:
            # Keep the previous belief rather than letting garbage move it.
            return self.value
        self._raw = self._alpha * clean + (1 - self._alpha) * self._raw
        self._weight = self._alpha + (1 - self._alpha) * self._weight
        return self.value

    @property
    def value(self) -> float:
        if self._weight <= 0:
            return 0.0
        return self._raw / self._weight

    @property
    def initialized(self) -> bool:
        return self._weight > 0


class TrendEstimator:
    """Least-squares slope of latency over time, in **ms per second**.

    This is what makes scaling *predictive* rather than reactive: a positive
    slope means the current p95 understates where latency will be by the time
    a new replica finishes its cold start.
    """

    def __init__(self, window_s: float = 90.0, max_points: int = 256) -> None:
        self._window_s = max(window_s, 1.0)
        self._points: deque[tuple[float, float]] = deque(maxlen=max_points)

    def add(self, timestamp: float, value: object) -> None:
        clean = finite_or_none(value)
        if clean is None:
            return
        # A clock that jumped backwards would invert the regression; drop the
        # history rather than emit a bogus negative trend.
        if self._points and timestamp < self._points[-1][0]:
            self._points.clear()
        self._points.append((timestamp, clean))
        cutoff = timestamp - self._window_s
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def slope_ms_per_s(self) -> float:
        """Return the fitted slope, or 0.0 when there is not enough signal."""
        n = len(self._points)
        if n < 3:
            return 0.0
        t0 = self._points[0][0]
        xs = [t - t0 for t, _ in self._points]
        ys = [v for _, v in self._points]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom <= 1e-9:  # all samples at (effectively) the same instant
            return 0.0
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        return num / denom

    def project(self, current: float, horizon_s: float) -> float:
        """Extrapolate ``current`` forward by ``horizon_s`` along the trend."""
        current = sanitize(current)
        horizon_s = max(sanitize(horizon_s), 0.0)
        projected = current + self.slope_ms_per_s() * horizon_s
        # Latency cannot go negative, and a falling trend must never project
        # below zero and read as "infinitely healthy".
        return max(projected, 0.0)


class StalenessDetector:
    """Flags a metrics feed that has stopped advancing.

    A frozen exporter keeps serving its last payload. The counts stop moving
    while the values look perfectly plausible -- the single most dangerous
    failure mode for a closed-loop controller, because it looks like health.
    We key on the monotonically increasing request count rather than the
    latency value (identical latencies are legitimate; identical *counts*
    under live traffic are not).
    """

    def __init__(self, staleness_ticks: int = 3) -> None:
        self._limit = max(staleness_ticks, 1)
        self._last_count: Optional[int] = None
        self._repeats = 0

    def observe(self, sample_count: int) -> bool:
        """Record an observation; return True when the feed looks stale."""
        if self._last_count is not None and sample_count == self._last_count:
            self._repeats += 1
        else:
            self._repeats = 0
        self._last_count = sample_count
        return self.is_stale

    @property
    def is_stale(self) -> bool:
        return self._repeats >= self._limit

    def reset(self) -> None:
        self._last_count = None
        self._repeats = 0


class FlapDamper:
    """Detects scale direction oscillation and widens the dead band.

    Flapping is expensive in a GPU fleet: each cycle pays a cold start and
    evicts a warm CUDA context. When a model changes direction repeatedly we
    do not freeze it outright (that would strand a genuine ramp) -- we widen
    the dead band so only a *decisively* out-of-band signal moves it.
    """

    def __init__(self, window: int = 6, threshold: int = 3) -> None:
        self._directions: deque[int] = deque(maxlen=max(window, 2))
        self._threshold = max(threshold, 1)

    def record(self, direction: int) -> None:
        """Record a scaling action: +1 scale-out, -1 scale-in, 0 hold."""
        if direction != 0:
            self._directions.append(1 if direction > 0 else -1)

    @property
    def reversals(self) -> int:
        d = list(self._directions)
        return sum(1 for a, b in zip(d, d[1:]) if a != b)

    @property
    def is_flapping(self) -> bool:
        return self.reversals >= self._threshold

    def reset(self) -> None:
        self._directions.clear()


@dataclass(frozen=True)
class ConditionedSignal:
    """The trustworthy view of one model's latency the controller acts on."""

    name: str
    raw_p95_ms: float
    smoothed_p95_ms: float
    projected_p95_ms: float
    trend_ms_per_s: float
    sample_count: int
    is_stale: bool
    is_flapping: bool
    has_estimate: bool = True

    @property
    def trustworthy(self) -> bool:
        """Whether this signal may drive a scaling action at all.

        Requires a live feed (not stale), traffic to measure (samples), and an
        estimator that has actually seen a usable value -- a model whose feed
        has only ever produced garbage has no belief to act on.
        """
        return not self.is_stale and self.sample_count > 0 and self.has_estimate


class SignalTracker:
    """Per-model signal state: smoothing, trend, staleness and flap damping."""

    def __init__(self, name: str, cfg: SignalConfig) -> None:
        self.name = name
        self._cfg = cfg
        self._ewma = Ewma(cfg.ewma_alpha)
        self._trend = TrendEstimator(cfg.trend_window_s)
        self._staleness = StalenessDetector(cfg.staleness_ticks)
        self._flap = FlapDamper(cfg.flap_window, cfg.flap_threshold)

    def update(
        self,
        *,
        p95_ms: object,
        sample_count: object,
        timestamp: float,
        horizon_s: float,
    ) -> ConditionedSignal:
        clean = finite_or_none(p95_ms)
        count = int(sanitize(sample_count))
        stale = self._staleness.observe(count)

        # A stale feed must not be folded into the smoothed estimate, or the
        # frozen value would slowly become "the truth". Unusable values are
        # dropped for the same reason (see finite_or_none).
        if not stale and clean is not None:
            self._ewma.update(clean)
            self._trend.add(timestamp, clean)

        smoothed = self._ewma.value
        return ConditionedSignal(
            name=self.name,
            raw_p95_ms=clean if clean is not None else 0.0,
            smoothed_p95_ms=smoothed,
            projected_p95_ms=self._trend.project(smoothed, horizon_s),
            trend_ms_per_s=self._trend.slope_ms_per_s(),
            sample_count=count,
            is_stale=stale,
            is_flapping=self._flap.is_flapping,
            has_estimate=self._ewma.initialized,
        )

    def record_action(self, direction: int) -> None:
        self._flap.record(direction)

    def deadband_multiplier(self) -> float:
        """Dead-band widening currently in force (1.0 = normal)."""
        return self._cfg.flap_deadband_widening if self._flap.is_flapping else 1.0
