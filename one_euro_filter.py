import math
import time
from typing import Tuple, Optional


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[float] = None
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s

    def reset(self):
        self.y = None
        self.s = None


class OneEuroFilter:
    """
    Standard One Euro Filter (Casiez et al., CHI 2012).
    Minimizes jitter at low speeds (via min_cutoff) while maintaining low latency
    during high-speed sweeps (via beta).
    """

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt) if dt > 0 else 1.0

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()

        if self.last_time is None:
            dt = 1.0 / self.freq
        else:
            dt = max(timestamp - self.last_time, 1e-6)

        self.last_time = timestamp

        # Estimate derivative (velocity)
        prev_x = self.x_filt.s
        dx = 0.0 if prev_x is None else (x - prev_x) / dt
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(cutoff, dt))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


class DeadbandOneEuroFilter2D:
    """
    2D coordinate filter combining:
    1. Tunable spatial deadband radius to lock micro-tremors when resting.
    2. Independent OneEuroFilters for X and Y for continuous, low-latency trajectory.
    """

    def __init__(
        self,
        deadband_radius: float = 12.0,
        min_cutoff: float = 0.8,
        beta: float = 0.008,
        d_cutoff: float = 1.0,
    ):
        self.deadband_radius = deadband_radius
        self.anchor_x: Optional[float] = None
        self.anchor_y: Optional[float] = None

        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def filter(
        self, raw_x: float, raw_y: float, timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.time()

        if self.anchor_x is None or self.anchor_y is None:
            self.anchor_x = raw_x
            self.anchor_y = raw_y

        # Calculate distance to current deadband anchor
        dist = math.hypot(raw_x - self.anchor_x, raw_y - self.anchor_y)

        if dist < self.deadband_radius:
            # Inside deadband: keep input pinned to anchor to eliminate micro-jitter
            input_x, input_y = self.anchor_x, self.anchor_y
        else:
            # Exited deadband: pull anchor toward raw position to maintain continuity
            pull = (dist - self.deadband_radius) / max(dist, 1e-6)
            self.anchor_x += (raw_x - self.anchor_x) * pull
            self.anchor_y += (raw_y - self.anchor_y) * pull
            input_x, input_y = self.anchor_x, self.anchor_y

        smooth_x = self.filter_x.filter(input_x, timestamp)
        smooth_y = self.filter_y.filter(input_y, timestamp)
        return smooth_x, smooth_y

    def reset(self):
        self.anchor_x = None
        self.anchor_y = None
        self.filter_x.reset()
        self.filter_y.reset()
