"""Minimal 2D N-body simulation core (no plotting).

Simple API:

from gravitational_system import gravitational_system

s = gravitational_system(positions=[(x1,y1),...],
                         velocities=[(vx1,vy1),...],
                         masses=[m1,...])
s.change(total_time, steps)

Note: plotting and animation have been moved to `painting.py`.
Use painting.plot_trajectory/painting.animate/painting.animate_build.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Optional, Iterable

import numpy as np
from integrators import rk4_2nd_order


# Gravitational constant (dimensionless units: G = 1)
G = 1.0


class gravitational_system:
    """2D N-body gravitational system (simulation only).

    Uses dimensionless units with G = 1:
    - positions: list of (x, y) in length units
    - velocities: list of (vx, vy) in length/time units
    - masses: list of masses (typically normalized so total mass = 1)
    - time: in dynamical units sqrt(r^3 / GM)
    """

    def __init__(self,
                 positions: Sequence[Sequence[float]],
                 velocities: Sequence[Sequence[float]],
                 masses: Sequence[float]):
        if not (len(positions) == len(velocities) == len(masses)):
            raise ValueError("positions, velocities, masses must have the same length")
        self.N = len(masses)
        self.r = np.asarray(positions, dtype=float).reshape(self.N, 2)
        self.v = np.asarray(velocities, dtype=float).reshape(self.N, 2)
        self.m = np.asarray(masses, dtype=float).reshape(self.N)
        self.time = 0.0
        self.history: Optional[np.ndarray] = None  # shape: (steps+1, N, 2)
        self._dt: Optional[float] = None

    def change(self, total_time: float, steps: int = 1000):
        """Integrate for `total_time` using `steps` and RK4 (via integrator)."""
        dt = float(total_time) / int(steps)
        N = self.N
        r = self.r.copy()
        v = self.v.copy()
        hist = np.zeros((steps + 1, N, 2), dtype=float)
        hist[0] = r

        accel = lambda rr: _accelerations(rr, self.m)
        for k in range(steps):
            r, v = rk4_2nd_order(r, v, dt, accel)
            hist[k + 1] = r

        self.r, self.v = r, v
        self.history = hist
        self.time += total_time
        self._dt = dt
        return self


def _accelerations(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Compute pairwise gravitational accelerations in 2D.

    r: (N, 2), m: (N,) -> a: (N, 2)
    """
    N = r.shape[0]
    a = np.zeros_like(r)
    for i in range(N):
        ai = np.zeros(r.shape[1], dtype=float)
        for j in range(N):
            if i == j:
                continue
            d = r[j] - r[i]
            dist = float(np.linalg.norm(d))
            if dist == 0.0:
                continue
            inv_dist3 = 1.0 / (dist**3)
            ai += G * m[j] * d * inv_dist3
        a[i] = ai
    return a
