from __future__ import annotations

import numpy as np
from typing import Callable


def rk4_2nd_order(r: np.ndarray,
                  v: np.ndarray,
                  dt: float,
                  accel: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Single RK4 step for second-order system r'' = a(r).

    r: (N, D), v: (N, D)
    accel: function mapping r -> a(r) of shape (N, D)
    returns: (r_next, v_next)
    """
    a1 = accel(r)
    K1r = v
    K1v = a1

    a2 = accel(r + 0.5 * dt * K1r)
    K2r = v + 0.5 * dt * K1v
    K2v = a2

    a3 = accel(r + 0.5 * dt * K2r)
    K3r = v + 0.5 * dt * K2v
    K3v = a3

    a4 = accel(r + dt * K3r)
    K4r = v + dt * K3v
    K4v = a4

    r_next = r + (dt / 6.0) * (K1r + 2 * K2r + 2 * K3r + K4r)
    v_next = v + (dt / 6.0) * (K1v + 2 * K2v + 2 * K3v + K4v)
    return r_next, v_next
