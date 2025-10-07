import numpy as np
from stellar_system import stellar_system


def two():
    """Sun–Earth (2D) demo with animation."""
    positions = [
        (0.0, 0.0),
        (1.5e11, 0.0),
    ]
    velocities = [
        (0.0, 0.0),
        (0.0, 29780.0),
    ]
    masses = [1.989e30, 5.972e24]

    s = stellar_system(positions, velocities, masses)
    s.change(3.15e7, 2000)
    s.animate(interval=30)
    return s


def three():
    """Sun–Earth–Moon (approx 2D) demo."""
    G = 6.67428e-11
    sun_m = 2.0e30
    earth_m = 6.0e24
    moon_m = 7.3e22

    r_se = 1.5e11
    r_em = 4.0e8
    v_e = np.sqrt(G * sun_m / r_se)
    v_m_rel = np.sqrt(G * earth_m / r_em)

    positions = [
        (0.0, 0.0),
        (r_se, 0.0),
        (r_se + r_em, 0.0),
    ]
    velocities = [
        (0.0, 0.0),
        (0.0, v_e),
        (0.0, v_e + v_m_rel),
    ]
    masses = [sun_m, earth_m, moon_m]

    s = stellar_system(positions, velocities, masses)
    s.change(1.0e7, 4000)
    s.animate(interval=20)
    return s
