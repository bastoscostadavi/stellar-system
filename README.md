# Stellar System (2D) — Minimal N‑body Simulation

A tiny, self‑contained Python simulation of 2D N‑body gravity with RK4 integration, static trajectory plots, and simple animation.

## Overview

This project implements a gravitational N-body problem solver using the 4th-order Runge-Kutta method for accurate numerical integration. It can simulate the orbital mechanics of multiple celestial bodies (planets, moons, stars) under their mutual gravitational influences, following Newton's law of universal gravitation.

## Features

- 2D N‑body gravitational simulation (Newtonian, pairwise forces)
- 4th‑order Runge–Kutta (RK4) fixed‑step integrator
- Static trajectory plotting and simple 2D animation
- Minimal API: positions, velocities, masses in SI units

## Project Structure

```
stellar_system/
├── README.md
├── stellar_system.py   # Single-file implementation (class + plotting + animation)
└── test.py            # Small examples (2D)
```

## API

All in `stellar_system.py`:
- Class `stellar_system(positions, velocities, masses)`
  - `positions`: list of `(x, y)` in meters
  - `velocities`: list of `(vx, vy)` in m/s
  - `masses`: list of masses in kg
  - `change(total_time, steps=1000)`: integrate with RK4
  - `trajectory()`: plot static trajectories
  - `animate(interval=30, trail=None)`: animate motion with optional trailing path

## Installation

### Requirements
- Python 3.6+
- NumPy
- Matplotlib

### Setup
```bash
# Clone or download the project
cd stellar_system

# Install dependencies
pip install numpy matplotlib
```

## Usage

### Basic Example (2D)
```python
import numpy as np
from stellar_system import stellar_system

# Sun–Earth in the plane (meters, m/s, kg)
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
s.change(3.15e7, 2000)  # ~1 year, 2000 steps
s.animate(interval=30)
```

## Mathematical Foundation

### Equations of Motion
The system solves the gravitational N-body problem:

```
d²r_i/dt² = G Σ(j≠i) m_j(r_j - r_i)/|r_j - r_i|³
```

Where:
- `r_i`: position vector of body i
- `m_i`: mass of body i  
- `G`: gravitational constant (6.67428 × 10⁻¹¹ m³/kg⋅s²)

### Numerical Integration
The 4th-order Runge-Kutta method provides high accuracy:
- **Error**: O(dt⁴)
- **Stability**: λ·dt ∈ (-2.71, 0)
- **Stages**: 4 per time step

## Input Format

- Positions: `[(x1, y1), ..., (xN, yN)]` in meters
- Velocities: `[(vx1, vy1), ..., (vxN, vyN)]` in m/s
- Masses: `[m1, ..., mN]` in kg

## Physical Constants

- **Gravitational constant**: G = 6.67428 × 10⁻¹¹ m³/(kg⋅s²)
- **Typical scales**:
  - Earth orbital radius: ~1.5 × 10¹¹ m
  - Earth orbital velocity: ~29,780 m/s
  - Solar mass: ~2 × 10³⁰ kg
  - Earth mass: ~6 × 10²⁴ kg

## Limitations

- **Numerical precision**: Limited by floating-point arithmetic
- **Close encounters**: May become unstable for very close body approaches
- **Relativistic effects**: Not included (Newtonian mechanics only)
- **Perturbations**: Does not account for non-gravitational forces

## Contributing

- Keep it simple; avoid extra abstractions unless necessary
- Include small, runnable scenarios in `test.py`

## License

No explicit license specified. Contact the project author for usage rights.

## Design Notes

- The implementation is intentionally minimal and 2D-only.
- RK4 with fixed step is used for clarity; small steps improve stability.

## References

- Classical mechanics and celestial dynamics
- Numerical methods for ordinary differential equations
- N-body problem in astrophysics
