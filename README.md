# Stellar System Dynamics Simulation

A Python-based N-body gravitational simulation that models the dynamics of celestial bodies in stellar systems using numerical integration methods.

## Overview

This project implements a gravitational N-body problem solver using the 4th-order Runge-Kutta method for accurate numerical integration. It can simulate the orbital mechanics of multiple celestial bodies (planets, moons, stars) under their mutual gravitational influences, following Newton's law of universal gravitation.

## Features

- **N-body gravitational simulation**: Supports any number of celestial bodies
- **4th-order Runge-Kutta integration**: High-precision numerical solver with O(dt⁴) accuracy
- **Orbital trajectory visualization**: Plots the paths of celestial bodies over time
- **Flexible initial conditions**: Easy setup of different stellar system configurations
- **Pre-configured scenarios**: Sun-Earth-Moon system and other planetary configurations

## Project Structure

```
stellar_system/
├── README.md              # This file
├── stellar_system.py      # Main simulation class
├── numerical_methods.py   # Numerical integration and physics engine
└── test.py               # Example scenarios and test cases
```

## Core Components

### `stellar_system.py`
The main simulation class that manages:
- System state (positions, velocities, masses)
- Time evolution through numerical integration
- Trajectory plotting and visualization

### `numerical_methods.py`
Contains the physics engine with:
- **EDO function**: Ordinary Differential Equation solver
- **Runge-Kutta method**: 4th-order numerical integration scheme
- **Gravitational model**: Implementation of Newton's law of gravitation

### `test.py`
Provides example scenarios:
- `three()`: Sun-Earth-Moon system
- `four()`: Sun-Earth-Mars-Moon system
- `five()`: Extended system with Mercury
- `four_plus_intruder()`: System with an additional massive body

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

### Basic Example
```python
import numpy as np
from stellar_system import stellar_system

# Define initial conditions [x1,y1,z1,vx1,vy1,vz1,...,xN,yN,zN,vxN,vyN,vzN,m1,m2,...,mN]
# Positions in meters, velocities in m/s, masses in kg
initial_state = [
    # Sun at origin
    0, 0, 0, 0, 0, 0,
    # Earth at 1.5e11 m with orbital velocity
    1.5e11, 0, 0, 0, 29780, 0,
    # Masses: Sun, Earth
    1.989e30, 5.972e24
]

# Create system
system = stellar_system(initial_state)

# Simulate for 1 year (≈31.5M seconds) with 10000 time steps
system.change(3.15e7, 10000)

# Plot trajectories
system.trajectory()
```

### Pre-configured Scenarios
```python
from test import three, four, five

# Run Sun-Earth-Moon simulation
sun_earth_moon = three()

# Run Sun-Earth-Mars-Moon simulation
four()

# Run extended system with Mercury
five()
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

The initial state vector format is:
```
[x₁, y₁, z₁, vₓ₁, vᵧ₁, vᵤ₁, ..., xₙ, yₙ, zₙ, vₓₙ, vᵧₙ, vᵤₙ, m₁, m₂, ..., mₙ]
```

- First 6N elements: positions and velocities (3D coordinates for N bodies)
- Last N elements: masses of the N bodies
- Units: meters, m/s, kg

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

This appears to be an educational/research project. When contributing:
1. Maintain the existing code structure and conventions
2. Add documentation for new features
3. Include test cases for new scenarios
4. Follow the existing Portuguese/English comment style

## License

No explicit license specified. Contact the project author for usage rights.

## References

- Classical mechanics and celestial dynamics
- Numerical methods for ordinary differential equations
- N-body problem in astrophysics