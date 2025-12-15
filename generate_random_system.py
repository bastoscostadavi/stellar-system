#!/usr/bin/env python3
"""Generate random N-body systems and simulate them."""

import numpy as np
import os
from gravitational_system import gravitational_system


def generate_random_system(N: int = 5,
                          position_scale: float = 1.0,
                          seed: int = None):
    """Generate a random N-body system with zero initial velocities.

    Uses dimensionless units with G = 1, total mass = 1, and unit length scale.

    Args:
        N: number of bodies
        mass_ratio: ratio between largest and smallest mass (log-uniform distribution, default: 1000)
        position_scale: characteristic length scale (default: 1.0)
        seed: random seed for reproducibility

    Returns:
        positions: (N, 2) array in length units
        velocities: (N, 2) array (all zeros) in length/time units
        masses: (N,) array normalized so sum(masses) = 1
    """
    if seed is not None:
        np.random.seed(seed)

    # uniform mass distribution (very wide Gaussian), then normalize to total mass = 1
    masses = np.random.uniform(0, 1, N)
    masses = masses / masses.sum()  # Normalize: sum(masses) = 1

    # Random positions (uniform in a disk of radius position_scale)
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = position_scale * np.sqrt(np.random.uniform(0, 1, N))
    positions = np.column_stack([radii * np.cos(angles),
                                 radii * np.sin(angles)])

    # Zero initial velocities (cold start - gravitational collapse)
    velocities = np.zeros((N, 2))

    return positions, velocities, masses


def generate_and_simulate(N: int = 5,
                         simulation_time: float = 20.0,
                         steps: int = 10000,
                         seed: int = None):
    """Generate a random system and simulate it.

    Uses dimensionless units with G = 1, total mass = 1.

    Args:
        N: number of bodies
        simulation_time: time to simulate in dimensionless units (default: 20)
        steps: number of integration steps (default: 10000)
        seed: random seed for reproducibility

    Returns:
        filename: path to saved simulation file in simulations/
    """
    # Generate random system
    positions, velocities, masses = generate_random_system(N, seed=seed)

    # Create and simulate system
    s = gravitational_system(positions.tolist(), velocities.tolist(), masses.tolist())
    s.change(simulation_time, steps=steps)

    # Build filename with N and seed
    if seed is not None:
        filename = f"simulations/system_n{N}_seed{seed}.npz"
    else:
        import time
        timestamp = int(time.time())
        filename = f"simulations/system_n{N}_t{timestamp}.npz"

    # Save simulation data
    np.savez(filename,
             positions=positions,
             velocities=velocities,
             masses=masses,
             history=s.history,
             dt=s._dt,
             N=N,
             seed=seed if seed is not None else -1,
             simulation_time=simulation_time,
             steps=steps)

    print(f"✓ Saved simulation: {filename}")
    return filename


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and simulate random N-body systems (dimensionless units, G=1, M=1)")
    parser.add_argument("-n", "--bodies", type=int, default=5,
                       help="Number of bodies (default: 5)")
    parser.add_argument("-t", "--time", type=float, default=20.0,
                       help="Simulation time in dynamical units (default: 20.0)")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Number of integration steps (default: 10000)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    generate_and_simulate(
        N=args.bodies,
        simulation_time=args.time,
        steps=args.steps,
        seed=args.seed
    )
