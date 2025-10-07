#!/usr/bin/env python3
"""Create a static 2D plot of N-body trajectories.

Defaults to an Earth–Sun–Moon setup if no inputs are provided.

Usage examples:
  # Default Earth–Sun–Moon
  python painting.py

  # Custom explicit inputs
  python painting.py -t 1e7 -n 4000 \
      --positions "0,0;1.5e11,0;1.504e11,0" \
      --velocities "0,0;0,29780;0,30780" \
      --masses "2e30,6e24,7.3e22"

  # Random N bodies, equal masses, at rest
  python painting.py --random 6 --mass 6e24 --spread 1e11 -t 1e7 -n 4000

  # Random N plus a heavy central body at the origin
  python painting.py --random 20 --mass 6e24 --spread 1e11 \
      --central-mass 1e28 --animate -n 10000

  # Random N with randomized masses (uniform range)
  python painting.py --random 20 --mass-min 1e24 --mass-max 1e26 --spread 1e11 --animate -n 10000

  # Random N with log-uniform masses over a range
  python painting.py --random 20 --mass-min 1e22 --mass-max 1e28 --log-uniform-mass --spread 1e11 --animate -n 10000

  # Add an equilateral triangle of 3 bodies (additional),
  # velocities pointing to the next vertex so each draws one side
  python painting.py --random 20 --mass 6e24 --spread 1e11 --add-triangle --animate -n 10000
"""

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np

from stellar_system import stellar_system


G = 6.67428e-11  # m^3 kg^-1 s^-2


def _parse_pairs(s: str) -> List[Tuple[float, float]]:
    pairs = []
    for part in s.split(';'):
        part = part.strip()
        if not part:
            continue
        x_str, y_str = part.split(',')
        pairs.append((float(x_str), float(y_str)))
    return pairs


def _parse_floats(s: str) -> List[float]:
    return [float(tok.strip()) for tok in s.split(',') if tok.strip()]


def default_earth_sun_moon():
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
    total_time = 1.0e7
    steps = 4000
    return positions, velocities, masses, total_time, steps


def main() -> None:
    p = argparse.ArgumentParser(description="Static 2D N-body plot with grayscale speed and log-mass thickness")
    p.add_argument("--positions", "-p", type=str, help="Semicolon-separated x,y pairs, e.g. '0,0;1.5e11,0'" )
    p.add_argument("--velocities", "-v", type=str, help="Semicolon-separated vx,vy pairs" )
    p.add_argument("--masses", "-m", type=str, help="Comma-separated masses, e.g. '2e30,6e24'" )
    p.add_argument("--time", "-t", type=float, help="Total simulation time in seconds")
    p.add_argument("--steps", "-n", type=int, help="Number of RK4 steps")
    p.add_argument("--animate", action="store_true", help="Animate the painting process (build-up)")
    p.add_argument("--interval", type=int, default=30, help="Animation frame interval in ms (with --animate)")
    # Random configuration
    p.add_argument("--random", type=int, metavar="N", help="Generate N bodies with equal mass at rest, random positions")
    p.add_argument("--mass", type=float, default=6.0e24, help="Mass for each body when using --random (default: 6e24)")
    p.add_argument("--spread", type=float, default=1.0e11, help="Spatial spread (radius) for random positions (default: 1e11)")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility when using --random")
    p.add_argument("--central-mass", type=float, help="If set, adds one heavy body at the origin with zero velocity")
    p.add_argument("--mass-min", type=float, help="Minimum mass for random bodies (use with --random)")
    p.add_argument("--mass-max", type=float, help="Maximum mass for random bodies (use with --random)")
    p.add_argument("--log-uniform-mass", action="store_true", help="Sample masses log-uniformly between --mass-min and --mass-max")
    # Additional triangle bodies
    p.add_argument("--add-triangle", action="store_true", help="Add 3 bodies in an equilateral triangle")
    p.add_argument("--triangle-radius", type=float, help="Circumradius for triangle (default: spread/2 in random mode) in meters")
    p.add_argument("--triangle-mass", type=float, help="Mass for each triangle body (default: equal mass or mean of masses)")
    # Batch/save options
    p.add_argument("--save-dir", type=str, help="If set, saves images to this directory")
    p.add_argument("--runs", type=int, default=1, help="Number of images to generate (with --save-dir)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved images")
    p.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    p.add_argument("--height", type=float, default=10.0, help="Figure height in inches")
    # Snapshot series (single initial condition, multiple times)
    p.add_argument("--snapshots", type=int, help="If set, save N snapshots at t,2t,...,N*t")
    p.add_argument("--snapshot-interval", type=float, help="Base time interval t for snapshots (seconds). If omitted, uses --time")
    p.add_argument("--steps-per-snapshot", type=int, default=2000, help="RK4 steps per snapshot interval when using --snapshots")
    args = p.parse_args()

    # If save options were provided
    save_dir = getattr(args, 'save_dir', None)
    runs = getattr(args, 'runs', 1)
    dpi = getattr(args, 'dpi', 300)
    width = getattr(args, 'width', 10.0)
    height = getattr(args, 'height', 10.0)

    save = save_dir is not None
    if save:
        os.makedirs(save_dir, exist_ok=True)

    base_seed = args.seed if args.seed is not None else None
    total_runs = max(1, int(runs))
    # Timestamp prefix to avoid overwriting existing images
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if save else None
    for i in range(total_runs):
        # Derive per-run seed if requested
        seed_i = None if base_seed is None else base_seed + i

        if args.random:
            N = int(args.random)
            rng = np.random.default_rng(seed_i)
            r = args.spread * np.sqrt(rng.random(N))
            theta = 2 * np.pi * rng.random(N)
            positions = [(float(r[k] * np.cos(theta[k])), float(r[k] * np.sin(theta[k]))) for k in range(N)]
            velocities = [(0.0, 0.0) for _ in range(N)]
            # Masses: either equal or randomized
            if args.mass_min is not None and args.mass_max is not None:
                m_lo = float(args.mass_min)
                m_hi = float(args.mass_max)
                if not (m_lo > 0 and m_hi > 0 and m_hi >= m_lo):
                    raise SystemExit("--mass-min and --mass-max must be positive, with max >= min")
                if args.log_uniform_mass:
                    masses = np.exp(rng.uniform(np.log(m_lo), np.log(m_hi), size=N)).tolist()
                else:
                    masses = rng.uniform(m_lo, m_hi, size=N).tolist()
            else:
                masses = [float(args.mass) for _ in range(N)]
            if args.central_mass is not None and args.central_mass > 0:
                positions.append((0.0, 0.0))
                velocities.append((0.0, 0.0))
                masses.append(float(args.central_mass))
            if args.time:
                total_time = float(args.time)
            else:
                Mtot = float(np.sum(masses))
                R = float(args.spread)
                total_time = float(np.sqrt(R**3 / (G * Mtot)))
            steps = int(args.steps) if args.steps else 2000
        elif args.positions or args.velocities or args.masses or args.time or args.steps:
            if not (args.positions and args.velocities and args.masses and args.time):
                raise SystemExit("When providing custom inputs, you must pass --positions, --velocities, --masses, and --time")
            positions = _parse_pairs(args.positions)
            velocities = _parse_pairs(args.velocities)
            masses = _parse_floats(args.masses)
            total_time = float(args.time)
            steps = int(args.steps) if args.steps else 2000
        else:
            positions, velocities, masses, total_time, steps = default_earth_sun_moon()

        # Optionally add triangle bodies
        if args.add_triangle:
            if args.triangle_radius is not None:
                Rtri = float(args.triangle_radius)
            elif args.random:
                Rtri = float(args.spread) * 0.5
            else:
                xs0 = [p[0] for p in positions]
                ys0 = [p[1] for p in positions]
                Rtri = 0.5 * float(max(max(map(abs, xs0)), max(map(abs, ys0)), 1.0))
            angles = np.deg2rad([0.0, 120.0, 240.0])
            tri_pos = [(float(Rtri * np.cos(a)), float(Rtri * np.sin(a))) for a in angles]
            side = float(np.sqrt(3.0) * Rtri)
            speed = side / float(total_time) if total_time > 0 else 0.0
            tri_vel = []
            for k in range(3):
                p0 = np.array(tri_pos[k])
                p1 = np.array(tri_pos[(k + 1) % 3])
                d = p1 - p0
                norm = float(np.hypot(d[0], d[1]))
                u = d / norm if norm > 0 else np.array([1.0, 0.0])
                v = (speed * u).tolist()
                tri_vel.append((float(v[0]), float(v[1])))
            if args.triangle_mass is not None:
                mtri = [float(args.triangle_mass)] * 3
            elif args.mass_min is not None and args.mass_max is not None:
                mtri_val = float(np.sqrt(float(args.mass_min) * float(args.mass_max)))
                mtri = [mtri_val] * 3
            elif args.mass is not None:
                mtri = [float(args.mass)] * 3
            else:
                mtri_val = float(np.mean(masses)) if len(masses) > 0 else 6.0e24
                mtri = [mtri_val] * 3
            positions.extend(tri_pos)
            velocities.extend(tri_vel)
            masses.extend(mtri)

        s = stellar_system(positions, velocities, masses)
        # Fix plot limits to 2x spread for random mode
        if args.random:
            R = float(args.spread)
            s.fixed_limits = (-2.0 * R, 2.0 * R, -2.0 * R, 2.0 * R)
        # Snapshot mode: override total_time/steps and save multiple frames from one run
        if args.snapshots:
            Nsnap = int(args.snapshots)
            interval = args.snapshot_interval if args.snapshot_interval is not None else args.time
            if interval is None:
                raise SystemExit("--snapshots requires --snapshot-interval or --time to specify base t")
            interval = float(interval)
            steps_per = int(args.steps_per_snapshot)
            total_time = interval * Nsnap
            steps = steps_per * Nsnap
            s.change(total_time, steps)

            # Save each snapshot at k * interval
            for k in range(1, Nsnap + 1):
                max_frame = k * steps_per
                if save:
                    prefix = f"painting_{ts}_" if ts else "painting_"
                    fname = os.path.join(save_dir, f"{prefix}{i+1:02d}_k{k:02d}.png")
                    s.trajectory(save_path=fname, dpi=dpi, figsize=(width, height), max_frame=max_frame)
                else:
                    s.trajectory(max_frame=max_frame)
        else:
            s.change(total_time, steps)
            if save:
                prefix = f"painting_{ts}_" if ts else "painting_"
                fname = os.path.join(save_dir, f"{prefix}{i+1:02d}.png")
                s.trajectory(save_path=fname, dpi=dpi, figsize=(width, height))
            else:
                if args.animate:
                    s.animate_build(interval=args.interval)
                else:
                    s.trajectory()


if __name__ == "__main__":
    main()
