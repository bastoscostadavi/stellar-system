"""Minimal 2D N-body simulation with plotting and animation.

Simple API:

from stellar_system import stellar_system

s = stellar_system(positions=[(x1,y1),...], velocities=[(vx1,vy1),...], masses=[m1,...])
s.change(total_time, steps)
s.trajectory()  # static plot
s.animate(interval=30)  # animated plot
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection


# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67428e-11


class stellar_system:
    """2D N-body system.

    - positions: list of (x, y) in meters
    - velocities: list of (vx, vy) in m/s
    - masses: list of masses in kg
    """

    def __init__(self,
                 positions: Sequence[Tuple[float, float]],
                 velocities: Sequence[Tuple[float, float]],
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
        # Optional fixed axis limits: (xmin, xmax, ymin, ymax)
        self.fixed_limits: Optional[Tuple[float, float, float, float]] = None

    # Backwards-compatible alias with old code style
    def change(self, total_time: float, steps: int = 1000):
        """Integrate for `total_time` using `steps` RK4 steps."""
        dt = float(total_time) / int(steps)
        N = self.N
        r = self.r.copy()
        v = self.v.copy()
        hist = np.zeros((steps + 1, N, 2), dtype=float)
        hist[0] = r

        for k in range(steps):
            # RK4 for second-order system r'' = a(r)
            a1 = _accelerations_2d(r, self.m)
            K1r = v
            K1v = a1

            a2 = _accelerations_2d(r + 0.5 * dt * K1r, self.m)
            K2r = v + 0.5 * dt * K1v
            K2v = a2

            a3 = _accelerations_2d(r + 0.5 * dt * K2r, self.m)
            K3r = v + 0.5 * dt * K2v
            K3v = a3

            a4 = _accelerations_2d(r + dt * K3r, self.m)
            K4r = v + dt * K3v
            K4v = a4

            r = r + (dt / 6.0) * (K1r + 2 * K2r + 2 * K3r + K4r)
            v = v + (dt / 6.0) * (K1v + 2 * K2v + 2 * K3v + K4v)

            hist[k + 1] = r

        self.r, self.v = r, v
        self.history = hist
        self.time += total_time
        self._dt = dt
        return self

    def trajectory(self,
                   save_path: Optional[str] = None,
                   dpi: Optional[int] = None,
                   figsize: Optional[Tuple[float, float]] = None,
                   max_frame: Optional[int] = None):
        if self.history is None:
            raise RuntimeError("No history to plot. Call change() first.")
        pos = self.history
        assert pos is not None
        T, N, _ = pos.shape
        # Determine how many frames to include (inclusive frame index)
        if max_frame is None:
            max_frame = T - 1
        max_frame = int(max(1, min(max_frame, T - 1)))

        # Prepare figure with black background
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()

        # Fixed bounds: prefer explicit limits if provided; otherwise initial positions
        if self.fixed_limits is not None:
            xmin, xmax, ymin, ymax = self.fixed_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            xs0 = pos[0, :, 0]
            ys0 = pos[0, :, 1]
            pad_x = 0.05 * (xs0.max() - xs0.min() + 1e-9)
            pad_y = 0.05 * (ys0.max() - ys0.min() + 1e-9)
            ax.set_xlim(xs0.min() - pad_x, xs0.max() + pad_x)
            ax.set_ylim(ys0.min() - pad_y, ys0.max() + pad_y)

        # Compute speeds per segment for color mapping (robust, higher contrast)
        dt = self._dt if self._dt is not None else 1.0
        speeds = np.linalg.norm(pos[1:] - pos[:-1], axis=2) / dt  # (T-1, N)
        speeds_ref = speeds[:max_frame] if speeds.size else speeds
        g0 = 0.4  # base gray
        gamma = 0.5  # brighten mid/low speeds
        if speeds_ref.size:
            vlo = float(np.percentile(speeds_ref, 5))
            vhi = float(np.percentile(speeds_ref, 95))
            if vhi <= vlo:
                vhi = vlo + 1e-12
        else:
            vlo, vhi = 0.0, 1.0

        # Size/line width proportional to log masses (normalized)
        m = self.m
        logm = np.log(m)
        denom = (logm.max() - logm.min()) if (logm.max() - logm.min()) != 0 else 1.0
        norm = (logm - logm.min()) / denom
        # Choose a visual radius range in points
        r_min, r_max = 1.5, 4.0  # marker radius in points (smaller)
        radii = r_min + (r_max - r_min) * norm
        marker_sizes = 2.0 * radii  # markersize is approx diameter in points
        lwidths = radii  # line thickness ~ body radius

        # Draw colored trajectories using LineCollection
        for i in range(N):
            pts = pos[:max_frame + 1, i, :]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (max_frame, 2, 2)
            sp = speeds[:max_frame, i] if speeds.size else np.array([0.0])
            x = (sp - vlo) / (vhi - vlo)
            x = np.clip(x, 0.0, 1.0)
            x = x ** gamma
            g = g0 + (1.0 - g0) * x
            colors = np.stack([g, g, g, np.ones_like(g)], axis=1)
            lc = LineCollection(segs, colors=colors, linewidths=float(lwidths[i]))
            ax.add_collection(lc)
            # Draw final positions as white points
            ax.plot(pts[-1, 0], pts[-1, 1], 'o', color='white', markersize=marker_sizes[i])

        if save_path:
            fig.savefig(save_path, dpi=(dpi or 300), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

    def animate(self, interval: int = 30):
        if self.history is None:
            raise RuntimeError("No history to animate. Call change() first.")
        N = self.N
        pos = self.history  # (T, N, 2)
        assert pos is not None
        T = pos.shape[0]

        # Fixed bounds based on initial positions only (or explicit limits)
        xs0 = pos[0, :, 0]
        ys0 = pos[0, :, 1]
        pad_x = 0.05 * (xs0.max() - xs0.min() + 1e-9)
        pad_y = 0.05 * (ys0.max() - ys0.min() + 1e-9)

        # Figure with black background
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_aspect('equal', adjustable='box')
        if self.fixed_limits is not None:
            xmin, xmax, ymin, ymax = self.fixed_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(xs0.min() - pad_x, xs0.max() + pad_x)
            ax.set_ylim(ys0.min() - pad_y, ys0.max() + pad_y)
        ax.set_axis_off()

        # Compute speeds for color mapping (robust, higher contrast)
        dt = self._dt if self._dt is not None else 1.0
        speeds = np.linalg.norm(pos[1:] - pos[:-1], axis=2) / dt  # (T-1, N)
        g0 = 0.4
        gamma = 0.5
        if speeds.size:
            vlo = float(np.percentile(speeds, 5))
            vhi = float(np.percentile(speeds, 95))
            if vhi <= vlo:
                vhi = vlo + 1e-12
        else:
            vlo, vhi = 0.0, 1.0

        # Sizes from log-mass (and corresponding line width)
        m = self.m
        logm = np.log(m)
        denom = (logm.max() - logm.min()) if (logm.max() - logm.min()) != 0 else 1.0
        norm = (logm - logm.min()) / denom
        r_min, r_max = 1.5, 4.0  # smaller radii
        radii = r_min + (r_max - r_min) * norm
        marker_sizes = 2.0 * radii
        lwidths = radii

        # Create a LineCollection per body for the FULL trajectory (static), and a white point
        collections: List[LineCollection] = []
        points = []
        for i in range(N):
            # Full set of segments for this body
            pts = pos[:, i, :]
            segs = np.stack([pts[:-1], pts[1:]], axis=1) if T > 1 else np.empty((0, 2, 2))
            sp = speeds[:, i] if speeds.size else np.array([0.0])
            if sp.size:
                x = (sp - vlo) / (vhi - vlo)
                x = np.clip(x, 0.0, 1.0)
                x = x ** gamma
                g = g0 + (1.0 - g0) * x
            else:
                g = np.full(0, g0)
            cols = np.stack([g, g, g, np.ones_like(g)], axis=1) if sp.size else np.empty((0, 4))
            lc = LineCollection(segs, colors=cols, linewidths=float(lwidths[i]))
            ax.add_collection(lc)
            collections.append(lc)
            (pt,) = ax.plot([], [], 'o', color='white', markersize=marker_sizes[i])
            points.append(pt)

        def init():
            # Lines are static and already added; set initial point locations
            for i, pt in enumerate(points):
                pt.set_data(pos[0, i, 0], pos[0, i, 1])
            return collections + points

        def update(frame):
            for i in range(N):
                points[i].set_data(pos[frame, i, 0], pos[frame, i, 1])
            return points

        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=T, interval=interval, blit=False)
        plt.show()
        return anim

    def animate_build(self, interval: int = 30):
        """Animate the progressive drawing of the final painting.

        Trails grow over time; grayscale per segment reflects speed.
        Bodies are white points with size proportional to log(mass).
        """
        if self.history is None:
            raise RuntimeError("No history to animate. Call change() first.")
        N = self.N
        pos = self.history  # (T, N, 2)
        assert pos is not None
        T = pos.shape[0]

        # Fixed bounds based on initial positions only (or explicit limits)
        xs0 = pos[0, :, 0]
        ys0 = pos[0, :, 1]
        pad_x = 0.05 * (xs0.max() - xs0.min() + 1e-9)
        pad_y = 0.05 * (ys0.max() - ys0.min() + 1e-9)

        # Figure with black background
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_aspect('equal', adjustable='box')
        if self.fixed_limits is not None:
            xmin, xmax, ymin, ymax = self.fixed_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_xlim(xs0.min() - pad_x, xs0.max() + pad_x)
            ax.set_ylim(ys0.min() - pad_y, ys0.max() + pad_y)
        ax.set_axis_off()

        # Speed-based grayscale (robust, higher contrast)
        dt = self._dt if self._dt is not None else 1.0
        speeds = np.linalg.norm(pos[1:] - pos[:-1], axis=2) / dt  # (T-1, N)
        g0 = 0.4
        gamma = 0.5
        if speeds.size:
            vlo = float(np.percentile(speeds, 5))
            vhi = float(np.percentile(speeds, 95))
            if vhi <= vlo:
                vhi = vlo + 1e-12
        else:
            vlo, vhi = 0.0, 1.0

        # Sizes/line widths from log-mass
        m = self.m
        logm = np.log(m)
        denom = (logm.max() - logm.min()) if (logm.max() - logm.min()) != 0 else 1.0
        norm = (logm - logm.min()) / denom
        r_min, r_max = 1.5, 4.0  # smaller radii
        radii = r_min + (r_max - r_min) * norm
        marker_sizes = 2.0 * radii
        lwidths = radii

        # Precompute per-body segments and segment colors
        segs_list = []
        cols_list = []
        for i in range(N):
            pts = pos[:, i, :]
            segs = np.stack([pts[:-1], pts[1:]], axis=1) if T > 1 else np.empty((0, 2, 2))
            segs_list.append(segs)
            sp = speeds[:, i] if speeds.size else np.array([0.0])
            if sp.size:
                x = (sp - vlo) / (vhi - vlo)
                x = np.clip(x, 0.0, 1.0)
                x = x ** gamma
                g = g0 + (1.0 - g0) * x
            else:
                g = np.full(0, g0)
            cols = np.stack([g, g, g, np.ones_like(g)], axis=1) if sp.size else np.empty((0, 4))
            cols_list.append(cols)

        # Artists
        collections: List[LineCollection] = []
        points = []
        for i in range(N):
            lc = LineCollection([], colors=[], linewidths=float(lwidths[i]))
            ax.add_collection(lc)
            collections.append(lc)
            (pt,) = ax.plot([], [], 'o', color='white', markersize=marker_sizes[i])
            points.append(pt)

        def init():
            # Start with nothing drawn
            for lc, pt in zip(collections, points):
                lc.set_segments([])
                lc.set_colors([])
                pt.set_data(pos[0, 0], pos[0, 1])
            return collections + points

        def update(frame):
            # frame in [0, T)
            for i in range(N):
                segs = segs_list[i]
                cols = cols_list[i]
                k = min(frame, segs.shape[0])  # number of segments to show
                if k > 0:
                    collections[i].set_segments(segs[:k])
                    collections[i].set_colors(cols[:k])
                else:
                    collections[i].set_segments([])
                    collections[i].set_colors([])
                pidx = min(frame, T - 1)
                points[i].set_data(pos[pidx, i, 0], pos[pidx, i, 1])
            return collections + points

        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=T, interval=interval, blit=False)
        plt.show()
        return anim


def _accelerations_2d(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Compute pairwise gravitational accelerations in 2D.

    r: (N, 2), m: (N,) -> a: (N, 2)
    """
    N = r.shape[0]
    a = np.zeros_like(r)
    for i in range(N):
        ai = np.array([0.0, 0.0])
        for j in range(N):
            if i == j:
                continue
            d = r[j] - r[i]
            dist = float(np.hypot(d[0], d[1]))
            if dist == 0.0:
                continue
            inv_dist3 = 1.0 / (dist**3)
            ai += G * m[j] * d * inv_dist3
        a[i] = ai
    return a
