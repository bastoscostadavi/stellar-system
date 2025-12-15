"""Visualization module for gravitational systems.

Creates paired images:
1. Initial state: white dots showing positions (size ∝ log(mass))
2. Gravitational painting: trajectories with intensity ∝ log(velocity)
"""

import numpy as np
from typing import Tuple, Optional
from gravitational_system import gravitational_system


def compute_frame(positions: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute coordinate frame centered at center of mass.

    Returns:
        com: center of mass (2,)
        scale: half-width of the grid = 1.25 * max_distance
    """
    com = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    centered = positions - com
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    scale = 1.25 * max_dist
    return com, scale


def world_to_pixel(positions: np.ndarray, com: np.ndarray, scale: float,
                   resolution: int = 512) -> np.ndarray:
    """Convert world coordinates to pixel coordinates.

    Frame: [-scale, scale] × [-scale, scale] maps to [0, resolution] × [0, resolution]
    Origin (com) is at center of image.
    """
    centered = positions - com
    # Normalize to [-1, 1]
    normalized = centered / scale
    # Map to [0, resolution], with y-axis flipped for image coordinates
    pixels = np.zeros_like(normalized)
    pixels[:, 0] = (normalized[:, 0] + 1.0) * resolution / 2.0
    pixels[:, 1] = (1.0 - normalized[:, 1]) * resolution / 2.0  # flip y
    return pixels


def render_initial_points(positions: np.ndarray, masses: np.ndarray,
                         resolution: int = 512) -> Tuple[np.ndarray, np.ndarray, float]:
    """Render initial positions as white dots (size ∝ log(mass)).

    Returns:
        image: (resolution, resolution) array in [0, 1]
        com: center of mass
        scale: frame half-width
    """
    com, scale = compute_frame(positions, masses)
    image = np.zeros((resolution, resolution), dtype=float)

    # Convert to pixel coordinates
    pixels = world_to_pixel(positions, com, scale, resolution)

    # Draw each body
    N = len(masses)
    # Power law scaling for more diversity: masses² amplifies differences
    line_radii = 1 + 2*N*masses
    radii = 1.3 * line_radii  # dots slightly bigger

    # Create coordinate grids
    y, x = np.ogrid[:resolution, :resolution]

    for i in range(N):
        px, py = pixels[i]
        if not (0 <= px < resolution and 0 <= py < resolution):
            continue  # skip if outside frame

        # Compute distance from this body's center
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        # Add Gaussian spot
        mask = dist <= 3 * radii[i]  # limit to 3σ for efficiency
        image[mask] += np.exp(-0.5 * (dist[mask] / radii[i])**2)

    return image, com, scale


def render_gravitational_painting(system: gravitational_system,
                                  com: np.ndarray, scale: float,
                                  resolution: int = 512,
                                  v_min: float = 0.001,
                                  masses: np.ndarray = None) -> np.ndarray:
    """Render trajectory painting with log-velocity intensity.

    Args:
        system: gravitational_system with history computed
        com: center of mass for coordinate frame
        scale: frame half-width
        resolution: image size
        v_min: minimum velocity for log scale (to avoid log(0)) - use 0.001 for dimensionless units

    Returns:
        image: (resolution, resolution) array in [0, 1]
    """
    if system.history is None:
        raise ValueError("System has no history. Call system.change() first.")

    history = system.history  # shape: (steps+1, N, 2)
    steps, N, _ = history.shape

    # Compute velocities at each timestep
    dt = system._dt
    velocities = np.zeros((steps - 1, N, 2))
    for k in range(steps - 1):
        velocities[k] = (history[k + 1] - history[k]) / dt

    # Compute velocity magnitudes
    v_mag = np.linalg.norm(velocities, axis=2)  # (steps-1, N)

    #line radii (average mass is 1/N, so 2*N*masses is about 2, for an average mass).
    line_radii = 1 + 2*N*masses

    # Render each body to separate image (maximum within trajectory)
    # Then combine bodies additively (sum at crossings)
    body_images = []

    # Process each body separately
    for i in range(N):
        body_image = np.zeros((resolution, resolution), dtype=float)
        trajectory = history[:, i, :]  # (steps, 2)

        # Compute cumulative arc length along trajectory
        segments = np.diff(trajectory, axis=0)  # (steps-1, 2)
        seg_lengths = np.linalg.norm(segments, axis=1)  # (steps-1,)
        cumulative_length = np.concatenate([[0], np.cumsum(seg_lengths)])  # (steps,)
        total_length = cumulative_length[-1]

        if total_length < 1e-10:  # Skip stationary bodies
            continue

        # Create uniform samples in arc length (spatial sampling)
        # Smaller spacing = smoother, more continuous appearance
        sample_spacing = 0.001  # uniform spacing in world units (reduced for continuity)
        n_samples = int(total_length / sample_spacing)
        if n_samples < 2:
            continue

        uniform_arc_lengths = np.linspace(0, total_length, n_samples)

        # Interpolate positions at uniform arc lengths
        sampled_x = np.interp(uniform_arc_lengths, cumulative_length, trajectory[:, 0])
        sampled_y = np.interp(uniform_arc_lengths, cumulative_length, trajectory[:, 1])
        sampled_positions = np.column_stack([sampled_x, sampled_y])

        # Interpolate velocity magnitudes at uniform arc lengths
        # Velocities are defined at segment midpoints in arc length space
        velocity_arc_lengths = (cumulative_length[:-1] + cumulative_length[1:]) / 2
        velocity_mags = v_mag[:, i]  # (steps-1,)
        sampled_v_mag = np.interp(uniform_arc_lengths, velocity_arc_lengths, velocity_mags)

        # Compute intensity for each sample (bounded [0.5, 1.0])
        sampled_intensities = 1 / (1 + np.exp(-sampled_v_mag))

        # No scaling needed with maximum blending (no accumulation)

        # Get radius for this body
        radius = line_radii[i]

        # Convert to pixel coordinates
        pixels = world_to_pixel(sampled_positions, com, scale, resolution)

        # Render each resampled point
        for j in range(n_samples):
            px, py = pixels[j]
            if not (0 <= px < resolution and 0 <= py < resolution):
                continue

            intensity = sampled_intensities[j]

            # Compute only in local bounding box (more efficient)
            px_int, py_int = int(px), int(py)
            r_int = int(np.ceil(radius))

            y_min = max(0, py_int - r_int)
            y_max = min(resolution, py_int + r_int + 1)
            x_min = max(0, px_int - r_int)
            x_max = min(resolution, px_int + r_int + 1)

            # Create local grid
            y_local, x_local = np.ogrid[y_min:y_max, x_min:x_max]

            # Compute distance from this point (only in local region)
            dist = np.sqrt((x_local - px)**2 + (y_local - py)**2)

            # Maximum blending: no accumulation along trajectory
            # Each point shows local velocity-based intensity, no saturation artifacts
            weights = intensity * np.exp(-0.5 * (dist / radius)**2)
            body_image[y_min:y_max, x_min:x_max] = np.maximum(body_image[y_min:y_max, x_min:x_max], weights)

        # Store this body's rendered trajectory
        body_images.append(body_image)

    # Combine all body images: additive at crossings, capped at 1.0
    image = np.zeros((resolution, resolution), dtype=float)
    for body_image in body_images:
        image = np.minimum(1.0, image + body_image)

    # Draw final positions as white dots (if masses provided)
    if masses is not None:
        # Simple continuous mapping: mass → radius (same as initial points)
        radii = 1.2*line_radii

        # Get final positions
        final_positions = history[-1]
        final_pixels = world_to_pixel(final_positions, com, scale, resolution)

        # Create coordinate grids
        y, x = np.ogrid[:resolution, :resolution]

        for i in range(N):
            px, py = final_pixels[i]
            if not (0 <= px < resolution and 0 <= py < resolution):
                continue

            # Draw black outline (larger)
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            outline_mask = dist <= radii[i] * 1.3
            image[outline_mask] = 0  # Black outline

            # Draw white dot on top
            dot_mask = dist <= radii[i]
            image[dot_mask] = 1.0  # White dot

    return image


def save_image_pair(points_image: np.ndarray, painting_image: np.ndarray,
                   filename_prefix: str):
    """Save the paired images as PNG files.

    Args:
        points_image: initial points rendering
        painting_image: trajectory painting
        filename_prefix: e.g., "example" -> "example_points.png", "example_painting.png"
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow is required for saving images. Install with: pip install pillow")

    # Convert to uint8
    points_uint8 = (points_image * 255).astype(np.uint8)
    painting_uint8 = (painting_image * 255).astype(np.uint8)

    # Save as grayscale images
    Image.fromarray(points_uint8, mode='L').save(f"{filename_prefix}_points.png")
    Image.fromarray(painting_uint8, mode='L').save(f"{filename_prefix}_painting.png")
    print(f"Saved: {filename_prefix}_points.png, {filename_prefix}_painting.png")


def create_painting_from_simulation(sim_file: str, resolution: int = 2048):
    """Load a simulation and create paintings.

    Args:
        sim_file: path to simulation NPZ file (e.g., "simulations/system_n5_seed123.npz")
        resolution: image resolution (default: 2048)

    Returns:
        Tuple of (points_filename, painting_filename)
    """
    import os
    from PIL import Image

    # Load simulation data
    data = np.load(sim_file)
    positions = data['positions']
    masses = data['masses']
    history = data['history']
    dt = data['dt']
    N = int(data['N'])
    seed = int(data['seed'])

    # Reconstruct system object
    s = gravitational_system(positions.tolist(), [[0, 0]] * len(masses), masses.tolist())
    s.history = history
    s._dt = dt

    # Build output filename from simulation filename
    sim_basename = os.path.basename(sim_file).replace('.npz', '')
    output_prefix = f"paintings/{sim_basename}"

    # Render images
    points_img, com, scale = render_initial_points(positions, masses, resolution=resolution)
    painting_img = render_gravitational_painting(s, com, scale, resolution=resolution, masses=masses)

    # Save as PNG
    points_uint8 = (points_img * 255).astype(np.uint8)
    painting_uint8 = (painting_img * 255).astype(np.uint8)

    points_file = f"{output_prefix}_points.png"
    painting_file = f"{output_prefix}_painting.png"

    Image.fromarray(points_uint8).save(points_file)
    Image.fromarray(painting_uint8).save(painting_file)

    print(f"✓ Created paintings: {points_file}, {painting_file}")
    return points_file, painting_file


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Create gravitational paintings from simulation files")
    parser.add_argument("simulation", nargs='?', type=str, default=None,
                       help="Path to simulation file (e.g., simulations/system_n5_seed123.npz). If not provided, processes all files in simulations/")
    parser.add_argument("-r", "--resolution", type=int, default=2048,
                       help="Image resolution (default: 2048)")

    args = parser.parse_args()

    if args.simulation:
        # Process single file
        create_painting_from_simulation(args.simulation, resolution=args.resolution)
    else:
        # Process all simulation files
        sim_files = glob.glob("simulations/*.npz")
        if not sim_files:
            print("No simulation files found in simulations/")
            print("Generate some with: python3 generate_random_system.py")
        else:
            print(f"Processing {len(sim_files)} simulation(s)...")
            for sim_file in sim_files:
                create_painting_from_simulation(sim_file, resolution=args.resolution)