#!/usr/bin/env python3
"""
load_scan_points.py

Loads a scan_i_points.dat file containing lines of:
    sensorPosX sensorPosY sensorPosZ  pointX pointY pointZ

and splits it into:
  - sensor_positions: an (N,3) array of sensor poses
  - points:           an (N,3) array of measured 3D points
"""

import argparse
import os

import numpy as np
from tabulate import tabulate


def load_scan_points(path):
    """
    Load a scan_i_points.dat file.

    Returns:
        sensor_positions: np.ndarray of shape (N,3)
        points:           np.ndarray of shape (N,3)
    """
    # Load all six columns into a (N,6) array
    data = np.loadtxt(path, dtype=np.float64)

    if data.ndim == 1:
        # Single line: make it 2D
        data = data[np.newaxis, :]

    # First three columns are sensor position, next three are point
    sensor_positions = data[:, :3]
    points = data[:, 3:]
    return sensor_positions, points


def get_statistics(array):
    """
    Calculate statistics for a numpy array.

    Returns:
        dict: Statistics including min, max, mean, range for each dimension
    """
    stats = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "range": np.max(array, axis=0) - np.min(array, axis=0),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Load and analyze scan_i_points.dat files.")
    parser.add_argument(
        "--base_dir",
        default="resources/octomap/freiburgCampus360_3D",
        help="Base directory containing scan files",
    )
    parser.add_argument(
        "--output_dir",
        default="resources/octomap/data",
        help="Directory to save the .npy files",
    )
    parser.add_argument(
        "--scan_range",
        default="1-77",
        help="Range of scan files to process (e.g., 1-77 or 1,5,10-15)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")
    parser.add_argument("--save", action="store_true", help="Save data as .npy files")
    parser.add_argument(
        "--concat_target",
        type=int,
        default=None,
        help="Target point count for the concatenated points.npy corpus "
        "(scans are consumed in ascending numeric order and truncated once reached). "
        "Default: no truncation -- use every point in --scan_range. The on-disk file "
        "size doesn't determine memory use at run time; see BT_TREE_INPUT_SIZE and the "
        "per-device table in docs/instruction-for-ai/05-profiling.md for how many of "
        "these points a given target actually loads.",
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="Recenter/scale concatenated points into [domain_min, domain_min + domain_range)",
    )
    parser.add_argument(
        "--domain_min",
        type=float,
        default=0.0,
        help="Lower bound of the recentered coordinate domain (matches tree::kMinCoord)",
    )
    parser.add_argument(
        "--domain_range",
        type=float,
        default=1024.0,
        help="Width of the recentered coordinate domain (matches tree::kRange)",
    )
    args = parser.parse_args()

    # Parse scan range
    scan_nums = []
    parts = args.scan_range.split(",")
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            scan_nums.extend(range(start, end + 1))
        else:
            scan_nums.append(int(part))
    # Concatenation order must be fixed/deterministic regardless of how
    # --scan_range was written (e.g. "5,1-3" vs "1-3,5"): ascending scan number.
    scan_nums = sorted(set(scan_nums))

    # Create output directory if it doesn't exist and save is enabled
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving points to .npy files in: {args.output_dir}")

    # Initialize arrays to collect overall statistics
    all_sensors = []
    all_points = []

    # Table headers for summary
    summary_data = []

    # Process each scan file
    for scan_num in scan_nums:
        scan_file = f"scan_{scan_num:03d}_points.dat"
        scan_path = os.path.join(args.base_dir, scan_file)

        if not os.path.isfile(scan_path):
            print(f"Warning: file not found: {scan_path}")
            continue

        try:
            sensors, pts = load_scan_points(scan_path)

            # Get file size in MB
            file_size_mb = os.path.getsize(scan_path) / (1024 * 1024)

            # Calculate statistics
            sensor_stats = get_statistics(sensors)
            point_stats = get_statistics(pts)

            # Save data as .npy files if requested
            if args.save:
                base_name = f"scan_{scan_num:03d}"
                points_file = os.path.join(args.output_dir, f"{base_name}_points.npy")
                np.save(points_file, pts)
                print(f"Saved {points_file}")

            # Add to overall data
            all_sensors.append(sensors)
            all_points.append(pts)

            # Add to summary table
            summary_data.append(
                [
                    scan_file,
                    f"{file_size_mb:.1f}MB",
                    len(sensors),
                    f"({sensor_stats['min'][0]:.1f}, {sensor_stats['min'][1]:.1f}, {sensor_stats['min'][2]:.1f})",
                    f"({sensor_stats['max'][0]:.1f}, {sensor_stats['max'][1]:.1f}, {sensor_stats['max'][2]:.1f})",
                    f"({point_stats['range'][0]:.1f}, {point_stats['range'][1]:.1f}, {point_stats['range'][2]:.1f})",
                ]
            )

            if args.verbose:
                print(f"\n=== {scan_file} ({file_size_mb:.1f}MB, {len(sensors)} measurements) ===")
                print("Sensor positions statistics:")
                print(f"  Min:   {sensor_stats['min']}")
                print(f"  Max:   {sensor_stats['max']}")
                print(f"  Mean:  {sensor_stats['mean']}")
                print(f"  Range: {sensor_stats['range']}")

                print("\nPoints statistics:")
                print(f"  Min:   {point_stats['min']}")
                print(f"  Max:   {point_stats['max']}")
                print(f"  Mean:  {point_stats['mean']}")
                print(f"  Range: {point_stats['range']}")

        except Exception as e:
            print(f"Error processing {scan_path}: {e}")

    # Print summary table
    print("\n=== Summary of All Scan Files ===")
    headers = [
        "File",
        "Size",
        "Points",
        "Sensor Min (x,y,z)",
        "Sensor Max (x,y,z)",
        "Point Range (x,y,z)",
    ]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))

    # Calculate and print overall statistics if any files were processed
    if all_sensors and all_points:
        all_sensors_array = np.concatenate(all_sensors)
        all_points_array = np.concatenate(all_points)

        all_sensor_stats = get_statistics(all_sensors_array)
        all_point_stats = get_statistics(all_points_array)

        print("\n=== Overall Statistics ===")
        print(f"Total points: {len(all_sensors_array)}")
        print("\nAll sensor positions:")
        print(f"  Min:   {all_sensor_stats['min']}")
        print(f"  Max:   {all_sensor_stats['max']}")
        print(f"  Range: {all_sensor_stats['range']}")

        print("\nAll points:")
        print(f"  Min:   {all_point_stats['min']}")
        print(f"  Max:   {all_point_stats['max']}")
        print(f"  Range: {all_point_stats['range']}")

        # Save combined data if requested
        if args.save:
            combined_points_file = os.path.join(args.output_dir, "all_points.npy")
            np.save(combined_points_file, all_points_array)
            print(f"\nSaved combined points data:")
            print(f"  {combined_points_file} ({all_points_array.shape})")

            # points.npy: the tree app's real-data corpus (contracts/tree-real-data-
            # contract.md). Scans were already consumed in ascending numeric order
            # above, so a prefix of this array is deterministic and reproducible.
            # concat_target=None (default) keeps every point -- the on-disk file size
            # doesn't drive memory use at run time, BT_TREE_INPUT_SIZE does.
            corpus = (
                all_points_array
                if args.concat_target is None
                else all_points_array[: args.concat_target]
            )
            if args.recenter:
                lo, hi = float(corpus.min()), float(corpus.max())
                scale = args.domain_range / (hi - lo) if hi > lo else 1.0
                corpus = (corpus - lo) * scale + args.domain_min
            corpus = np.ascontiguousarray(corpus, dtype="<f4")

            points_file = os.path.join(args.output_dir, "points.npy")
            np.save(points_file, corpus)
            print(f"\nSaved real-data corpus:")
            print(
                f"  {points_file} ({corpus.shape[0]} points"
                + (f", target was {args.concat_target})" if args.concat_target is not None else ")")
            )
            if args.concat_target is not None and corpus.shape[0] < args.concat_target:
                print(
                    f"  WARNING: corpus has fewer points ({corpus.shape[0]}) than "
                    f"--concat_target ({args.concat_target}); add more scans via --scan_range"
                )


if __name__ == "__main__":
    main()
