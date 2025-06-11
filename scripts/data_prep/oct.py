#!/usr/bin/env python3
"""
load_scan_points.py

Loads a scan_i_points.dat file containing lines of:
    sensorPosX sensorPosY sensorPosZ  pointX pointY pointZ

and splits it into:
  - sensor_positions: an (N,3) array of sensor poses
  - points:           an (N,3) array of measured 3D points
"""

import numpy as np
import argparse
import os
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
    parser = argparse.ArgumentParser(
        description="Load and analyze scan_i_points.dat files."
    )
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
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed statistics"
    )
    parser.add_argument("--save", action="store_true", help="Save data as .npy files")
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
                print(
                    f"\n=== {scan_file} ({file_size_mb:.1f}MB, {len(sensors)} measurements) ==="
                )
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


if __name__ == "__main__":
    main()
