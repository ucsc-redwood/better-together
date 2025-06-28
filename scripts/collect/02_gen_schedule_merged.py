#!/usr/bin/env python3
"""
Simplified schedule generation script that uses modular SMT components.

This script orchestrates the schedule optimization process by:
1. Loading and processing CSV data
2. Getting baseline information
3. Running the SMT solver
4. Outputting results
"""

import argparse
import os
import sys

from smt.baselines import get_baseline_for_config
from smt.data_loader import load_csv_and_compute_averages
from smt.solver import solve_optimization_problem
from smt.solution_analyzer import dump_solutions_as_json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve scheduling optimization problem using data from a CSV file."
    )
    parser.add_argument(
        "--csv_folder",
        type=str,
        help="Root folder path containing CSV data in device/app/backend structure",
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--backend", required=True, choices=["vk", "cu"])
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "fully"],
        required=True,
        help="Mode to select CSV file: 'normal' for normal.csv or 'fully' for fully.csv",
    )
    parser.add_argument(
        "-n",
        "--num_solutions",
        type=int,
        help="Number of solutions to find",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Root path to write the JSON output file (optional)",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
        default=False,
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    backend = args.backend
    device = args.device
    app = args.app
    mode = args.mode
    verbose = args.verbose

    # Get baseline data for this configuration
    baseline_data = get_baseline_for_config(device, app, backend)
    if baseline_data:
        if verbose:
            print(f"Baseline data for {device}/{app}/{backend}:")
            print(f"  CPU (OpenMP): {baseline_data['omp']} ms")
            print(f"  GPU ({backend}): {baseline_data[backend]} ms")
            print(f"  Fastest: {baseline_data['fastest']} ms")
    else:
        print(f"No baseline data available for {device}/{app}/{backend}")

    # Input CSV path with mode-based file selection
    csv_filename = f"{mode}.csv"
    csv_path = os.path.join(args.csv_folder, device, app, backend, csv_filename)

    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(
            f"Warning: CSV file {csv_path} does not exist. Skipping this combination."
        )
        sys.exit(0)
    else:
        print(f"Loading data from CSV file: {csv_path}")
        print(f"Mode: {mode}")

    # Output path for schedule JSON
    if args.output_folder:
        # Create output directory structure if needed
        output_dir = os.path.join(args.output_folder, device, app, backend)
        os.makedirs(output_dir, exist_ok=True)

        # Use mode-specific output filename
        if mode == "normal":
            out_path = os.path.join(output_dir, "schedules_normal.json")
        else:  # mode == "fully"
            out_path = os.path.join(output_dir, "schedules.json")
    else:
        print("No output folder specified, skipping")
        sys.exit(0)

    try:
        stage_timings, use_cuda = load_csv_and_compute_averages(csv_path, app, verbose)

        # Store which GPU backend was used
        gpu_backend = "gpu_cuda" if use_cuda else "gpu_vulkan"

        # Solve the optimization problem
        solutions = solve_optimization_problem(stage_timings, args.num_solutions, app)

        # Update the solutions to reflect the correct GPU backend
        for solution in solutions:
            for chunk in solution["chunks"]:
                if chunk["core_type"] == "GPU":
                    chunk["hardware"] = gpu_backend

        # Output the solutions
        dump_solutions_as_json(solutions, baseline_data, "pretty", out_path)

    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
