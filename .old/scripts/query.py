#!/usr/bin/env python3

import os
import sys
import argparse
from typing import List, Tuple
from common import filter_json_files, init_db, query, ParsedRunName


def format_results(results: List[Tuple[ParsedRunName, float, str]]) -> str:
    """
    Format the accumulated results into a readable table.

    Args:
        results: List of (ParsedRunName, real_time, time_unit) tuples

    Returns:
        Formatted string containing the results table
    """
    # Sort results by application, backend, stage, and core_type
    sorted_results = sorted(
        results,
        key=lambda x: (
            x[0].application,
            x[0].backend,
            x[0].stage,
            x[0].core_type or "",
            x[0].num_threads or 0,
        ),
    )

    # Create table header
    output = []
    output.append("\nPerformance Results:")
    output.append("-" * 80)
    output.append(
        f"{'Application':<15} {'Backend':<8} {'Stage':<6} {'Core':<8} {'Threads':<8} {'Time':<12} {'Unit'}"
    )
    output.append("-" * 80)

    # Add data rows
    for run_name, time, unit in sorted_results:
        core_type = run_name.core_type or "N/A"
        threads = str(run_name.num_threads) if run_name.num_threads else "N/A"
        output.append(
            f"{run_name.application:<15} "
            f"{run_name.backend:<8} "
            f"{run_name.stage:<6} "
            f"{core_type:<8} "
            f"{threads:<8} "
            f"{time:<12.2f} "
            f"{unit}"
        )

    output.append("-" * 80)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Filter and analyze JSON benchmark files by device ID"
    )
    parser.add_argument(
        "-i", "--input_dir", help="Input directory containing JSON files", required=True
    )
    parser.add_argument(
        "-d",
        "--device_id",
        help="Device ID to filter files (e.g., jetson, 3A021JEHN02756)",
        required=True,
    )
    # Add query parameters
    parser.add_argument("-a", "--application", help="Filter by application name")
    parser.add_argument("-b", "--backend", help="Filter by backend (OMP, CUDA, VK)")
    parser.add_argument("-s", "--stage", type=int, help="Filter by stage number")
    parser.add_argument(
        "-c", "--core-type", help="Filter by core type (little, medium, big)"
    )
    parser.add_argument("-t", "--threads", type=int, help="Filter by number of threads")

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist")
        sys.exit(1)

    all_results = init_db(args.input_dir, args.device_id)

    # Apply query filters if specified
    filtered_results = query(
        all_results,
        application=args.application,
        backend=args.backend,
        stage=args.stage,
        core_type=args.core_type,
        num_threads=args.threads,
    )

    # Print results in a formatted table
    print(format_results(filtered_results))


if __name__ == "__main__":
    main()
