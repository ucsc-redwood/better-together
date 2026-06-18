#!/usr/bin/env python3
"""
Main script for schedule parsing and analysis.

This script coordinates the parsing of log files, calculation of statistics,
model comparison, and visualization generation using modular components.
"""

import sys
import os
import argparse
from typing import Tuple

# Put the optimizer package root on sys.path (direct-run; pytest uses pyproject pythonpath)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import our modular components
from analysis.results.log_parser import find_log_files, process_log_file
from analysis.results.statistics import (
    print_individual_statistics,
    calculate_aggregated_statistics,
    extract_widest_chunks,
    print_aggregated_statistics,
    print_widest_chunk_summary,
)
from analysis.results.model_comparison import (
    load_model_predictions,
    print_comparison_results,
    perform_statistical_analysis,
)
from analysis.results.visualization import create_comparison_visualization


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse schedule log files")

    parser.add_argument(
        "input",
        help="Path to log file or directory containing log files",
        type=str,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed statistics for each log file",
    )
    parser.add_argument(
        "--schedule-file",
        help="Path to the schedule JSON file containing predictions. If not specified, no model comparison will be performed.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for visualization files (optional, no figures will be generated if not specified)",
    )
    parser.add_argument(
        "--time-window",
        "-t",
        help="Time window for analysis (format: start-end, values between 0.0 and 1.0)",
        default="0.0-1.0",
    )
    parser.add_argument(
        "--max-schedules",
        "-n",
        type=int,
        help="Maximum number of schedules to analyze (from 0 to N-1). If not specified, all schedules will be analyzed.",
    )
    return parser.parse_args()


def parse_time_window(time_window_str: str) -> Tuple[float, float]:
    """Parse the time window argument and return a tuple of (start, end)."""
    try:
        time_window_parts = time_window_str.split("-")
        if len(time_window_parts) != 2:
            raise ValueError("Time window must be in format 'start-end'")

        start = float(time_window_parts[0])
        end = float(time_window_parts[1])

        if start < 0 or start > 1 or end < 0 or end > 1 or start >= end:
            raise ValueError(
                "Time window values must be between 0 and 1, and start must be less than end"
            )

        return (start, end)
    except ValueError as e:
        print(f"Error parsing time window: {e}")
        print("Using default time window (0.0-1.0)")
        return (0.0, 1.0)


def main():
    """Main function to process all log files."""
    args = parse_arguments()

    # Validate max-schedules parameter
    if args.max_schedules is not None and args.max_schedules <= 0:
        print("Error: --max-schedules must be a positive integer")
        return 1

    # Parse the time window argument
    time_window = parse_time_window(args.time_window)
    print(f"Using time window: {time_window[0]:.2f}-{time_window[1]:.2f}")

    # Load model predictions if specified
    model_predictions = {}
    if args.schedule_file:
        model_predictions = load_model_predictions(args.schedule_file)

    # Find all log files in the specified folder or use the specified file
    log_files = find_log_files(args.input)
    if not log_files:
        print(f"No log files found at {args.input}")
        return 1

    # Process each log file
    all_schedules = []

    for log_file in log_files:
        schedules_data = process_log_file(log_file, time_window)
        all_schedules.extend(schedules_data)

    if not all_schedules:
        print("No schedule data was found in any of the log files.")
        return 1

    # Apply max-schedules limit if specified
    if args.max_schedules is not None:
        # Get all unique schedule UIDs and sort them for consistency
        unique_uids = sorted(
            list(set(schedule["schedule_uid"] for schedule in all_schedules))
        )

        print(f"Found {len(unique_uids)} unique schedule UIDs")
        print(f"Limiting analysis to first {args.max_schedules} schedules")

        # Take only the first N UIDs
        limited_uids = unique_uids[: args.max_schedules]

        # Filter all_schedules to only include schedules with these UIDs
        all_schedules = [
            schedule
            for schedule in all_schedules
            if schedule["schedule_uid"] in limited_uids
        ]

        print(
            f"After filtering: analyzing {len(all_schedules)} schedule instances from {len(limited_uids)} unique UIDs"
        )
        print(f"Selected schedule UIDs: {', '.join(limited_uids)}")
    else:
        unique_uids = list(set(schedule["schedule_uid"] for schedule in all_schedules))
        print(f"Analyzing all {len(unique_uids)} unique schedule UIDs")

    # Print individual statistics if verbose mode is enabled
    if args.verbose:
        print_individual_statistics(all_schedules)

    # Calculate and print aggregated statistics
    aggregated_stats, raw_data_by_uid = calculate_aggregated_statistics(all_schedules)
    print_aggregated_statistics(aggregated_stats)

    # Extract widest chunks for comparison with model
    widest_chunks = extract_widest_chunks(aggregated_stats)

    # Print widest chunk summary
    print_widest_chunk_summary(widest_chunks, time_window)

    # Compare with model predictions if available
    if model_predictions:
        # Print comparison table
        print_comparison_results(widest_chunks, model_predictions)

        # Perform statistical analysis
        perform_statistical_analysis(widest_chunks, model_predictions)

        # Create visualization only if output directory is specified
        if args.output:
            create_comparison_visualization(
                widest_chunks, model_predictions, args.output, raw_data_by_uid
            )
        else:
            print(
                "\nSkipping visualization generation because no output directory was specified."
            )
            print("Use --output/-o to specify an output directory for visualizations.")

    print(
        f"\nProcessed {len(log_files)} log files with a total of {len(all_schedules)} schedule instances"
    )
    print(f"Time window used for analysis: {time_window[0]:.2f}-{time_window[1]:.2f}")

    # Show schedule limit info if applied
    if args.max_schedules is not None:
        total_unique = len(set(schedule["schedule_uid"] for schedule in all_schedules))
        print(
            f"Schedule limit: analyzed first {args.max_schedules} of {len(unique_uids)} unique schedule UIDs"
        )
        print(f"Unique schedule UIDs analyzed: {total_unique}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
