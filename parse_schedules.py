#!/usr/bin/env python3
"""
Parse schedule log files from a specified folder and extract timing data.

Usage:
    python parse_schedules.py /path/to/log/folder
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict
import csv


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse schedule log files")
    parser.add_argument(
        "input", help="Path to log file or directory containing log files"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output folder for JSON and CSV files",
        default="./results",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Print schedule statistics to console",
    )
    return parser.parse_args()


def find_log_files(input_path):
    """Find all log files matching the pattern in the specified directory or return the input file if it's a file."""
    log_files = []

    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Check if the file matches our pattern
        filename = os.path.basename(input_path)
        if re.match(r"^[^_]+_[^_]+_schedules_\d+\.log$", filename):
            log_files.append(input_path)
            print(f"Using log file: {input_path}")
        else:
            print(
                f"Warning: File {input_path} doesn't match expected pattern for log files"
            )
            log_files.append(input_path)  # Include it anyway
    elif os.path.isdir(input_path):
        # It's a directory, search for matching files
        pattern = re.compile(r"^[^_]+_[^_]+_schedules_\d+\.log$")
        try:
            for filename in os.listdir(input_path):
                if pattern.match(filename):
                    log_files.append(os.path.join(input_path, filename))
        except Exception as e:
            print(f"Error searching directory {input_path}: {e}")
            return []

        print(f"Found {len(log_files)} log files in {input_path}")
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return []

    return log_files


def extract_python_sections(content):
    """Extract all Python sections between '### Python Begin ###' and '### Python End ###'."""
    python_sections = re.findall(
        r"### Python Begin ###(.*?)### Python End ###", content, re.DOTALL
    )
    return python_sections


def extract_schedule_uid(section):
    """Extract Schedule_UID from a Python section."""
    uid_match = re.search(r"Schedule_UID=([A-Za-z0-9\-]+)", section)
    if uid_match:
        return uid_match.group(1)
    return None


def extract_frequency(section):
    """Extract frequency information from a Python section."""
    freq_match = re.search(r"Frequency=(\d+) Hz", section)
    if freq_match:
        return int(freq_match.group(1))
    return 24576000  # Default frequency in Hz (as seen in the original script)


def parse_task_data(section):
    """Parse task data from a Python section."""
    tasks = {}
    pattern = r"Task=(\d+) Chunk=(\d+) Start=(\d+) End=(\d+) Duration=(\d+)"
    task_matches = re.findall(pattern, section)

    for match in task_matches:
        task_id = int(match[0])
        chunk_id = int(match[1])
        start = int(match[2])
        end = int(match[3])
        duration = int(match[4])

        if task_id not in tasks:
            tasks[task_id] = {}

        tasks[task_id][chunk_id] = {
            "start": start,
            "end": end,
            "duration_cycles": duration,
        }

    return tasks


def process_log_file(log_file):
    """Process a single log file and extract all schedule data."""
    print(f"Processing {log_file}...")
    schedules_data = []

    try:
        with open(log_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return []

    # Extract device and application name from filename
    filename = os.path.basename(log_file)
    parts = filename.split("_")
    if len(parts) >= 2:
        device = parts[0]
        application = parts[1]
    else:
        device = "unknown"
        application = "unknown"

    # Extract all Python sections
    python_sections = extract_python_sections(content)
    print(f"Found {len(python_sections)} schedule sections in {log_file}")

    # Process each Python section
    for section_idx, section in enumerate(python_sections):
        # Extract schedule information
        schedule_uid = extract_schedule_uid(section)
        if not schedule_uid:
            print(f"Warning: Could not find Schedule_UID in section {section_idx+1}")
            continue

        frequency = extract_frequency(section)
        tasks = parse_task_data(section)

        # Calculate cycles to ms conversion factor
        cycles_to_ms = 1e3 / frequency

        # Calculate additional metrics per task and chunk
        task_metrics = {}
        chunk_metrics = defaultdict(lambda: {"total_duration": 0, "task_count": 0})

        for task_id, chunks in tasks.items():
            task_total_duration = 0
            task_metrics[task_id] = {"chunks": {}}

            for chunk_id, chunk_data in chunks.items():
                duration_ms = chunk_data["duration_cycles"] * cycles_to_ms

                # Update task metrics
                task_metrics[task_id]["chunks"][chunk_id] = {
                    "start_ms": chunk_data["start"] * cycles_to_ms,
                    "end_ms": chunk_data["end"] * cycles_to_ms,
                    "duration_ms": duration_ms,
                }
                task_total_duration += duration_ms

                # Update chunk metrics
                chunk_metrics[chunk_id]["total_duration"] += duration_ms
                chunk_metrics[chunk_id]["task_count"] += 1

            task_metrics[task_id]["total_duration_ms"] = task_total_duration

        # Calculate average duration per chunk
        for chunk_id, metrics in chunk_metrics.items():
            if metrics["task_count"] > 0:
                metrics["avg_duration"] = (
                    metrics["total_duration"] / metrics["task_count"]
                )
            else:
                metrics["avg_duration"] = 0

        # Create schedule data
        schedule_data = {
            "device": device,
            "application": application,
            "schedule_uid": schedule_uid,
            "frequency_hz": frequency,
            "tasks": task_metrics,
            "chunks": dict(chunk_metrics),  # Convert defaultdict to dict
            "num_tasks": len(task_metrics),
            "num_chunks": len(chunk_metrics),
        }

        schedules_data.append(schedule_data)

    return schedules_data


def print_schedule_statistics(schedules_data):
    """Print statistics for each schedule."""
    print("\n===== SCHEDULE STATISTICS =====")

    for i, schedule in enumerate(schedules_data):
        device = schedule["device"]
        application = schedule["application"]
        schedule_uid = schedule["schedule_uid"]

        # Calculate total time across all tasks and chunks
        total_time_ms = 0
        for task_id, task_data in schedule["tasks"].items():
            total_time_ms += task_data["total_duration_ms"]

        print(f"\nSchedule {i+1}: {schedule_uid}")
        print(f"Device: {device}, Application: {application}")
        print(f"Total time: {total_time_ms:.2f} ms")

        # Print average time by chunks
        print("Average time by chunks:")
        for chunk_id, chunk_data in sorted(schedule["chunks"].items()):
            avg_duration = chunk_data["avg_duration"]
            # task_count = chunk_data["task_count"]
            # total_duration = chunk_data["total_duration"]
            print(f"  Chunk {chunk_id}: {avg_duration:.2f} ms (avg)")
        print("-" * 50)


def main():
    """Main function to process all log files."""
    args = parse_arguments()

    # Find all log files in the specified folder or use the specified file
    log_files = find_log_files(args.input)
    if not log_files:
        print(f"No log files found at {args.input}")
        return 1

    # Create output folder if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process each log file
    all_schedules = []

    for log_file in log_files:
        schedules_data = process_log_file(log_file)
        all_schedules.extend(schedules_data)

    # Print statistics if requested
    if args.stats:
        print_schedule_statistics(all_schedules)
    else:
        # Always print basic summary
        print(
            f"\nProcessed {len(all_schedules)} schedules. Use --stats for detailed statistics."
        )

    print(
        f"Processed {len(log_files)} log files with a total of {len(all_schedules)} schedules"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
