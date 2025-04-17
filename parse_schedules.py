#!/usr/bin/env python3
"""
Parse schedule log files and generate averaged statistics per schedule.

Usage:
    python parse_schedules.py /path/to/log/files
"""

import os
import re
import sys
import argparse
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parse schedule log files")
    parser.add_argument(
        "input", help="Path to log file or directory containing log files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed statistics for each log file",
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
    return 24576000  # Default frequency in Hz


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
            "log_file": log_file,
        }

        schedules_data.append(schedule_data)

    return schedules_data


def print_individual_statistics(schedules_data):
    """Print statistics for each schedule in each log file."""
    print("\n===== INDIVIDUAL SCHEDULE STATISTICS =====")

    for i, schedule in enumerate(schedules_data):
        device = schedule["device"]
        application = schedule["application"]
        schedule_uid = schedule["schedule_uid"]
        log_file = os.path.basename(schedule["log_file"])

        # Calculate total time across all tasks and chunks
        total_time_ms = 0
        for task_id, task_data in schedule["tasks"].items():
            total_time_ms += task_data["total_duration_ms"]

        print(f"\nSchedule {i+1}: {schedule_uid} (from {log_file})")
        print(f"Device: {device}, Application: {application}")
        print(f"Total time: {total_time_ms:.2f} ms")

        # Print average time by chunks
        print("Average time by chunks:")
        for chunk_id, chunk_data in sorted(schedule["chunks"].items()):
            avg_duration = chunk_data["avg_duration"]
            task_count = chunk_data["task_count"]
            total_duration = chunk_data["total_duration"]
            print(
                f"  Chunk {chunk_id}: {avg_duration:.2f} ms (avg) / {total_duration:.2f} ms (total) / {task_count} tasks"
            )

        print("-" * 50)


def calculate_aggregated_statistics(all_schedules):
    """Aggregate statistics across all log files, grouped by schedule UID."""
    # Group schedules by their UID
    grouped_schedules = defaultdict(list)
    for schedule in all_schedules:
        grouped_schedules[schedule["schedule_uid"]].append(schedule)

    # Calculate aggregated statistics for each schedule UID
    aggregated_stats = {}

    for schedule_uid, schedules in grouped_schedules.items():
        # Initialize aggregation data structure
        chunk_data = defaultdict(lambda: {"durations": [], "task_counts": []})
        log_files = set()
        devices = set()
        applications = set()

        # Collect data from all instances of this schedule
        for schedule in schedules:
            log_files.add(os.path.basename(schedule["log_file"]))
            devices.add(schedule["device"])
            applications.add(schedule["application"])

            # Collect chunk data
            for chunk_id, chunk_metrics in schedule["chunks"].items():
                chunk_data[chunk_id]["durations"].append(chunk_metrics["avg_duration"])
                chunk_data[chunk_id]["task_counts"].append(chunk_metrics["task_count"])

        # Calculate averages
        avg_by_chunk = {}
        for chunk_id, data in chunk_data.items():
            if data["durations"]:
                avg_duration = sum(data["durations"]) / len(data["durations"])
                avg_task_count = sum(data["task_counts"]) / len(data["task_counts"])
                avg_by_chunk[chunk_id] = {
                    "avg_duration_ms": avg_duration,
                    "avg_task_count": avg_task_count,
                    "sample_count": len(data["durations"]),
                }

        # Store aggregated stats
        aggregated_stats[schedule_uid] = {
            "devices": list(devices),
            "applications": list(applications),
            "log_files": list(log_files),
            "num_samples": len(schedules),
            "chunks": avg_by_chunk,
        }

    return aggregated_stats


def print_aggregated_statistics(aggregated_stats):
    """Print the aggregated statistics across all log files."""
    print("\n===== AGGREGATED STATISTICS BY SCHEDULE =====")

    # Store widest chunk data for each schedule for the summary at the end
    widest_chunks = {}

    for i, (schedule_uid, stats) in enumerate(sorted(aggregated_stats.items())):
        print(f"\nSchedule {i+1}: {schedule_uid}")
        print(
            f"Samples: {stats['num_samples']} (from {len(stats['log_files'])} log files)"
        )
        print(f"Devices: {', '.join(stats['devices'])}")
        print(f"Applications: {', '.join(stats['applications'])}")

        print("\nAverage time by chunks (across all log files):")

        # Find the widest chunk for this schedule
        widest_chunk_id = None
        widest_chunk_duration = 0

        for chunk_id, chunk_stats in sorted(stats["chunks"].items()):
            avg_duration = chunk_stats["avg_duration_ms"]
            avg_task_count = chunk_stats["avg_task_count"]
            sample_count = chunk_stats["sample_count"]

            print(
                f"  Chunk {chunk_id}: {avg_duration:.2f} ms (avg) / {avg_task_count:.1f} tasks (avg) / {sample_count} samples"
            )

            # Track widest chunk
            if avg_duration > widest_chunk_duration:
                widest_chunk_duration = avg_duration
                widest_chunk_id = chunk_id

        # Store widest chunk info for summary
        if widest_chunk_id is not None:
            widest_chunks[schedule_uid] = {
                "chunk_id": widest_chunk_id,
                "duration_ms": widest_chunk_duration,
            }

        print("-" * 50)

    # Print summary of widest chunks
    print("\n===== WIDEST CHUNK SUMMARY =====")
    print("Schedule UID                    : Chunk ID  Duration (ms)")
    print("-" * 60)

    for schedule_uid, chunk_info in sorted(widest_chunks.items()):
        print(
            f"{schedule_uid:30} : Chunk {chunk_info['chunk_id']:2}   {chunk_info['duration_ms']:.2f} ms"
        )


def main():
    """Main function to process all log files."""
    args = parse_arguments()

    # Find all log files in the specified folder or use the specified file
    log_files = find_log_files(args.input)
    if not log_files:
        print(f"No log files found at {args.input}")
        return 1

    # Process each log file
    all_schedules = []

    for log_file in log_files:
        schedules_data = process_log_file(log_file)
        all_schedules.extend(schedules_data)

    if not all_schedules:
        print("No schedule data was found in any of the log files.")
        return 1

    # Print individual statistics if verbose mode is enabled
    if args.verbose:
        print_individual_statistics(all_schedules)

    # Calculate and print aggregated statistics
    aggregated_stats = calculate_aggregated_statistics(all_schedules)
    print_aggregated_statistics(aggregated_stats)

    print(
        f"\nProcessed {len(log_files)} log files with a total of {len(all_schedules)} schedule instances"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
