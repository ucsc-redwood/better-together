#!/usr/bin/env python3
"""
Log file parsing module for schedule analysis.

This module handles the parsing of log files to extract schedule information,
task data, and timing metrics.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional


def find_log_files(input_path: str) -> List[str]:
    """Find all log files matching the pattern in the specified directory or return the input file if it's a file."""
    log_files = []

    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Check if the file matches our pattern
        filename = os.path.basename(input_path)
        if re.match(r"schedule_run_\d+\.log$", filename):
            log_files.append(input_path)
            print(f"Using log file: {input_path}")
        else:
            print(
                f"Warning: File {input_path} doesn't match expected pattern for log files"
            )
            log_files.append(input_path)  # Include it anyway
    elif os.path.isdir(input_path):
        # It's a directory, search for matching files
        pattern = re.compile(r"schedule_run_\d+\.log$")
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


def extract_python_sections(content: str) -> List[str]:
    """Extract all Python sections between '### Python Begin ###' and '### Python End ###'."""
    python_sections = re.findall(
        r"### Python Begin ###(.*?)### Python End ###", content, re.DOTALL
    )
    return python_sections


def extract_schedule_uid(section: str) -> Optional[str]:
    """Extract Schedule_UID from a Python section."""
    uid_match = re.search(r"Schedule_UID=([A-Za-z0-9\-]+)", section)
    if uid_match:
        return uid_match.group(1)
    return None


def extract_frequency(section: str) -> int:
    """Extract frequency information from a Python section."""
    freq_match = re.search(r"Frequency=(\d+) Hz", section)
    if freq_match:
        return int(freq_match.group(1))
    return 24576000  # Default frequency in Hz


def parse_task_data(section: str) -> Dict[int, Dict[int, Dict[str, int]]]:
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


def process_log_file(
    log_file: str, time_window: Tuple[float, float] = (0.0, 1.0)
) -> List[Dict[str, Any]]:
    """Process a single log file and extract all schedule data."""
    print(
        f"Processing {log_file}... (time window: {time_window[0]:.2f}-{time_window[1]:.2f})"
    )
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

        # A malformed/zero Frequency= line would make cycles->ms a division by zero.
        # Skip the section with a warning instead of aborting the whole run.
        if frequency <= 0:
            print(
                f"Warning: non-positive frequency ({frequency} Hz) in section "
                f"{section_idx+1} ({schedule_uid}); skipping."
            )
            continue

        # Calculate cycles to ms conversion factor
        cycles_to_ms = 1e3 / frequency

        # Find the overall schedule time range
        min_start_time = float("inf")
        max_end_time = 0

        for task_id, chunks in tasks.items():
            for chunk_id, chunk_data in chunks.items():
                start_ms = chunk_data["start"] * cycles_to_ms
                end_ms = chunk_data["end"] * cycles_to_ms

                min_start_time = min(min_start_time, start_ms)
                max_end_time = max(max_end_time, end_ms)

        if min_start_time == float("inf"):
            # No tasks found
            continue

        schedule_duration = max_end_time - min_start_time

        # Calculate the absolute time window limits based on percentage
        window_start = min_start_time + (schedule_duration * time_window[0])
        window_end = min_start_time + (schedule_duration * time_window[1])

        print(
            f"  Schedule {schedule_uid} duration: {schedule_duration:.2f} ms, window: {window_start:.2f}-{window_end:.2f} ms"
        )

        # Calculate additional metrics per task and chunk
        task_metrics = {}
        chunk_metrics = {}

        for task_id, chunks in tasks.items():
            task_total_duration = 0
            task_metrics[task_id] = {"chunks": {}}

            for chunk_id, chunk_data in chunks.items():
                start_ms = chunk_data["start"] * cycles_to_ms
                end_ms = chunk_data["end"] * cycles_to_ms
                duration_ms = chunk_data["duration_cycles"] * cycles_to_ms

                # Check if this task is within our time window
                # We include tasks that at least partially overlap with the window
                if end_ms < window_start or start_ms > window_end:
                    continue

                # Update task metrics
                task_metrics[task_id]["chunks"][chunk_id] = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                }
                task_total_duration += duration_ms

                # Update chunk metrics
                if chunk_id not in chunk_metrics:
                    chunk_metrics[chunk_id] = {"total_duration": 0, "task_count": 0}
                chunk_metrics[chunk_id]["total_duration"] += duration_ms
                chunk_metrics[chunk_id]["task_count"] += 1

            if task_metrics[task_id]["chunks"]:
                task_metrics[task_id]["total_duration_ms"] = task_total_duration
            else:
                # Remove tasks with no chunks within the time window
                del task_metrics[task_id]

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
            "chunks": chunk_metrics,
            "num_tasks": len(task_metrics),
            "num_chunks": len(chunk_metrics),
            "log_file": log_file,
            "time_window": time_window,
            "schedule_start_ms": min_start_time,
            "schedule_end_ms": max_end_time,
            "schedule_duration_ms": schedule_duration,
        }

        schedules_data.append(schedule_data)

    return schedules_data
