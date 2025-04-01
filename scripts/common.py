#!/usr/bin/env python3

import os
import re
import json
from typing import Tuple, Optional, List
from dataclasses import dataclass
import sys


@dataclass
class ParsedRunName:
    backend: str
    application: str
    stage: int
    core_type: Optional[str]
    num_threads: Optional[int]


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse benchmark filename to extract metadata.

    Args:
        filename: Filename like 'BM_CifarDense_OMP_3A021JEHN02756.json'

    Returns:
        Tuple of (application_name, backend, device_name)

    Raises:
        ValueError: If filename doesn't match expected format
    """
    base = os.path.basename(filename)
    root, _ = os.path.splitext(base)
    parts = root.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    return parts[1], parts[2], parts[3]


def parse_run_name(line: str) -> Optional[ParsedRunName]:
    """
    Parses a single line of the form:
        {Backend}_{Application}/{StageInfo}[/NumThreads]

    Where:
      - Backend ∈ {OMP, CUDA, VK}
      - Application ∈ {CifarDense, CifarSparse, Tree}
      - StageInfo is either 'Baseline' or 'StageN' (N in 1..9),
        optionally with '_little', '_medium', or '_big'.
      - num_threads is an integer in the third segment (if present).
      - Any line containing 'std' is ignored (returns None).

    Returns:
        ParsedRunName if the line is valid, otherwise None.
    """
    # If line contains 'std', ignore it
    if "std" in line:
        return None

    segments = line.split("/")
    if len(segments) < 2:
        return None  # Invalid format

    # The first segment should be "Backend_Application"
    first_segment = segments[0].split("_", 1)
    if len(first_segment) != 2:
        return None
    backend, application = first_segment

    second_segment = segments[1]
    stage = None
    core_type = None

    # Handle Baseline => stage = 0
    if second_segment == "Baseline":
        stage = 0
    else:
        # Should match "Stage{N}" or "Stage{N}_{core_type}"
        # N must be a digit 1..9, and core_type ∈ {little, medium, big}
        match = re.match(r"Stage(\d)(?:_(little|medium|big))?$", second_segment)
        if not match:
            return None
        stage = int(match.group(1))
        if match.group(2):
            core_type = match.group(2)

    num_threads = None
    # If there's a third segment, parse out the leading integer
    if len(segments) > 2:
        third_segment = segments[2]
        m = re.match(r"(\d+)", third_segment)
        if m:
            num_threads = int(m.group(1))

    return ParsedRunName(
        backend=backend,
        application=application,
        stage=stage,
        core_type=core_type,
        num_threads=num_threads,
    )


def filter_json_files(input_dir: str, device_id: str) -> List[str]:
    """
    Filter JSON files in a directory by device ID.

    Args:
        input_dir: Directory containing JSON files
        device_id: Device ID to filter files (e.g., jetson, 3A021JEHN02756)

    Returns:
        List of matching filenames
    """
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    # Filter files that match the device_id pattern
    matching_files = []
    for file in json_files:
        # Split the filename by underscore
        parts = file.split("_")
        if len(parts) >= 4:  # Ensure we have enough parts
            file_device_id = parts[-1].replace(".json", "")
            if file_device_id == device_id:
                matching_files.append(file)

    return matching_files


def parse_benchmark_file(file_path: str) -> List[Tuple[ParsedRunName, float, str]]:
    """
    Parse a single benchmark JSON file and extract performance data.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of tuples containing (ParsedRunName, real_time, time_unit)
    """
    results = []
    with open(file_path, "r") as f:
        data = json.load(f)

    if "benchmarks" not in data:
        print(f"Warning: Invalid JSON file {file_path} (no 'benchmarks' key)")
        return results

    for benchmark in data["benchmarks"]:
        if benchmark.get("aggregate_name") == "mean":
            run_name = parse_run_name(benchmark["run_name"])
            if run_name:
                results.append(
                    (run_name, benchmark["real_time"], benchmark["time_unit"])
                )

    return results


def init_db(input_dir: str, device_id: str) -> List[Tuple[ParsedRunName, float, str]]:
    # Get matching files
    matching_files = filter_json_files(input_dir, device_id)

    if not matching_files:
        print(f"\nNo files found matching device ID '{device_id}'")
        sys.exit(1)

    # Parse all matching files and accumulate results
    all_results = []
    for file in matching_files:
        file_path = os.path.join(input_dir, file)
        results = parse_benchmark_file(file_path)
        all_results.extend(results)

    return all_results


def query(
    db: List[Tuple[ParsedRunName, float, str]],
    application: Optional[str] = None,
    backend: Optional[str] = None,
    stage: Optional[int] = None,
    core_type: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> List[Tuple[ParsedRunName, float, str]]:
    """
    Query the benchmark database with various filters.

    Args:
        db: List of (ParsedRunName, real_time, time_unit) tuples
        application: Filter by application name
        backend: Filter by backend (OMP, CUDA, VK)
        stage: Filter by stage number
        core_type: Filter by core type (little, medium, big)
        num_threads: Filter by number of threads

    Returns:
        Filtered list of results matching all criteria
    """
    filtered_results = db

    if application:
        filtered_results = [
            r for r in filtered_results if r[0].application == application
        ]

    if backend:
        filtered_results = [r for r in filtered_results if r[0].backend == backend]

    if stage is not None:
        filtered_results = [r for r in filtered_results if r[0].stage == stage]

    if core_type:
        filtered_results = [r for r in filtered_results if r[0].core_type == core_type]

    if num_threads is not None:
        filtered_results = [
            r for r in filtered_results if r[0].num_threads == num_threads
        ]

    return filtered_results
