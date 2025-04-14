#!/usr/bin/env python3

import re
import json
import os
import glob
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import argparse


@dataclass
class ScheduleResult:
    device: str
    app_name: str
    schedule_id: str
    max_chunk_time: float
    avg_time_per_task: float
    difference_percentage: float
    is_better: bool  # True if real measure is better than expected
    load_balance_ratio: float = None  # Add load balance ratio
    cpu_baseline_time: float = None
    gpu_baseline_time: float = None
    cpu_speedup: float = None  # Calculated speedup compared to CPU baseline
    gpu_speedup: float = None  # Calculated speedup compared to GPU baseline
    real_cpu_speedup: float = None  # Real speedup compared to CPU baseline
    real_gpu_speedup: float = None  # Real speedup compared to GPU baseline


def parse_benchmark_output(output_text: str) -> List[Tuple[str, str, str, float]]:
    """
    Parse benchmark output and extract device, app name, schedule ID and average time per task.

    Args:
        output_text (str): The benchmark output text

    Returns:
        List[Tuple[str, str, str, float]]: List of tuples containing (device, app_name, schedule_id, avg_time_per_task)
    """
    results = []

    # Skip the header lines
    lines = output_text.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue

        # Extract components from the beginning of the line
        parts = line.split("/")
        if len(parts) < 2:
            continue

        full_id = parts[0]
        id_components = full_id.split("_")

        if len(id_components) < 4:
            print(f"Warning: Could not parse components from line: {line}")
            continue

        device = id_components[0]
        app_name = id_components[1]
        # Assuming 'schedule' is always part of the component
        schedule_id = id_components[3]

        # Extract avg_time_per_task from the end of the line
        avg_time_match = re.search(r"avg_time_per_task=(\d+\.\d+)", line)
        if avg_time_match:
            avg_time = float(avg_time_match.group(1))
            results.append((device, app_name, schedule_id, avg_time))
        else:
            print(f"Warning: Could not find avg_time_per_task in line: {line}")

    return results


def read_schedule_file(schedule_path: str) -> Dict:
    """
    Read a schedule file and return its contents.

    Args:
        schedule_path (str): Path to the schedule file

    Returns:
        Dict: Schedule file contents
    """
    with open(schedule_path, "r") as f:
        return json.load(f)


def find_schedule_file(
    device: str, app_name: str, schedule_id: str, schedule_root: str
) -> str:
    """
    Find the path to a schedule file based on device, app name, and schedule ID.

    Args:
        device (str): Device name (e.g., "jetson")
        app_name (str): Application name (e.g., "Tree")
        schedule_id (str): Schedule ID (e.g., "001")
        schedule_root (str): Root directory for schedule files

    Returns:
        str: Path to the schedule file, or None if not found
    """
    # Construct the expected path pattern
    schedule_path = (
        Path(schedule_root) / device / app_name / f"schedule_{schedule_id}.json"
    )

    # Check if the file exists
    if schedule_path.exists():
        return str(schedule_path)

    # If the exact path doesn't exist, try to find it with a glob pattern
    potential_files = list(
        Path(schedule_root).glob(
            f"**/*{device}*{app_name}*schedule_{schedule_id}*.json"
        )
    )
    if potential_files:
        return str(potential_files[0])

    return None


def compare_benchmark_with_schedule(
    device: str, app_name: str, schedule_id: str, avg_time: float, schedule_root: str
) -> ScheduleResult:
    """
    Compare benchmark result with schedule file data.

    Args:
        device (str): Device name
        app_name (str): Application name
        schedule_id (str): Schedule ID
        avg_time (float): Average time per task from benchmark
        schedule_root (str): Root directory for schedule files

    Returns:
        ScheduleResult: Comparison result, or None if schedule file not found
    """
    # Find the schedule file
    schedule_file_path = find_schedule_file(
        device, app_name, schedule_id, schedule_root
    )

    if not schedule_file_path:
        print(
            f"Warning: Schedule file not found for {device}_{app_name}_schedule_{schedule_id}"
        )
        return None

    # Read the schedule file
    schedule_data = read_schedule_file(schedule_file_path)

    # Extract the max_chunk_time
    max_chunk_time = schedule_data.get("max_chunk_time")
    if max_chunk_time is None:
        print(
            f"Warning: max_chunk_time not found in schedule file {schedule_file_path}"
        )
        return None

    # Extract additional metrics from schedule
    load_balance_ratio = schedule_data.get("load_balance_ratio")
    cpu_baseline_time = schedule_data.get("cpu_baseline_time")
    gpu_baseline_time = schedule_data.get("gpu_baseline_time")
    cpu_speedup = schedule_data.get("cpu_speedup")
    gpu_speedup = schedule_data.get("gpu_speedup")

    # Calculate real speedups based on actual benchmark time
    real_cpu_speedup = cpu_baseline_time / avg_time if cpu_baseline_time else None
    real_gpu_speedup = gpu_baseline_time / avg_time if gpu_baseline_time else None

    # Calculate difference
    # Lower time is better, so if avg_time < max_chunk_time, performance is better
    is_better = avg_time < max_chunk_time

    # Calculate signed percentage difference
    # Positive percentage means faster than expected (better)
    # Negative percentage means slower than expected (worse)
    difference = max_chunk_time - avg_time
    difference_percentage = (difference / max_chunk_time) * 100

    return ScheduleResult(
        device=device,
        app_name=app_name,
        schedule_id=schedule_id,
        max_chunk_time=max_chunk_time,
        avg_time_per_task=avg_time,
        difference_percentage=difference_percentage,
        is_better=is_better,
        load_balance_ratio=load_balance_ratio,
        cpu_baseline_time=cpu_baseline_time,
        gpu_baseline_time=gpu_baseline_time,
        cpu_speedup=cpu_speedup,
        gpu_speedup=gpu_speedup,
        real_cpu_speedup=real_cpu_speedup,
        real_gpu_speedup=real_gpu_speedup,
    )


def process_benchmark_file(
    benchmark_file: str, schedule_root: str
) -> List[ScheduleResult]:
    """
    Process a single benchmark file and return comparison results.

    Args:
        benchmark_file (str): Path to the benchmark file
        schedule_root (str): Root directory for schedule files

    Returns:
        List[ScheduleResult]: List of comparison results
    """
    print(f"\nProcessing file: {benchmark_file}")

    # Read benchmark output from file
    with open(benchmark_file, "r") as f:
        benchmark_output = f.read()

    # Parse benchmark results
    benchmark_results = parse_benchmark_output(benchmark_output)

    if not benchmark_results:
        print(f"Warning: No benchmark results found in {benchmark_file}")
        return []

    print(f"Found {len(benchmark_results)} benchmark results")

    comparison_results = []
    for device, app_name, schedule_id, avg_time in benchmark_results:
        result = compare_benchmark_with_schedule(
            device, app_name, schedule_id, avg_time, schedule_root
        )
        if result:
            comparison_results.append(result)

    if not comparison_results:
        print(f"Warning: No schedule files found for comparison in {benchmark_file}")

    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark output and compare with schedule files"
    )
    parser.add_argument(
        "benchmark_path",
        help="Path to a benchmark file or directory containing benchmark files",
    )
    parser.add_argument(
        "--schedule-root",
        default="data/schedule_files_v2",
        help="Root directory for schedule files",
    )
    parser.add_argument(
        "--sort-by",
        choices=[
            "difference",
            "max_chunk_time",
            "avg_time",
            "load_balance",
            "cpu_speedup",
            "gpu_speedup",
            "real_cpu_speedup",
            "real_gpu_speedup",
        ],
        default="difference",
        help="Sort results by this metric",
    )
    args = parser.parse_args()

    # Check if the path is a directory or a file
    if os.path.isdir(args.benchmark_path):
        # Process all .txt files in the directory
        benchmark_files = glob.glob(os.path.join(args.benchmark_path, "*.txt"))
        if not benchmark_files:
            print(f"No .txt files found in directory: {args.benchmark_path}")
            return
        print(f"Found {len(benchmark_files)} benchmark files to process")
    else:
        # Process a single file
        if not args.benchmark_path.endswith(".txt"):
            print("Warning: Benchmark file should be a .txt file")
        benchmark_files = [args.benchmark_path]

    all_results = []
    for benchmark_file in benchmark_files:
        results = process_benchmark_file(benchmark_file, args.schedule_root)
        all_results.extend(results)

    if not all_results:
        print("No comparison results found across all files")
        return

    # Sort results based on the specified criterion
    if args.sort_by == "difference":
        sorted_results = sorted(
            all_results, key=lambda r: r.difference_percentage, reverse=True
        )
    elif args.sort_by == "max_chunk_time":
        sorted_results = sorted(all_results, key=lambda r: r.max_chunk_time)
    elif args.sort_by == "avg_time":
        sorted_results = sorted(all_results, key=lambda r: r.avg_time_per_task)
    elif args.sort_by == "load_balance":
        sorted_results = sorted(
            all_results,
            key=lambda r: (
                r.load_balance_ratio if r.load_balance_ratio is not None else 0
            ),
            reverse=True,
        )
    elif args.sort_by == "cpu_speedup":
        sorted_results = sorted(
            all_results,
            key=lambda r: r.cpu_speedup if r.cpu_speedup is not None else 0,
            reverse=True,
        )
    elif args.sort_by == "gpu_speedup":
        sorted_results = sorted(
            all_results,
            key=lambda r: r.gpu_speedup if r.gpu_speedup is not None else 0,
            reverse=True,
        )
    elif args.sort_by == "real_cpu_speedup":
        sorted_results = sorted(
            all_results,
            key=lambda r: r.real_cpu_speedup if r.real_cpu_speedup is not None else 0,
            reverse=True,
        )
    elif args.sort_by == "real_gpu_speedup":
        sorted_results = sorted(
            all_results,
            key=lambda r: r.real_gpu_speedup if r.real_gpu_speedup is not None else 0,
            reverse=True,
        )

    # Print the combined results
    print("\nCombined Schedule Comparison Results:")
    print("-" * 180)
    print(
        f"{'Device':<10} {'App':<10} {'Sched ID':<10} {'Max Chunk':<10} {'Avg Task':<10} {'Diff %':<10} "
        f"{'Load Bal':<10} {'CPU Base':<10} {'GPU Base':<10} {'CPU Spd':<12} {'GPU Spd':<12} {'Real CPU Sp':<12} {'Real GPU Sp':<12}"
    )
    print("-" * 180)

    for result in sorted_results:
        # Format the difference percentage with sign (+ for better, - for worse)
        sign = "+" if result.is_better else "-"
        formatted_diff = f"{sign}{abs(result.difference_percentage):.2f}%"

        # Format metrics if available
        load_balance = (
            f"{result.load_balance_ratio:.2f}"
            if result.load_balance_ratio is not None
            else "N/A"
        )
        cpu_baseline = (
            f"{result.cpu_baseline_time:.2f}"
            if result.cpu_baseline_time is not None
            else "N/A"
        )
        gpu_baseline = (
            f"{result.gpu_baseline_time:.2f}"
            if result.gpu_baseline_time is not None
            else "N/A"
        )
        cpu_speedup = (
            f"{result.cpu_speedup:.2f}x" if result.cpu_speedup is not None else "N/A"
        )
        gpu_speedup = (
            f"{result.gpu_speedup:.2f}x" if result.gpu_speedup is not None else "N/A"
        )
        real_cpu_speedup = (
            f"{result.real_cpu_speedup:.2f}x"
            if result.real_cpu_speedup is not None
            else "N/A"
        )
        real_gpu_speedup = (
            f"{result.real_gpu_speedup:.2f}x"
            if result.real_gpu_speedup is not None
            else "N/A"
        )

        print(
            f"{result.device:<10} {result.app_name:<10} {result.schedule_id:<10} "
            f"{result.max_chunk_time:<10.2f} {result.avg_time_per_task:<10.2f} "
            f"{formatted_diff:<10} {load_balance:<10} "
            f"{cpu_baseline:<10} {gpu_baseline:<10} {cpu_speedup:<12} {gpu_speedup:<12} "
            f"{real_cpu_speedup:<12} {real_gpu_speedup:<12}"
        )

    print("-" * 180)

    # Print summary statistics
    better_count = sum(1 for r in all_results if r.is_better)
    total_count = len(all_results)
    if total_count > 0:
        better_percentage = (better_count / total_count) * 100
        print(
            f"\nSummary: {better_count}/{total_count} ({better_percentage:.2f}%) of results are better than expected"
        )

        # Calculate average real CPU/GPU speedups
        valid_cpu_speedups = [
            r.real_cpu_speedup for r in all_results if r.real_cpu_speedup is not None
        ]
        valid_gpu_speedups = [
            r.real_gpu_speedup for r in all_results if r.real_gpu_speedup is not None
        ]

        if valid_cpu_speedups:
            avg_cpu_speedup = sum(valid_cpu_speedups) / len(valid_cpu_speedups)
            print(f"Average real CPU speedup: {avg_cpu_speedup:.2f}x")

        if valid_gpu_speedups:
            avg_gpu_speedup = sum(valid_gpu_speedups) / len(valid_gpu_speedups)
            print(f"Average real GPU speedup: {avg_gpu_speedup:.2f}x")


if __name__ == "__main__":
    main()
