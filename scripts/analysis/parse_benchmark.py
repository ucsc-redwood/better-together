#!/usr/bin/env python3

import re
import json
import os
import glob
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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


def perform_statistical_analysis(results: List[ScheduleResult], output_dir: str = None):
    """
    Perform statistical analysis on the benchmark results.

    Args:
        results: List of ScheduleResult objects
        output_dir: Directory to save plots to (optional)
    """
    if not results:
        print("No results to analyze")
        return

    # Extract relevant data for analysis
    estimated_times = np.array([r.max_chunk_time for r in results])
    measured_times = np.array([r.avg_time_per_task for r in results])
    diff_percentages = np.array([r.difference_percentage for r in results])

    # Filter out None values for load balance analysis
    valid_load_balance = [
        (r.load_balance_ratio, r.difference_percentage)
        for r in results
        if r.load_balance_ratio is not None
    ]

    if valid_load_balance:
        load_balance_ratios, lb_diff_percentages = zip(*valid_load_balance)
        load_balance_ratios = np.array(load_balance_ratios)
        lb_diff_percentages = np.array(lb_diff_percentages)

    # Extract speedup data
    valid_real_cpu_speedups = [
        r.real_cpu_speedup for r in results if r.real_cpu_speedup is not None
    ]
    valid_real_gpu_speedups = [
        r.real_gpu_speedup for r in results if r.real_gpu_speedup is not None
    ]
    valid_expected_cpu_speedups = [
        r.cpu_speedup
        for r in results
        if r.cpu_speedup is not None and r.real_cpu_speedup is not None
    ]
    valid_expected_gpu_speedups = [
        r.gpu_speedup
        for r in results
        if r.gpu_speedup is not None and r.real_gpu_speedup is not None
    ]

    # 1. Basic statistics
    print("\nSTATISTICAL ANALYSIS")
    print("=" * 50)

    # Prediction error analysis
    mean_abs_error = np.mean(np.abs(estimated_times - measured_times))
    mean_rel_error = np.mean(np.abs(diff_percentages))

    print(f"Mean Absolute Error: {mean_abs_error:.4f}")
    print(f"Mean Relative Error: {mean_rel_error:.4f}%")

    # 2. Correlation analysis
    correlation, p_value = stats.pearsonr(estimated_times, measured_times)

    print(
        f"\nCorrelation between estimated and measured times: {correlation:.4f} (p-value: {p_value:.4f})"
    )
    print(f"R-squared: {correlation**2:.4f}")

    # 3. Regression analysis
    X = estimated_times.reshape(-1, 1)
    model = LinearRegression().fit(X, measured_times)
    r_squared = model.score(X, measured_times)
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"\nLinear Regression: measured = {slope:.4f} * estimated + {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    # 4. Load balance analysis
    if valid_load_balance:
        lb_correlation, lb_p_value = stats.pearsonr(
            load_balance_ratios, lb_diff_percentages
        )
        print(
            f"\nCorrelation between load balance ratio and prediction error: {lb_correlation:.4f} (p-value: {lb_p_value:.4f})"
        )

        # Analyze if higher load balance ratios lead to better predictions
        high_lb_indices = np.where(
            load_balance_ratios > np.median(load_balance_ratios)
        )[0]
        low_lb_indices = np.where(
            load_balance_ratios <= np.median(load_balance_ratios)
        )[0]

        high_lb_errors = np.abs([lb_diff_percentages[i] for i in high_lb_indices])
        low_lb_errors = np.abs([lb_diff_percentages[i] for i in low_lb_indices])

        print("\nError analysis by load balance:")
        print(
            f"  High load balance (>{np.median(load_balance_ratios):.4f}): Mean error = {np.mean(high_lb_errors):.4f}%"
        )
        print(
            f"  Low load balance (<={np.median(load_balance_ratios):.4f}): Mean error = {np.mean(low_lb_errors):.4f}%"
        )

        t_stat, t_p_value = stats.ttest_ind(
            high_lb_errors, low_lb_errors, equal_var=False
        )
        print(
            f"  T-test p-value: {t_p_value:.4f} {'(significant)' if t_p_value < 0.05 else '(not significant)'}"
        )

    # 5. Speedup analysis
    if valid_real_cpu_speedups and valid_expected_cpu_speedups:
        cpu_speedup_corr, cpu_speedup_p = stats.pearsonr(
            valid_expected_cpu_speedups, valid_real_cpu_speedups
        )
        print(
            f"\nCPU Speedup correlation (expected vs. real): {cpu_speedup_corr:.4f} (p-value: {cpu_speedup_p:.4f})"
        )
        print(f"Mean expected CPU speedup: {np.mean(valid_expected_cpu_speedups):.2f}x")
        print(f"Mean real CPU speedup: {np.mean(valid_real_cpu_speedups):.2f}x")
        print(
            f"CPU speedup prediction error: {(np.mean(valid_real_cpu_speedups) - np.mean(valid_expected_cpu_speedups)) / np.mean(valid_expected_cpu_speedups) * 100:.2f}%"
        )

    if valid_real_gpu_speedups and valid_expected_gpu_speedups:
        gpu_speedup_corr, gpu_speedup_p = stats.pearsonr(
            valid_expected_gpu_speedups, valid_real_gpu_speedups
        )
        print(
            f"\nGPU Speedup correlation (expected vs. real): {gpu_speedup_corr:.4f} (p-value: {gpu_speedup_p:.4f})"
        )
        print(f"Mean expected GPU speedup: {np.mean(valid_expected_gpu_speedups):.2f}x")
        print(f"Mean real GPU speedup: {np.mean(valid_real_gpu_speedups):.2f}x")
        print(
            f"GPU speedup prediction error: {(np.mean(valid_real_gpu_speedups) - np.mean(valid_expected_gpu_speedups)) / np.mean(valid_expected_gpu_speedups) * 100:.2f}%"
        )

    # Generate plots if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Plot estimated vs measured times
        plt.figure(figsize=(10, 6))
        plt.scatter(estimated_times, measured_times, alpha=0.7)
        plt.plot(
            [min(estimated_times), max(estimated_times)],
            [min(estimated_times), max(estimated_times)],
            "r--",
            label="Perfect prediction",
        )
        plt.plot(
            [min(estimated_times), max(estimated_times)],
            [
                intercept + slope * min(estimated_times),
                intercept + slope * max(estimated_times),
            ],
            "g-",
            label="Regression line",
        )
        plt.xlabel("Estimated Time (max_chunk_time)")
        plt.ylabel("Measured Time (avg_time_per_task)")
        plt.title(
            f"Estimated vs Measured Times (r={correlation:.2f}, R²={correlation**2:.2f})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "estimated_vs_measured.png"))

        # Plot load balance ratio vs prediction error
        if valid_load_balance:
            plt.figure(figsize=(10, 6))
            plt.scatter(load_balance_ratios, np.abs(lb_diff_percentages), alpha=0.7)
            plt.xlabel("Load Balance Ratio")
            plt.ylabel("Absolute Prediction Error (%)")
            plt.title(
                f"Load Balance Ratio vs Prediction Error (r={lb_correlation:.2f})"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "load_balance_vs_error.png"))

        # Plot real vs expected speedups
        if valid_real_cpu_speedups and valid_expected_cpu_speedups:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_expected_cpu_speedups, valid_real_cpu_speedups, alpha=0.7)
            plt.plot(
                [min(valid_expected_cpu_speedups), max(valid_expected_cpu_speedups)],
                [min(valid_expected_cpu_speedups), max(valid_expected_cpu_speedups)],
                "r--",
                label="Perfect prediction",
            )
            plt.xlabel("Expected CPU Speedup")
            plt.ylabel("Real CPU Speedup")
            plt.title(f"Expected vs Real CPU Speedup (r={cpu_speedup_corr:.2f})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "cpu_speedup_comparison.png"))

        # Plot real vs expected GPU speedups
        if valid_real_gpu_speedups and valid_expected_gpu_speedups:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_expected_gpu_speedups, valid_real_gpu_speedups, alpha=0.7)
            plt.plot(
                [min(valid_expected_gpu_speedups), max(valid_expected_gpu_speedups)],
                [min(valid_expected_gpu_speedups), max(valid_expected_gpu_speedups)],
                "r--",
                label="Perfect prediction",
            )
            plt.xlabel("Expected GPU Speedup")
            plt.ylabel("Real GPU Speedup")
            plt.title(f"Expected vs Real GPU Speedup (r={gpu_speedup_corr:.2f})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "gpu_speedup_comparison.png"))

        plt.close("all")


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
    parser.add_argument(
        "--output-dir",
        help="Directory to save analysis plots to",
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

    # Perform detailed statistical analysis
    perform_statistical_analysis(all_results, args.output_dir)


if __name__ == "__main__":
    main()
