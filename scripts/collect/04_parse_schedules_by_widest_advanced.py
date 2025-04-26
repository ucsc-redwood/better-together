#!/usr/bin/env python3
"""
Parse schedule log files and generate averaged statistics per schedule.

Takes .log files from a folder
Takes Json Schedule file

Produces:
- Aggreated data for each schedule run
- Measured vs Predicted Stats
- Visualization of Measured vs Predicted


Usage:
    python parse_schedules.py /path/to/log/files [--model /path/to/model.json] [--time-window 0.0-1.0]
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument(
        "--model",
        "-m",
        help="Path to JSON file containing model predictions",
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
    return parser.parse_args()


def find_log_files(input_path):
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


def process_log_file(log_file, time_window=(0.0, 1.0)):
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
        chunk_metrics = defaultdict(lambda: {"total_duration": 0, "task_count": 0})

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
            "chunks": dict(chunk_metrics),  # Convert defaultdict to dict
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


def create_comparison_visualization(
    widest_chunks, model_predictions, output_dir, raw_data
):
    """Create visualization comparing measured results with model predictions."""
    if not model_predictions or not widest_chunks:
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Collect data for visualization
    schedule_data = []

    # Matching UIDs that have both measurements and predictions
    matching_uids = sorted(set(widest_chunks.keys()) & set(model_predictions.keys()))

    if not matching_uids:
        print("No matching UIDs found between measurements and predictions")
        return

    # Collect data
    for uid in matching_uids:
        measured = widest_chunks[uid]["duration_ms"]
        predicted = model_predictions[uid]
        std_dev = 0

        if uid in raw_data:
            # Calculate standard deviation for error bars from raw data
            durations = []
            for schedule in raw_data[uid]:
                chunk_id = widest_chunks[uid]["chunk_id"]
                if chunk_id in schedule["chunks"]:
                    durations.append(schedule["chunks"][chunk_id]["avg_duration"])

            if len(durations) > 1:
                std_dev = np.std(durations)

        schedule_data.append((uid, measured, predicted, std_dev))

    # Sort by predicted time (fastest to slowest)
    schedule_data.sort(key=lambda x: x[2])

    # Extract sorted data into separate lists
    schedule_uids = [item[0] for item in schedule_data]
    measured_times = np.array([item[1] for item in schedule_data])
    predicted_times = np.array([item[2] for item in schedule_data])
    error_bars = np.array([item[3] for item in schedule_data])

    # Create bar chart figure (original)
    plt.figure(figsize=(14, 8))

    # Calculate positions for bars
    x = np.arange(len(schedule_uids))
    width = 0.35

    # Plot bars
    measured_bars = plt.bar(
        x - width / 2, measured_times, width, label="Measured", alpha=0.7
    )
    predicted_bars = plt.bar(
        x + width / 2, predicted_times, width, label="Predicted", alpha=0.7
    )

    # Add error bars to measured data
    plt.errorbar(
        x - width / 2,
        measured_times,
        yerr=error_bars,
        fmt="none",
        ecolor="black",
        capsize=5,
    )

    # Add labels and title
    # plt.xlabel("Schedule UID (sorted by predicted time)")
    plt.ylabel("Time (ms)")
    # plt.title(
    #     "Comparison of Measured vs Predicted Execution Times (Sorted by Prediction)"
    # )
    plt.xticks(x, [uid.split("-")[1] for uid in schedule_uids], rotation=45, ha="right")
    plt.legend()

    # Add value labels on the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_labels(measured_bars)
    add_labels(predicted_bars)

    # Add grid and adjust layout
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, "comparison_chart.png"), dpi=300)
    print(f"Visualization saved to {os.path.join(output_dir, 'comparison_chart.png')}")

    # Create line chart visualization with data sorted by predicted time
    create_line_comparison_chart(
        schedule_uids,
        measured_times,
        predicted_times,
        error_bars,
        output_dir,
        "line_comparison_chart.png",
        "Comparison of Measured vs Predicted Execution Times",
        "by predicted time",
    )

    # Create a second line chart with data sorted by measured time
    # Sort by measured time (fastest to slowest)
    schedule_data_by_measured = sorted(schedule_data, key=lambda x: x[1])

    # Extract sorted data
    schedule_uids_by_measured = [item[0] for item in schedule_data_by_measured]
    measured_times_sorted = np.array([item[1] for item in schedule_data_by_measured])
    predicted_times_sorted = np.array([item[2] for item in schedule_data_by_measured])
    error_bars_sorted = np.array([item[3] for item in schedule_data_by_measured])

    # Create line chart with data sorted by measured time
    create_line_comparison_chart(
        schedule_uids_by_measured,
        measured_times_sorted,
        predicted_times_sorted,
        error_bars_sorted,
        output_dir,
        "line_comparison_by_measured.png",
        "Comparison of Measured vs Predicted Execution Times",
        "by measured time",
    )

    # Create scatter plot for correlation
    create_correlation_plots(schedule_uids, predicted_times, measured_times, output_dir)


def create_line_comparison_chart(
    schedule_uids,
    measured_times,
    predicted_times,
    error_bars,
    output_dir,
    filename="line_comparison_chart.png",
    title="Comparison of Measured vs Predicted Execution Times",
    sort_note="",
):
    """Create a line-based visualization comparing measured and predicted execution times."""
    # Create figure with white background
    plt.figure(figsize=(14, 6), facecolor="white")  # Reduced height from 8 to 6

    # Get x positions
    x = np.arange(len(schedule_uids))

    # Plot lines with markers
    plt.plot(
        x,
        predicted_times,
        "r--",
        marker="s",
        markersize=14,  # Increased from 14 to 20
        linewidth=1,  # Reduced from 2 to 1
        label="Predicted",
        alpha=0.9,
    )
    plt.plot(
        x,
        measured_times,
        "b-",
        marker="^",
        markersize=14,  # Increased from 14 to 20
        linewidth=1,  # Reduced from 2 to 1
        label="Measured (Arithmetic)",
        alpha=0.9,
    )

    # Add error bars to measured data - made more prominent
    plt.errorbar(
        x,
        measured_times,
        yerr=error_bars,
        fmt="none",
        ecolor="blue",
        capsize=8,  # Increased from 5 to 8
        alpha=0.9,  # Increased from 0.7 to 0.9
        elinewidth=2,  # Reduced from 3 to 2
    )

    # Add labels and title
    # plt.xlabel("Execution Schedule", fontsize=14, labelpad=10)
    plt.ylabel("Time (Execution in ms)", fontsize=14, labelpad=10)

    # Title is now commented out
    # if sort_note:
    #     plt.title(f"{title}\n(Sorted {sort_note})", fontsize=16, pad=20)
    # else:
    #     plt.title(title, fontsize=16, pad=20)

    # Add UID labels on x-axis
    shortened_uids = [uid.split("-")[1] for uid in schedule_uids]
    plt.xticks(x, shortened_uids, rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=12)

    # Add grid for both axes
    plt.grid(True, linestyle="--", alpha=0.7, which="both")

    # Create legend with larger font and better position
    plt.legend(fontsize=18, loc="upper left", markerscale=1.5)  # Increased font size and marker scale

    # Set y-axis to start at 0
    # Calculate a good maximum y value that leaves room for highest point plus error bar
    max_y = max(max(predicted_times), max(measured_times) + max(error_bars)) * 1.15
    plt.ylim(bottom=0, top=max_y)

    # Make plot lines thicker
    for line in plt.gca().get_lines():
        if line.get_linestyle() == "--":  # Predicted line
            line.set_linewidth(1.5)  # Reduced from 3 to 1.5
        elif line.get_marker() == "^":  # Measured line
            line.set_linewidth(1.5)  # Reduced from 3 to 1.5

    # Add minor tick lines for better readability
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", alpha=0.4)

    # Adjust layout
    plt.tight_layout()

    # Save figure with higher resolution
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Line visualization saved to {os.path.join(output_dir, filename)}")

    plt.close()


def create_correlation_plots(
    schedule_uids, predicted_times, measured_times, output_dir
):
    """Create various correlation plots to better visualize the data."""

    # Main correlation plot (standard)
    plt.figure(figsize=(10, 8))
    plt.scatter(predicted_times, measured_times, alpha=0.7)

    # Add diagonal line (perfect prediction)
    max_val = max(np.max(predicted_times), np.max(measured_times)) * 1.1
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")

    # Add labels
    plt.xlabel("Predicted Time (ms)")
    plt.ylabel("Measured Time (ms)")
    plt.title("Correlation between Predicted and Measured Times")

    # Add schedule labels to points
    for i, uid in enumerate(schedule_uids):
        plt.annotate(
            uid.split("-")[1],
            (predicted_times[i], measured_times[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )

    # Calculate correlation coefficient
    correlation = np.corrcoef(predicted_times, measured_times)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save standard scatter plot
    plt.savefig(os.path.join(output_dir, "correlation_plot.png"), dpi=300)
    print(
        f"Correlation plot saved to {os.path.join(output_dir, 'correlation_plot.png')}"
    )

    # 1. Create log-scale plot for better distribution visualization
    plt.figure(figsize=(10, 8))

    # Skip zeros and negative values for log scale
    valid_indices = (predicted_times > 0) & (measured_times > 0)
    valid_pred = predicted_times[valid_indices]
    valid_meas = measured_times[valid_indices]
    valid_uids = [schedule_uids[i] for i, valid in enumerate(valid_indices) if valid]

    if len(valid_pred) > 0:
        plt.scatter(valid_pred, valid_meas, alpha=0.7)

        # Add perfect prediction line on log scale
        min_val = min(np.min(valid_pred), np.min(valid_meas)) * 0.9
        max_val = max(np.max(valid_pred), np.max(valid_meas)) * 1.1
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Predicted Time (ms) - Log Scale")
        plt.ylabel("Measured Time (ms) - Log Scale")
        plt.title("Log-Scale Correlation between Predicted and Measured Times")

        # Add schedule labels to points
        for i, uid in enumerate(valid_uids):
            plt.annotate(
                uid.split("-")[1],
                (valid_pred[i], valid_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        # Calculate correlation coefficient for valid points
        log_correlation = np.corrcoef(np.log(valid_pred), np.log(valid_meas))[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Log Correlation: {log_correlation:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save log-scale plot
        plt.savefig(os.path.join(output_dir, "correlation_plot_log_scale.png"), dpi=300)
        print(
            f"Log-scale correlation plot saved to {os.path.join(output_dir, 'correlation_plot_log_scale.png')}"
        )

    # 2. Create a plot that excludes outliers
    plt.figure(figsize=(10, 8))

    # Identify outliers using IQR method
    q1_pred = np.percentile(predicted_times, 25)
    q3_pred = np.percentile(predicted_times, 75)
    iqr_pred = q3_pred - q1_pred

    q1_meas = np.percentile(measured_times, 25)
    q3_meas = np.percentile(measured_times, 75)
    iqr_meas = q3_meas - q1_meas

    # Define outlier boundaries
    lower_bound_pred = q1_pred - 1.5 * iqr_pred
    upper_bound_pred = q3_pred + 1.5 * iqr_pred

    lower_bound_meas = q1_meas - 1.5 * iqr_meas
    upper_bound_meas = q3_meas + 1.5 * iqr_meas

    # Identify non-outlier indices
    non_outlier_indices = (
        (predicted_times >= lower_bound_pred)
        & (predicted_times <= upper_bound_pred)
        & (measured_times >= lower_bound_meas)
        & (measured_times <= upper_bound_meas)
    )

    # Consider a point an outlier if it's beyond 3 standard deviations from the mean
    mean_pred = np.mean(predicted_times)
    std_pred = np.std(predicted_times)

    mean_meas = np.mean(measured_times)
    std_meas = np.std(measured_times)

    non_outlier_indices_std = (
        (predicted_times >= mean_pred - 3 * std_pred)
        & (predicted_times <= mean_pred + 3 * std_pred)
        & (measured_times >= mean_meas - 3 * std_meas)
        & (measured_times <= mean_meas + 3 * std_meas)
    )

    # Combine methods - a point is a non-outlier if it passes either test
    non_outlier_indices = non_outlier_indices | non_outlier_indices_std

    # Filter data
    non_outlier_pred = predicted_times[non_outlier_indices]
    non_outlier_meas = measured_times[non_outlier_indices]
    non_outlier_uids = [
        schedule_uids[i] for i, is_valid in enumerate(non_outlier_indices) if is_valid
    ]

    if len(non_outlier_indices) > 0:
        plt.scatter(non_outlier_pred, non_outlier_meas, alpha=0.7)

        # Add perfect prediction line
        min_val = min(np.min(non_outlier_pred), np.min(non_outlier_meas)) * 0.9
        max_val = max(np.max(non_outlier_pred), np.max(non_outlier_meas)) * 1.1
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        plt.xlabel("Predicted Time (ms)")
        plt.ylabel("Measured Time (ms)")
        plt.title("Correlation (Excluding Outliers)")

        # Add schedule labels
        for i, uid in enumerate(non_outlier_uids):
            plt.annotate(
                uid.split("-")[1],
                (non_outlier_pred[i], non_outlier_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        # Calculate correlation coefficient for non-outliers
        if len(non_outlier_pred) > 1:  # Need at least 2 points for correlation
            non_outlier_correlation = np.corrcoef(non_outlier_pred, non_outlier_meas)[
                0, 1
            ]
            plt.text(
                0.05,
                0.95,
                f"Correlation (excl. outliers): {non_outlier_correlation:.4f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save non-outlier plot
        plt.savefig(
            os.path.join(output_dir, "correlation_plot_no_outliers.png"), dpi=300
        )
        print(
            f"Correlation plot (excl. outliers) saved to {os.path.join(output_dir, 'correlation_plot_no_outliers.png')}"
        )

    # 3. Create zoomed-in plot of the cluster
    # Find the median values to center the zoom
    median_pred = np.median(predicted_times)
    median_meas = np.median(measured_times)

    # Define zoom window (2x the IQR)
    zoom_width_pred = 2 * iqr_pred
    zoom_width_meas = 2 * iqr_meas

    # Define zoom boundaries
    zoom_min_pred = max(0, median_pred - zoom_width_pred)
    zoom_max_pred = median_pred + zoom_width_pred

    zoom_min_meas = max(0, median_meas - zoom_width_meas)
    zoom_max_meas = median_meas + zoom_width_meas

    # Create zoomed plot
    plt.figure(figsize=(10, 8))

    # Plot all points but focus on the zoom area
    plt.scatter(predicted_times, measured_times, alpha=0.5, color="lightgray")

    # Highlight points in the zoom window
    zoom_indices = (
        (predicted_times >= zoom_min_pred)
        & (predicted_times <= zoom_max_pred)
        & (measured_times >= zoom_min_meas)
        & (measured_times <= zoom_max_meas)
    )

    zoom_pred = predicted_times[zoom_indices]
    zoom_meas = measured_times[zoom_indices]
    zoom_uids = [
        schedule_uids[i] for i, is_zoomed in enumerate(zoom_indices) if is_zoomed
    ]

    if len(zoom_indices) > 0:
        plt.scatter(zoom_pred, zoom_meas, alpha=0.9)

        # Add perfect prediction line just for the zoom window
        zoom_min = min(zoom_min_pred, zoom_min_meas)
        zoom_max = max(zoom_max_pred, zoom_max_meas)
        plt.plot(
            [zoom_min, zoom_max],
            [zoom_min, zoom_max],
            "r--",
            label="Perfect Prediction",
        )

        # Set limits to zoom window
        plt.xlim(zoom_min_pred, zoom_max_pred)
        plt.ylim(zoom_min_meas, zoom_max_meas)

        plt.xlabel("Predicted Time (ms)")
        plt.ylabel("Measured Time (ms)")
        plt.title("Zoomed Correlation View (Focused on Cluster)")

        # Add labels for points in the zoom window
        for i, uid in enumerate(zoom_uids):
            plt.annotate(
                uid.split("-")[1],
                (zoom_pred[i], zoom_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        # Calculate correlation coefficient for zoomed region
        if len(zoom_pred) > 1:  # Need at least 2 points for correlation
            zoom_correlation = np.corrcoef(zoom_pred, zoom_meas)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Zoom Correlation: {zoom_correlation:.4f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save zoomed plot
        plt.savefig(os.path.join(output_dir, "correlation_plot_zoomed.png"), dpi=300)
        print(
            f"Zoomed correlation plot saved to {os.path.join(output_dir, 'correlation_plot_zoomed.png')}"
        )

    # Close all figures to free memory
    plt.close("all")


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

    return aggregated_stats, grouped_schedules


def print_aggregated_statistics(aggregated_stats):
    """Print the aggregated statistics across all log files."""
    print("\n===== AGGREGATED STATISTICS BY SCHEDULE =====")

    for i, (schedule_uid, stats) in enumerate(sorted(aggregated_stats.items())):
        print(f"\nSchedule {i+1}: {schedule_uid}")
        print(
            f"Samples: {stats['num_samples']} (from {len(stats['log_files'])} log files)"
        )
        print(f"Devices: {', '.join(stats['devices'])}")
        print(f"Applications: {', '.join(stats['applications'])}")

        print("\nAverage time by chunks (across all log files):")
        for chunk_id, chunk_stats in sorted(stats["chunks"].items()):
            avg_duration = chunk_stats["avg_duration_ms"]
            avg_task_count = chunk_stats["avg_task_count"]
            sample_count = chunk_stats["sample_count"]
            print(
                f"  Chunk {chunk_id}: {avg_duration:.2f} ms (avg) / {avg_task_count:.1f} tasks (avg) / {sample_count} samples"
            )

        print("-" * 50)


def perform_statistical_analysis(widest_chunks, model_predictions):
    """Perform detailed statistical analysis on measured vs predicted times."""
    if not model_predictions or not widest_chunks:
        return

    # Extract data for matched UIDs
    matching_uids = sorted(set(widest_chunks.keys()) & set(model_predictions.keys()))

    if not matching_uids:
        print("No matching UIDs found between measurements and predictions")
        return

    # Collect data
    measured_times = []
    predicted_times = []
    abs_differences = []
    rel_differences_pct = []

    for uid in matching_uids:
        measured = widest_chunks[uid]["duration_ms"]
        predicted = model_predictions[uid]

        measured_times.append(measured)
        predicted_times.append(predicted)

        # Calculate differences
        abs_diff = measured - predicted
        abs_differences.append(abs_diff)

        # Calculate relative difference as percentage
        if predicted != 0:
            rel_diff_pct = (abs_diff / predicted) * 100
        else:
            rel_diff_pct = float("inf")
        rel_differences_pct.append(rel_diff_pct)

    # Convert to numpy arrays
    measured_times = np.array(measured_times)
    predicted_times = np.array(predicted_times)
    abs_differences = np.array(abs_differences)
    rel_differences_pct = np.array([d for d in rel_differences_pct if not np.isinf(d)])

    # Calculate basic statistics
    correlation = np.corrcoef(measured_times, predicted_times)[0, 1]
    r_squared = correlation**2

    mse = np.mean(abs_differences**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(abs_differences))

    # Check for under/over prediction bias
    under_predictions = sum(
        measured > predicted
        for measured, predicted in zip(measured_times, predicted_times)
    )
    over_predictions = sum(
        measured < predicted
        for measured, predicted in zip(measured_times, predicted_times)
    )
    exact_matches = sum(
        measured == predicted
        for measured, predicted in zip(measured_times, predicted_times)
    )

    # Count predictions within error margins
    within_5_pct = sum(abs(diff) <= 5 for diff in rel_differences_pct)
    within_10_pct = sum(abs(diff) <= 10 for diff in rel_differences_pct)
    within_20_pct = sum(abs(diff) <= 20 for diff in rel_differences_pct)

    # Print the analysis
    print("\n===== STATISTICAL ANALYSIS =====")
    print(f"Total comparisons: {len(matching_uids)}")

    print("\nCorrelation Statistics:")
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"Coefficient of determination (R²): {r_squared:.4f}")

    print("\nError Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f} ms²")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} ms")
    print(f"Mean Absolute Error (MAE): {mae:.4f} ms")

    print("\nError Distribution:")
    if len(rel_differences_pct) > 0:
        print(f"Mean percentage error: {np.mean(rel_differences_pct):.2f}%")
        print(f"Median percentage error: {np.median(rel_differences_pct):.2f}%")
        print(
            f"Standard deviation of percentage error: {np.std(rel_differences_pct):.2f}%"
        )
        print(f"Min percentage error: {np.min(rel_differences_pct):.2f}%")
        print(f"Max percentage error: {np.max(rel_differences_pct):.2f}%")

    print("\nPrediction Accuracy:")
    print(
        f"Within 5% margin: {within_5_pct} ({within_5_pct/len(rel_differences_pct)*100:.2f}% of valid comparisons)"
    )
    print(
        f"Within 10% margin: {within_10_pct} ({within_10_pct/len(rel_differences_pct)*100:.2f}% of valid comparisons)"
    )
    print(
        f"Within 20% margin: {within_20_pct} ({within_20_pct/len(rel_differences_pct)*100:.2f}% of valid comparisons)"
    )

    print("\nPrediction Bias:")
    print(
        f"Under-predictions (measured > predicted): {under_predictions} ({under_predictions/len(matching_uids)*100:.2f}%)"
    )
    print(
        f"Over-predictions (measured < predicted): {over_predictions} ({over_predictions/len(matching_uids)*100:.2f}%)"
    )
    print(
        f"Exact matches: {exact_matches} ({exact_matches/len(matching_uids)*100:.2f}%)"
    )

    # Return the metrics for potential further use
    return {
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "under_predictions": under_predictions,
        "over_predictions": over_predictions,
    }


def load_model_predictions(json_file_path):
    """Load model predictions from a JSON file."""
    if not os.path.exists(json_file_path):
        print(f"Error: Model file {json_file_path} not found")
        return {}

    try:
        with open(json_file_path, "r") as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"Error loading model file {json_file_path}: {e}")
        return {}

    # Create a dictionary mapping schedule UIDs to their predicted times
    predictions = {}
    for schedule in model_data:
        if (
            "uid" in schedule
            and "metrics" in schedule
            and "max_time" in schedule["metrics"]
        ):
            uid = schedule["uid"]
            predicted_time = schedule["metrics"]["max_time"]
            predictions[uid] = predicted_time

    print(f"Loaded {len(predictions)} model predictions from {json_file_path}")
    return predictions


def print_comparison_results(widest_chunks, model_predictions):
    """Print comparison between measured widest chunks and model predictions."""
    if not model_predictions:
        return

    print("\n===== MEASURED VS PREDICTED TIMES =====")
    print(
        "Schedule UID                    : Measured (ms)  Predicted (ms)  Difference (%)  "
    )
    print("-" * 80)

    # Count matches and total comparisons
    total_comparisons = 0
    within_5_percent = 0
    within_10_percent = 0
    within_20_percent = 0

    # Calculate statistics
    rmse = 0
    mae = 0

    # Create a list of tuples for sorting
    comparison_data = []
    for schedule_uid, chunk_info in widest_chunks.items():
        measured_time = chunk_info["duration_ms"]

        if schedule_uid in model_predictions:
            predicted_time = model_predictions[schedule_uid]
            comparison_data.append((schedule_uid, measured_time, predicted_time))

    # Sort by predicted_time (ascending)
    comparison_data.sort(key=lambda x: x[2])

    # Print sorted comparison data
    for schedule_uid, measured_time, predicted_time in comparison_data:
        difference = measured_time - predicted_time
        diff_percent = (
            (difference / predicted_time) * 100 if predicted_time != 0 else float("inf")
        )

        print(
            f"{schedule_uid:30} : {measured_time:12.2f}  {predicted_time:14.2f}  {diff_percent:+14.2f}%"
        )

        # Update statistics
        total_comparisons += 1
        if abs(diff_percent) <= 5:
            within_5_percent += 1
        if abs(diff_percent) <= 10:
            within_10_percent += 1
        if abs(diff_percent) <= 20:
            within_20_percent += 1

        # Update error metrics
        rmse += (measured_time - predicted_time) ** 2
        mae += abs(measured_time - predicted_time)

    # Print UIDs not in model predictions
    for schedule_uid, chunk_info in widest_chunks.items():
        if schedule_uid not in model_predictions:
            measured_time = chunk_info["duration_ms"]
            print(f"{schedule_uid:30} : {measured_time:12.2f}  {'N/A':14}  {'N/A':14}")

    # Print statistics summary
    if total_comparisons > 0:
        rmse = (rmse / total_comparisons) ** 0.5
        mae = mae / total_comparisons

        print("\nComparison Statistics:")
        print(f"Total comparisons: {total_comparisons}")
        print(
            f"Within 5% margin: {within_5_percent} ({within_5_percent/total_comparisons*100:.2f}%)"
        )
        print(
            f"Within 10% margin: {within_10_percent} ({within_10_percent/total_comparisons*100:.2f}%)"
        )
        print(
            f"Within 20% margin: {within_20_percent} ({within_20_percent/total_comparisons*100:.2f}%)"
        )
        print(f"Root Mean Square Error (RMSE): {rmse:.4f} ms")
        print(f"Mean Absolute Error (MAE): {mae:.4f} ms")


def main():
    """Main function to process all log files."""
    args = parse_arguments()

    # Parse the time window argument
    try:
        time_window_parts = args.time_window.split("-")
        if len(time_window_parts) != 2:
            raise ValueError("Time window must be in format 'start-end'")

        start = float(time_window_parts[0])
        end = float(time_window_parts[1])

        if start < 0 or start > 1 or end < 0 or end > 1 or start >= end:
            raise ValueError(
                "Time window values must be between 0 and 1, and start must be less than end"
            )

        time_window = (start, end)
    except ValueError as e:
        print(f"Error parsing time window: {e}")
        print("Using default time window (0.0-1.0)")
        time_window = (0.0, 1.0)

    print(f"Using time window: {time_window[0]:.2f}-{time_window[1]:.2f}")

    # Load model predictions if specified
    model_predictions = {}
    if args.model:
        model_predictions = load_model_predictions(args.model)

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

    # Print individual statistics if verbose mode is enabled
    if args.verbose:
        print_individual_statistics(all_schedules)

    # Calculate and print aggregated statistics
    aggregated_stats, raw_data_by_uid = calculate_aggregated_statistics(all_schedules)
    print_aggregated_statistics(aggregated_stats)

    # Extract widest chunks for comparison with model
    widest_chunks = {}
    for schedule_uid, stats in aggregated_stats.items():
        # Find the widest chunk for this schedule
        widest_chunk_id = None
        widest_chunk_duration = 0

        for chunk_id, chunk_stats in stats["chunks"].items():
            avg_duration = chunk_stats["avg_duration_ms"]
            if avg_duration > widest_chunk_duration:
                widest_chunk_duration = avg_duration
                widest_chunk_id = chunk_id

        # Store widest chunk info
        if widest_chunk_id is not None:
            widest_chunks[schedule_uid] = {
                "chunk_id": widest_chunk_id,
                "duration_ms": widest_chunk_duration,
            }

    # Print widest chunk summary
    print("\n===== WIDEST CHUNK SUMMARY =====")
    print(f"Time window: {time_window[0]:.2f}-{time_window[1]:.2f}")
    print("Schedule UID                    : Chunk ID  Duration (ms)")
    print("-" * 60)

    for schedule_uid, chunk_info in sorted(widest_chunks.items()):
        print(
            f"{schedule_uid:30} : Chunk {chunk_info['chunk_id']:2}   {chunk_info['duration_ms']:.2f} ms"
        )

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
