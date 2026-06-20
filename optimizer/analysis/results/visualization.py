#!/usr/bin/env python3
"""
Visualization module for schedule analysis.

This module handles the creation of charts and plots for comparing
measured vs predicted execution times.
"""

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def create_comparison_visualization(
    widest_chunks: Dict[str, Dict[str, Any]],
    model_predictions: Dict[str, float],
    output_dir: str,
    raw_data: Dict[str, List[Dict[str, Any]]],
) -> None:
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
    measured_bars = plt.bar(x - width / 2, measured_times, width, label="Measured", alpha=0.7)
    predicted_bars = plt.bar(x + width / 2, predicted_times, width, label="Predicted", alpha=0.7)

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
    plt.ylabel("Time (ms)", fontsize=16)

    # Use index IDs instead of UIDs
    plt.xticks(x, [str(i + 1) for i in range(len(x))], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)

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
                fontsize=10,
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
    schedule_uids: List[str],
    measured_times: np.ndarray,
    predicted_times: np.ndarray,
    error_bars: np.ndarray,
    output_dir: str,
    filename: str = "line_comparison_chart.png",
    title: str = "Comparison of Measured vs Predicted Execution Times",
    sort_note: str = "",
) -> None:
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
        markersize=16,  # Increased from 14 to 16
        linewidth=1.5,  # Increased from 1 to 1.5
        label="Predicted",
        alpha=0.9,
    )
    plt.plot(
        x,
        measured_times,
        "b-",
        marker="^",
        markersize=16,  # Increased from 14 to 16
        linewidth=1.5,  # Increased from 1 to 1.5
        label="Measured",
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
    plt.ylabel(
        "Time (Execution in ms)", fontsize=16, labelpad=10
    )  # Increased font size from 14 to 16

    # Add index IDs on x-axis instead of UIDs
    index_ids = [str(i + 1) for i in range(len(x))]
    plt.xticks(x, index_ids, fontsize=14)  # Increased font size from 10 to 14

    # Add grid for both axes
    plt.grid(True, linestyle="--", alpha=0.7, which="both")

    # Create legend with larger font and better position
    plt.legend(fontsize=20, loc="upper left", markerscale=1.5)  # Increased font size from 18 to 20

    # Set y-axis to start at 0
    # Calculate a good maximum y value that leaves room for highest point plus error bar
    max_y = max(max(predicted_times), max(measured_times) + max(error_bars)) * 1.15
    plt.ylim(bottom=0, top=max_y)
    plt.yticks(fontsize=14)  # Increased font size from 12 to 14

    # Make plot lines thicker
    for line in plt.gca().get_lines():
        if line.get_linestyle() == "--":  # Predicted line
            line.set_linewidth(2)  # Increased from 1.5 to 2
        elif line.get_marker() == "^":  # Measured line
            line.set_linewidth(2)  # Increased from 1.5 to 2

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
    schedule_uids: List[str],
    predicted_times: np.ndarray,
    measured_times: np.ndarray,
    output_dir: str,
) -> None:
    """Create various correlation plots to better visualize the data."""

    # Main correlation plot (standard)
    plt.figure(figsize=(10, 8))
    plt.scatter(predicted_times, measured_times, alpha=0.7, s=80)  # Increased point size

    # Add diagonal line (perfect prediction)
    max_val = max(np.max(predicted_times), np.max(measured_times)) * 1.1
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction", linewidth=2)

    # Add labels
    plt.xlabel("Predicted Time (ms)", fontsize=14)
    plt.ylabel("Measured Time (ms)", fontsize=14)
    plt.title("Correlation between Predicted and Measured Times", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add index labels to points instead of UIDs
    for i in range(len(schedule_uids)):
        plt.annotate(
            str(i + 1),  # Use index instead of UID
            (predicted_times[i], measured_times[i]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=10,  # Increased font size
        )

    # Calculate correlation coefficient
    correlation = np.corrcoef(predicted_times, measured_times)[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.4f}",
        transform=plt.gca().transAxes,
        fontsize=14,  # Increased font size
        verticalalignment="top",
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=14)  # Increased font size
    plt.tight_layout()

    # Save standard scatter plot
    plt.savefig(os.path.join(output_dir, "correlation_plot.png"), dpi=300)
    print(f"Correlation plot saved to {os.path.join(output_dir, 'correlation_plot.png')}")

    # 1. Create log-scale plot for better distribution visualization
    plt.figure(figsize=(10, 8))

    # Skip zeros and negative values for log scale
    valid_indices = (predicted_times > 0) & (measured_times > 0)
    valid_pred = predicted_times[valid_indices]
    valid_meas = measured_times[valid_indices]
    valid_uids = [schedule_uids[i] for i, valid in enumerate(valid_indices) if valid]

    if len(valid_pred) > 0:
        plt.scatter(valid_pred, valid_meas, alpha=0.7, s=80)  # Increased point size

        # Add perfect prediction line on log scale
        min_val = min(np.min(valid_pred), np.min(valid_meas)) * 0.9
        max_val = max(np.max(valid_pred), np.max(valid_meas)) * 1.1
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect Prediction",
            linewidth=2,
        )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Predicted Time (ms) - Log Scale", fontsize=14)
        plt.ylabel("Measured Time (ms) - Log Scale", fontsize=14)
        plt.title("Log-Scale Correlation between Predicted and Measured Times", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add index labels to points instead of UIDs
        for i, idx in enumerate(np.where(valid_indices)[0]):
            plt.annotate(
                str(idx + 1),  # Use index instead of UID
                (valid_pred[i], valid_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=10,  # Increased font size
            )

        # Calculate correlation coefficient for valid points
        log_correlation = np.corrcoef(np.log(valid_pred), np.log(valid_meas))[0, 1]
        plt.text(
            0.05,
            0.95,
            f"Log Correlation: {log_correlation:.4f}",
            transform=plt.gca().transAxes,
            fontsize=14,  # Increased font size
            verticalalignment="top",
        )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)  # Increased font size
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
    non_outlier_idxs = np.where(non_outlier_indices)[0]

    if len(non_outlier_indices) > 0:
        plt.scatter(non_outlier_pred, non_outlier_meas, alpha=0.7, s=80)  # Increased point size

        # Add perfect prediction line
        min_val = min(np.min(non_outlier_pred), np.min(non_outlier_meas)) * 0.9
        max_val = max(np.max(non_outlier_pred), np.max(non_outlier_meas)) * 1.1
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Perfect Prediction",
            linewidth=2,
        )

        plt.xlabel("Predicted Time (ms)", fontsize=14)
        plt.ylabel("Measured Time (ms)", fontsize=14)
        plt.title("Correlation (Excluding Outliers)", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add index labels
        for i, idx in enumerate(non_outlier_idxs):
            plt.annotate(
                str(idx + 1),  # Use index instead of UID
                (non_outlier_pred[i], non_outlier_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=10,
            )

        # Calculate correlation coefficient for non-outliers
        if len(non_outlier_pred) > 1:  # Need at least 2 points for correlation
            non_outlier_correlation = np.corrcoef(non_outlier_pred, non_outlier_meas)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Correlation (excl. outliers): {non_outlier_correlation:.4f}",
                transform=plt.gca().transAxes,
                fontsize=14,
                verticalalignment="top",
            )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()

        # Save non-outlier plot
        plt.savefig(os.path.join(output_dir, "correlation_plot_no_outliers.png"), dpi=300)
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
    plt.scatter(
        predicted_times, measured_times, alpha=0.5, color="lightgray", s=60
    )  # Increased point size

    # Highlight points in the zoom window
    zoom_indices = (
        (predicted_times >= zoom_min_pred)
        & (predicted_times <= zoom_max_pred)
        & (measured_times >= zoom_min_meas)
        & (measured_times <= zoom_max_meas)
    )

    zoom_pred = predicted_times[zoom_indices]
    zoom_meas = measured_times[zoom_indices]
    zoom_idxs = np.where(zoom_indices)[0]

    if len(zoom_indices) > 0:
        plt.scatter(zoom_pred, zoom_meas, alpha=0.9, s=80)  # Increased point size

        # Add perfect prediction line just for the zoom window
        zoom_min = min(zoom_min_pred, zoom_min_meas)
        zoom_max = max(zoom_max_pred, zoom_max_meas)
        plt.plot(
            [zoom_min, zoom_max],
            [zoom_min, zoom_max],
            "r--",
            label="Perfect Prediction",
            linewidth=2,
        )

        # Set limits to zoom window
        plt.xlim(zoom_min_pred, zoom_max_pred)
        plt.ylim(zoom_min_meas, zoom_max_meas)

        plt.xlabel("Predicted Time (ms)", fontsize=14)
        plt.ylabel("Measured Time (ms)", fontsize=14)
        plt.title("Zoomed Correlation View (Focused on Cluster)", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add labels for points in the zoom window
        for i, idx in enumerate(zoom_idxs):
            plt.annotate(
                str(idx + 1),  # Use index instead of UID
                (zoom_pred[i], zoom_meas[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=10,
            )

        # Calculate correlation coefficient for zoomed region
        if len(zoom_pred) > 1:  # Need at least 2 points for correlation
            zoom_correlation = np.corrcoef(zoom_pred, zoom_meas)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Zoom Correlation: {zoom_correlation:.4f}",
                transform=plt.gca().transAxes,
                fontsize=14,
                verticalalignment="top",
            )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()

        # Save zoomed plot
        plt.savefig(os.path.join(output_dir, "correlation_plot_zoomed.png"), dpi=300)
        print(
            f"Zoomed correlation plot saved to {os.path.join(output_dir, 'correlation_plot_zoomed.png')}"
        )

    # Close all figures to free memory
    plt.close("all")
