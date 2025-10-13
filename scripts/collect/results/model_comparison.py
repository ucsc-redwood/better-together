#!/usr/bin/env python3
"""
Model comparison module for schedule analysis.

This module handles loading model predictions and comparing them
with measured schedule execution times.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple


def load_model_predictions(json_file_path: str) -> Dict[str, float]:
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


def print_comparison_results(
    widest_chunks: Dict[str, Dict[str, Any]], model_predictions: Dict[str, float]
) -> None:
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


def perform_statistical_analysis(
    widest_chunks: Dict[str, Dict[str, Any]], model_predictions: Dict[str, float]
) -> Dict[str, float]:
    """Perform detailed statistical analysis on measured vs predicted times."""
    if not model_predictions or not widest_chunks:
        return {}

    # Extract data for matched UIDs
    matching_uids = sorted(set(widest_chunks.keys()) & set(model_predictions.keys()))

    if not matching_uids:
        print("No matching UIDs found between measurements and predictions")
        return {}

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
