#!/usr/bin/env python3
"""
Statistics calculation module for schedule analysis.

This module handles the calculation of aggregated statistics,
widest chunk analysis, and statistical reporting.
"""

import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def print_individual_statistics(schedules_data: List[Dict[str, Any]]) -> None:
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


def calculate_aggregated_statistics(
    all_schedules: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
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

    return aggregated_stats, dict(grouped_schedules)


def print_aggregated_statistics(aggregated_stats: Dict[str, Any]) -> None:
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


def extract_widest_chunks(
    aggregated_stats: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Extract widest chunks for comparison with model predictions."""
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

    return widest_chunks


def print_widest_chunk_summary(
    widest_chunks: Dict[str, Dict[str, Any]], time_window: Tuple[float, float]
) -> None:
    """Print widest chunk summary."""
    print("\n===== WIDEST CHUNK SUMMARY =====")
    print(f"Time window: {time_window[0]:.2f}-{time_window[1]:.2f}")
    print("Schedule UID                    : Chunk ID  Duration (ms)")
    print("-" * 60)

    for schedule_uid, chunk_info in sorted(widest_chunks.items()):
        print(
            f"{schedule_uid:30} : Chunk {chunk_info['chunk_id']:2}   {chunk_info['duration_ms']:.2f} ms"
        )
