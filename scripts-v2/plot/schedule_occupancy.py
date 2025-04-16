import os
import re
from collections import defaultdict
import sys
import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to the input file containing multiple schedules' timing data",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time as percentage (0.0-1.0) of overall execution",
        default=0.0,
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End time as percentage (0.0-1.0) of overall execution",
        default=1.0,
    )
    args = parser.parse_args()

    # Validate percentage inputs
    if args.start_time < 0.0 or args.start_time > 1.0:
        raise ValueError("start-time must be between 0.0 and 1.0")
    if args.end_time < 0.0 or args.end_time > 1.0:
        raise ValueError("end-time must be between 0.0 and 1.0")
    if args.end_time <= args.start_time:
        raise ValueError("end-time must be greater than start-time")

    return args


def extract_python_sections(content):
    """Extract all Python sections from the file."""
    python_sections = re.findall(
        r"### Python Begin ###(.*?)### Python End ###", content, re.DOTALL
    )

    if not python_sections:
        print("No Python sections found in the input file.")
        sys.exit(1)

    print(f"Found {len(python_sections)} schedule(s) in the input file.")
    return python_sections


def extract_frequency(section):
    """Extract frequency information from a Python section."""
    freq_match = re.search(r"# Frequency=(\d+) Hz", section)
    if freq_match:
        return int(freq_match.group(1))
    return 24576000  # Default frequency in Hz


def extract_schedule_annotations(section):
    """Extract schedule annotations from a Python section."""
    schedule_annotations = {}
    schedule_lines = []

    for line in section.strip().split("\n"):
        if line.startswith("["):
            schedule_lines.append(line)

    for line in schedule_lines:
        parts = line.split()
        if len(parts) >= 2:
            core_type = parts[0].strip("[]")
            chunk_ids = [int(chunk_id) for chunk_id in parts[1:]]
            schedule_annotations[core_type] = chunk_ids

    return schedule_annotations


def parse_task_data(section):
    """Parse task data from a Python section."""
    tasks = {}
    pattern = (
        r"Task=(\d+) Chunk=(\d+) Processor=(\w+) Start=(\d+) End=(\d+) Duration=(\d+)"
    )
    task_matches = re.findall(pattern, section)

    for match in task_matches:
        task_id = int(match[0])
        chunk_id = int(match[1])
        processor = match[2]
        start = int(match[3])
        end = int(match[4])
        duration = int(match[5])

        if task_id not in tasks:
            tasks[task_id] = {}

        tasks[task_id][chunk_id] = {
            "type": processor,
            "start": start,
            "end": end,
            "duration_cycles": duration,
        }

    return tasks


def get_chunk_types(tasks):
    """Collect all chunk types from the tasks."""
    chunk_types = {}
    for task_id in tasks:
        for chunk_id in tasks[task_id]:
            if chunk_id not in chunk_types and "type" in tasks[task_id][chunk_id]:
                chunk_types[chunk_id] = tasks[task_id][chunk_id]["type"]
    return chunk_types


def calculate_time_range(tasks, args, schedule_index):
    """Calculate the time range for visualization based on task data."""
    try:
        # Get non-zero start times
        start_times = [
            chunk_data["start"]
            for task_data in tasks.values()
            for chunk_data in task_data.values()
            if "start" in chunk_data and chunk_data["start"] > 0
        ]

        if not start_times:
            print(
                f"WARNING: No valid start times found in Schedule {schedule_index+1}, skipping."
            )
            return None

        # Find the minimum start time, but also check for outliers
        # Sort the start times
        sorted_starts = sorted(start_times)
        min_time = sorted_starts[0]

        # Check for potential outliers (very early starts that might skew visualization)
        if len(sorted_starts) > 1:
            # Calculate the gap between the smallest and second smallest
            gap = sorted_starts[1] - sorted_starts[0]

            # If the gap is large, it might indicate an outlier
            if gap > 1000000:  # arbitrary threshold, adjust as needed
                print(
                    f"WARNING: Possible outlier detected - earliest time: {sorted_starts[0]}, next: {sorted_starts[1]}, gap: {gap}"
                )
                print(f"This might cause some chunks to appear at incorrect positions.")

        # Find the maximum end time for percentage calculation
        end_times = [
            chunk_data["end"]
            for task_data in tasks.values()
            for chunk_data in task_data.values()
            if "end" in chunk_data and chunk_data["end"] > 0
        ]

        if not end_times:
            print(
                f"WARNING: No valid end times found in Schedule {schedule_index+1}, skipping."
            )
            return None

        max_time = max(end_times)

        # Calculate total time span
        time_span = max_time - min_time

        if time_span <= 0:
            print(
                f"WARNING: Invalid time span (<=0) in Schedule {schedule_index+1}, skipping."
            )
            return None

        # Calculate the actual start and end times based on percentages
        filter_start_time = min_time + (args.start_time * time_span)
        filter_end_time = min_time + (args.end_time * time_span)

        print(f"Total time span: {time_span} cycles")
        print(f"Min time: {min_time} cycles, Max time: {max_time} cycles")
        print(f"Filtering time range: {filter_start_time} to {filter_end_time} cycles")

        return {
            "min_time": min_time,
            "max_time": max_time,
            "time_span": time_span,
            "filter_start_time": filter_start_time,
            "filter_end_time": filter_end_time,
        }

    except Exception as e:
        print(
            f"ERROR: Failed to calculate time span for Schedule {schedule_index+1}: {e}"
        )
        return None


def normalize_task_times(tasks, min_time):
    """Normalize task times relative to min_time and calculate durations."""
    # First, find all valid start and end times
    all_start_times = []
    for task_id in tasks:
        for chunk_id in tasks[task_id]:
            if (
                "start" in tasks[task_id][chunk_id]
                and tasks[task_id][chunk_id]["start"] > 0
            ):
                all_start_times.append(tasks[task_id][chunk_id]["start"])

    # For debugging purposes, print out some stats
    if all_start_times:
        min_start = min(all_start_times)
        max_start = max(all_start_times)
        print(
            f"  Time stats - min start: {min_start}, max start: {max_start}, global min: {min_time}"
        )

        # If there's a large gap between min_time and min_start, warn about potential issue
        if min_start - min_time > 100000:  # arbitrary threshold to detect outliers
            print(
                f"  WARNING: Large gap detected between global min time ({min_time}) and minimum start time ({min_start})"
            )
            print(f"  This might cause some chunks to appear at incorrect positions.")

    # Now normalize all task times
    for task_id in tasks:
        for chunk_id in tasks[task_id]:
            if (
                "start" in tasks[task_id][chunk_id]
                and "end" in tasks[task_id][chunk_id]
            ):
                # Calculate relative time from start (in cycles)
                start_time = tasks[task_id][chunk_id]["start"]
                end_time = tasks[task_id][chunk_id]["end"]

                tasks[task_id][chunk_id]["start_norm"] = start_time - min_time
                tasks[task_id][chunk_id]["end_norm"] = end_time - min_time

                # Use provided duration if available, otherwise calculate it
                if "duration_cycles" in tasks[task_id][chunk_id]:
                    duration_cycles = tasks[task_id][chunk_id]["duration_cycles"]
                else:
                    duration_cycles = end_time - start_time

                # Store the duration in cycles
                tasks[task_id][chunk_id]["duration_cycles"] = duration_cycles


def collect_chunk_data(tasks, time_range, CYCLES_TO_MS, max_chunk_id):
    """Collect data about chunks for analysis."""
    chunks_data = {chunk_id: [] for chunk_id in range(max_chunk_id + 1)}
    tasks_in_region = set()
    task_durations_by_chunk = defaultdict(lambda: defaultdict(list))
    chunk_total_durations = defaultdict(float)

    # Debug: print min time to help debug positioning issues
    min_time = time_range["min_time"]

    for task_id in tasks:
        for chunk_id in tasks[task_id]:
            if (
                "start" in tasks[task_id][chunk_id]
                and "duration_cycles" in tasks[task_id][chunk_id]
            ):
                # Only include chunks that overlap with our time range
                start_time = tasks[task_id][chunk_id]["start"]
                end_time = tasks[task_id][chunk_id]["end"]

                # Skip chunks entirely outside our time range or with zero duration
                if (
                    end_time < time_range["filter_start_time"]
                    or start_time > time_range["filter_end_time"]
                    or tasks[task_id][chunk_id]["duration_cycles"] == 0
                ):
                    continue

                # Track tasks in this region
                tasks_in_region.add(task_id)

                # Track durations for each task in each chunk
                duration_cycles = tasks[task_id][chunk_id]["duration_cycles"]
                task_durations_by_chunk[chunk_id][task_id].append(duration_cycles)

                # Track total duration for each chunk
                chunk_total_durations[chunk_id] += duration_cycles

                # Ensure start_norm is calculated correctly
                if "start_norm" not in tasks[task_id][chunk_id]:
                    tasks[task_id][chunk_id]["start_norm"] = start_time - min_time

                chunks_data[chunk_id].append(
                    {
                        "task_id": task_id,
                        "start_ms": (tasks[task_id][chunk_id]["start_norm"])
                        * CYCLES_TO_MS,
                        "duration_ms": tasks[task_id][chunk_id]["duration_cycles"]
                        * CYCLES_TO_MS,
                        "type": tasks[task_id][chunk_id].get("type", ""),
                    }
                )

    return chunks_data, tasks_in_region, task_durations_by_chunk, chunk_total_durations


def calculate_average_durations(task_durations_by_chunk, CYCLES_TO_MS):
    """Calculate average duration for each chunk."""
    chunk_avg_durations_ms = {}
    for chunk_id in task_durations_by_chunk:
        all_durations = [
            d
            for task_durations in task_durations_by_chunk[chunk_id].values()
            for d in task_durations
        ]
        if all_durations:
            avg_duration_cycles = sum(all_durations) / len(all_durations)
            avg_duration_ms = avg_duration_cycles * CYCLES_TO_MS
            chunk_avg_durations_ms[chunk_id] = avg_duration_ms

    return chunk_avg_durations_ms


def create_chunk_labels(max_chunk_id, chunk_types, schedule_annotations):
    """Create labels for chunks based on their types and schedule annotations."""
    chunk_names = []
    for i in range(max_chunk_id + 1):
        # Find which core type this chunk belongs to based on schedule annotations
        core_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if i in chunks:
                core_type = core
                break

        processor_type = chunk_types.get(i, "Unknown")
        chunk_names.append(f"Chunk {i} ({processor_type}/{core_type})")

    return chunk_names


def print_analysis(
    schedule_index,
    tasks_in_region,
    task_durations_by_chunk,
    chunk_total_durations,
    chunk_avg_durations_ms,
    chunk_types,
    schedule_annotations,
    CYCLES_TO_MS,
    args,
    max_chunk_id,
):
    """Print analysis of the schedule execution."""
    # 1. Print all tasks in the region
    print(f"\n======= SCHEDULE {schedule_index+1} - TASKS IN SELECTED REGION =======")
    sorted_tasks = sorted(tasks_in_region)
    print(
        f"Found {len(sorted_tasks)} tasks in the region {args.start_time:.2f}-{args.end_time:.2f}"
    )
    print(f"Tasks: {', '.join(map(str, sorted_tasks))}")

    # 2. Calculate and print average duration for each task in each chunk
    print(
        f"\n======= SCHEDULE {schedule_index+1} - AVERAGE TASK DURATIONS BY CHUNK ======="
    )
    for chunk_id in range(max_chunk_id + 1):
        if chunk_id not in task_durations_by_chunk:
            continue

        # Find which core type this chunk belongs to based on schedule annotations
        core_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if chunk_id in chunks:
                core_type = core
                break

        processor_type = chunk_types.get(chunk_id, "Unknown")
        print(f"Chunk {chunk_id} ({processor_type}/{core_type}):")

        # Print average across all tasks for this chunk
        all_durations = [
            d
            for task_durations in task_durations_by_chunk[chunk_id].values()
            for d in task_durations
        ]
        if all_durations:
            avg_all_tasks_cycles = sum(all_durations) / len(all_durations)
            avg_all_tasks_ms = avg_all_tasks_cycles * CYCLES_TO_MS
            total_duration_cycles = chunk_total_durations[chunk_id]
            total_duration_ms = total_duration_cycles * CYCLES_TO_MS
            print(
                f"  All Tasks Average: {avg_all_tasks_cycles:.2f} cycles ({avg_all_tasks_ms:.6f} ms)"
            )
            print(
                f"  Total Duration: {total_duration_cycles:.2f} cycles ({total_duration_ms:.6f} ms)"
            )
        print()

    # 3. Find the widest chunk (most execution time) in the region
    print(f"\n======= SCHEDULE {schedule_index+1} - WIDEST CHUNK IN REGION =======")
    if chunk_total_durations:
        widest_chunk_id = max(chunk_total_durations, key=chunk_total_durations.get)
        widest_chunk_duration_cycles = chunk_total_durations[widest_chunk_id]
        widest_chunk_duration_ms = widest_chunk_duration_cycles * CYCLES_TO_MS

        # Find which core type this chunk belongs to based on schedule annotations
        core_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if widest_chunk_id in chunks:
                core_type = core
                break

        processor_type = chunk_types.get(widest_chunk_id, "Unknown")

        print(f"Widest chunk: Chunk {widest_chunk_id} ({processor_type}/{core_type})")
        print(
            f"Total execution time: {widest_chunk_duration_cycles:.2f} cycles ({widest_chunk_duration_ms:.6f} ms)"
        )

        # Calculate percentage of time spent in this chunk
        total_execution_cycles = sum(chunk_total_durations.values())
        percentage = (widest_chunk_duration_cycles / total_execution_cycles) * 100
        print(f"Percentage of selected region execution time: {percentage:.2f}%")

        # Show execution breakdown for all chunks
        print(f"\nExecution time breakdown for all chunks in region:")
        for chunk_id, duration in sorted(
            chunk_total_durations.items(), key=lambda x: x[1], reverse=True
        ):
            # Find which core type this chunk belongs to based on schedule annotations
            c_type = "Unknown"
            for core, chunks in schedule_annotations.items():
                if chunk_id in chunks:
                    c_type = core
                    break

            p_type = chunk_types.get(chunk_id, "Unknown")
            total_duration_cycles = duration
            total_duration_ms = duration * CYCLES_TO_MS
            chunk_percentage = (duration / total_execution_cycles) * 100

            # Add average duration info if available
            avg_info = ""
            if chunk_id in chunk_avg_durations_ms:
                avg_duration_ms = chunk_avg_durations_ms[chunk_id]
                avg_info = f" | Avg task: {avg_duration_ms:.6f} ms"

            print(
                f"  Chunk {chunk_id} ({p_type}/{c_type}): Total: {total_duration_cycles:.2f} cycles ({total_duration_ms:.6f} ms){avg_info}, {chunk_percentage:.2f}%"
            )
    else:
        print(
            f"No chunk execution data found in the selected region for Schedule {schedule_index+1}."
        )


def print_schedule_summary(
    schedule_index, 
    chunk_total_durations, 
    chunk_avg_durations_ms,
    chunk_types,
    schedule_annotations,
    CYCLES_TO_MS
):
    """Print a summary of the schedule execution."""
    print(f"\n======= SCHEDULE {schedule_index+1} - SUMMARY =======")
    
    if not chunk_total_durations:
        print("No data available for this schedule.")
        return
        
    # Calculate total execution time for this schedule
    total_execution_cycles = sum(chunk_total_durations.values())
    total_execution_ms = total_execution_cycles * CYCLES_TO_MS
    
    print(f"Total execution time: {total_execution_cycles:.2f} cycles ({total_execution_ms:.6f} ms)")
    
    # Group chunks by processor type
    processor_totals = defaultdict(float)
    core_totals = defaultdict(float)
    
    for chunk_id, duration in chunk_total_durations.items():
        # Get processor type
        p_type = chunk_types.get(chunk_id, "Unknown")
        processor_totals[p_type] += duration
        
        # Get core type
        c_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if chunk_id in chunks:
                c_type = core
                break
        core_totals[c_type] += duration
    
    # Print processor type breakdown
    print("\nExecution time by processor type:")
    for p_type, duration in sorted(processor_totals.items(), key=lambda x: x[1], reverse=True):
        duration_ms = duration * CYCLES_TO_MS
        percentage = (duration / total_execution_cycles) * 100
        print(f"  {p_type}: {duration:.2f} cycles ({duration_ms:.6f} ms), {percentage:.2f}%")
    
    # Print core type breakdown
    print("\nExecution time by core type:")
    for c_type, duration in sorted(core_totals.items(), key=lambda x: x[1], reverse=True):
        duration_ms = duration * CYCLES_TO_MS
        percentage = (duration / total_execution_cycles) * 100
        print(f"  {c_type}: {duration:.2f} cycles ({duration_ms:.6f} ms), {percentage:.2f}%")


def process_schedule(section, schedule_index, args):
    """Process a single schedule section."""
    print(f"\n=== Processing Schedule {schedule_index+1} ===")

    # Parse the task data
    frequency = extract_frequency(section)
    print(f"Detected frequency: {frequency} Hz")

    schedule_annotations = extract_schedule_annotations(section)
    if schedule_annotations:
        print(f"Detected schedule annotations: {schedule_annotations}")

    tasks = parse_task_data(section)
    if not tasks:
        print(f"No task data found in Schedule {schedule_index+1}, skipping.")
        return

    # Find the maximum chunk ID in the data
    max_chunk_id = max(
        [chunk_id for task_data in tasks.values() for chunk_id in task_data.keys()]
    )
    print(f"Detected {max_chunk_id + 1} chunks in Schedule {schedule_index+1}")

    # Collect all chunk types for the legend
    chunk_types = get_chunk_types(tasks)

    # Calculate time range
    time_range = calculate_time_range(tasks, args, schedule_index)
    if time_range is None:
        return

    # Normalize task times
    normalize_task_times(tasks, time_range["min_time"])

    # Calculate cycles to ms conversion factor
    CYCLES_TO_MS = 1e3 / frequency  # Convert cycles to milliseconds

    # Collect chunk data
    chunks_data, tasks_in_region, task_durations_by_chunk, chunk_total_durations = (
        collect_chunk_data(tasks, time_range, CYCLES_TO_MS, max_chunk_id)
    )

    # Calculate average durations
    chunk_avg_durations_ms = calculate_average_durations(
        task_durations_by_chunk, CYCLES_TO_MS
    )

    # Sort chunk data by start time
    for chunk_id in chunks_data:
        chunks_data[chunk_id].sort(key=lambda x: x["start_ms"])

    # Create chunk labels
    chunk_names = create_chunk_labels(max_chunk_id, chunk_types, schedule_annotations)

    # Print analysis
    print_analysis(
        schedule_index,
        tasks_in_region,
        task_durations_by_chunk,
        chunk_total_durations,
        chunk_avg_durations_ms,
        chunk_types,
        schedule_annotations,
        CYCLES_TO_MS,
        args,
        max_chunk_id,
    )
    
    # Print schedule summary
    print_schedule_summary(
        schedule_index,
        chunk_total_durations,
        chunk_avg_durations_ms,
        chunk_types,
        schedule_annotations,
        CYCLES_TO_MS
    )


def main():
    """Main function to process all schedules in the input file."""
    args = parse_arguments()

    # Read the input file
    with open(args.input_file, "r") as f:
        content = f.read()

    # Extract Python sections
    python_sections = extract_python_sections(content)

    # Process each Python section
    for schedule_index, section in enumerate(python_sections):
        process_schedule(section, schedule_index, args)

    print(f"\nAll {len(python_sections)} schedules have been processed.")


if __name__ == "__main__":
    main()
