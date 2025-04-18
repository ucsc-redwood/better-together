import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import argparse

# =======================================================================================
#  Given a log file (contains multiple schedules), plot the execution timeline for each
#  example:
#  python3 scripts/plot/timeline.py 3A021JEHN02756_cifar-sparse_schedules.log --output-dir tmp_folder
# =======================================================================================


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to the input file containing multiple schedules' timing data",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for the figures",
        default=".",
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


def extract_schedule_uid(section):
    """Extract Schedule_UID from a Python section."""
    uid_match = re.search(r"Schedule_UID: ([\w\-]+)", section)
    if uid_match:
        return uid_match.group(1)
    return None


def extract_schedule_annotations(section):
    """Extract schedule annotations from a Python section."""
    schedule_annotations = {}
    schedule_lines = []

    for line in section.strip().split("\n"):
        if line.startswith("  Chunk "):
            schedule_lines.append(line)

    for line in schedule_lines:
        # Extract chunk ID - Example: "  Chunk 0 [Vulkan    ]: 1, 2, 3, 4, 5, 6, 7"
        chunk_id_match = re.search(r"Chunk (\d+)", line)
        if not chunk_id_match:
            continue

        chunk_id = int(chunk_id_match.group(1))

        # Extract assigned tasks - Example: "]: 1, 2, 3, 4, 5, 6, 7"
        tasks_match = re.search(r"]: ([\d, ]+)", line)
        if not tasks_match:
            continue

        tasks_str = tasks_match.group(1)
        tasks = [int(task.strip()) for task in tasks_str.split(",") if task.strip()]

        # Store with chunk ID as the key
        schedule_annotations[chunk_id] = tasks

    return schedule_annotations


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


def get_chunk_types(tasks):
    """Collect all chunk types from the tasks."""
    # This function is no longer needed but kept as a stub for compatibility
    return {}


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
                # You could consider setting min_time = sorted_starts[1] here to exclude the outlier,
                # but that's a decision that depends on the specific dataset and requirements

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
    """Collect data about chunks for visualization and analysis."""
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
    """Create labels for chunks based only on their IDs and schedule annotations."""
    chunk_names = []
    for i in range(max_chunk_id + 1):
        # Get the tasks assigned to this chunk
        tasks = schedule_annotations.get(i, [])

        # Format the task list as a comma-separated string
        tasks_str = ", ".join(map(str, tasks))
        if tasks_str:
            chunk_names.append(f"Chunk {i} [{tasks_str}]")
        else:
            chunk_names.append(f"Chunk {i}")

    return chunk_names


def create_gantt_chart(
    chunks_data,
    chunk_names,
    time_range,
    chunk_avg_durations_ms,
    chunk_total_durations,
    CYCLES_TO_MS,
    schedule_index,
    schedule_uid,
    args,
    output_dir,
):
    """Create and save a Gantt chart visualization of the schedule execution."""
    # Get active chunks (those with data to display)
    active_chunks = [
        chunk_id for chunk_id in range(len(chunk_names)) if chunks_data[chunk_id]
    ]
    active_chunks.sort(reverse=True)  # Sort in reverse order for display

    # Check if there's any data to display
    if not active_chunks:
        print(
            f"WARNING: No chunks with non-zero duration found in the selected time range for Schedule {schedule_index+1}."
        )
        print("No figure will be generated.")
        return False

    # Calculate display range in milliseconds
    display_start_ms = (
        time_range["filter_start_time"] - time_range["min_time"]
    ) * CYCLES_TO_MS
    display_end_ms = (
        time_range["filter_end_time"] - time_range["min_time"]
    ) * CYCLES_TO_MS

    # Create a wider Gantt chart
    fig, ax = plt.subplots(figsize=(30, 8))  # Increased width to 30 inches

    # Task colors - use a color map for different tasks
    task_color_map = plt.colormaps["tab20"]

    # Draw the chunks
    y_ticks = []
    y_labels = []

    for i, chunk_id in enumerate(active_chunks):
        y_pos = i  # Position based on active chunks only
        y_ticks.append(y_pos)
        y_labels.append(chunk_names[chunk_id])

        # Draw each task in this chunk
        for task_data in chunks_data[chunk_id]:
            task_id = task_data["task_id"]
            start_ms = task_data["start_ms"]
            duration_ms = task_data["duration_ms"]

            # Generate color based on task_id for consistency
            color = task_color_map(task_id % 20)

            # Draw the bar
            bar = ax.barh(
                y_pos,
                duration_ms,
                left=start_ms,
                height=0.5,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # Add text label showing the task ID
            if duration_ms > 0.5:  # Only add label if bar is wide enough
                text_x = start_ms + duration_ms / 2
                ax.text(
                    text_x,
                    y_pos,
                    f"T{task_id}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                    fontsize=8,
                )

        # Add average duration text at the right side of each chunk's row
        if chunk_id in chunk_avg_durations_ms:
            avg_duration = chunk_avg_durations_ms[chunk_id]
            # Calculate total duration for this chunk
            total_duration_cycles = chunk_total_durations[chunk_id]
            total_duration_ms = total_duration_cycles * CYCLES_TO_MS

            ax.text(
                display_end_ms
                + (display_end_ms - display_start_ms)
                * 0.01,  # Position slightly to the right of the display range
                y_pos,
                f"Avg: {avg_duration:.4f} ms | Total: {total_duration_ms:.4f} ms",
                ha="left",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                    boxstyle="round,pad=0.3",
                ),
            )

    # Adjust the right margin to make room for the annotations
    plt.subplots_adjust(right=0.95)

    # Set chart properties
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (ms, converted from cycles)", fontsize=12)

    # Add schedule_uid to the title if available
    title = f"Schedule {schedule_index+1}"
    if schedule_uid:
        title += f" [UID: {schedule_uid}]"

    title += f" - Chunk Execution Timeline ({args.start_time*100:.0f}%-{args.end_time*100:.0f}% of execution)\n"
    title += f"Time range: {display_start_ms:.2f}ms - {display_end_ms:.2f}ms, {len(active_chunks)} active chunks"

    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Set the x-axis limits to our desired time range
    ax.set_xlim(display_start_ms, display_end_ms)

    # Save the figure with specified output name
    output_filename = f"schedule_{schedule_index+1}_timeline.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Chunk execution timeline saved to {os.path.abspath(output_path)}")
    print(f"Time range displayed: {display_start_ms:.2f}ms to {display_end_ms:.2f}ms")

    return True


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

        # Get task list for this chunk
        tasks = schedule_annotations.get(chunk_id, [])
        tasks_str = ", ".join(map(str, tasks))

        if tasks_str:
            print(f"Chunk {chunk_id} [{tasks_str}]:")
        else:
            print(f"Chunk {chunk_id}:")

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

        # Get task list for the widest chunk
        widest_tasks = schedule_annotations.get(widest_chunk_id, [])
        widest_tasks_str = ", ".join(map(str, widest_tasks))

        if widest_tasks_str:
            print(f"Widest chunk: Chunk {widest_chunk_id} [{widest_tasks_str}]")
        else:
            print(f"Widest chunk: Chunk {widest_chunk_id}")

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
            # Get task list for this chunk
            chunk_tasks = schedule_annotations.get(chunk_id, [])
            chunk_tasks_str = ", ".join(map(str, chunk_tasks))

            total_duration_cycles = duration
            total_duration_ms = duration * CYCLES_TO_MS
            chunk_percentage = (duration / total_execution_cycles) * 100

            # Add average duration info if available
            avg_info = ""
            if chunk_id in chunk_avg_durations_ms:
                avg_duration_ms = chunk_avg_durations_ms[chunk_id]
                avg_info = f" | Avg task: {avg_duration_ms:.6f} ms"

            if chunk_tasks_str:
                print(
                    f"  Chunk {chunk_id} [{chunk_tasks_str}]: Total: {total_duration_cycles:.2f} cycles ({total_duration_ms:.6f} ms){avg_info}, {chunk_percentage:.2f}%"
                )
            else:
                print(
                    f"  Chunk {chunk_id}: Total: {total_duration_cycles:.2f} cycles ({total_duration_ms:.6f} ms){avg_info}, {chunk_percentage:.2f}%"
                )
    else:
        print(
            f"No chunk execution data found in the selected region for Schedule {schedule_index+1}."
        )


def process_schedule(section, schedule_index, args):
    """Process a single schedule section."""
    print(f"\n=== Processing Schedule {schedule_index+1} ===")

    # Parse the task data
    frequency = extract_frequency(section)
    print(f"Detected frequency: {frequency} Hz")

    # Extract Schedule_UID
    schedule_uid = extract_schedule_uid(section)
    if schedule_uid:
        print(f"Detected Schedule_UID: {schedule_uid}")

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

    # Empty chunk_types for compatibility
    chunk_types = {}

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

    # Create and save Gantt chart
    chart_created = create_gantt_chart(
        chunks_data,
        chunk_names,
        time_range,
        chunk_avg_durations_ms,
        chunk_total_durations,
        CYCLES_TO_MS,
        schedule_index,
        schedule_uid,
        args,
        args.output_dir,
    )

    if chart_created:
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


def main():
    """Main function to process all schedules in the input file."""
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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
