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
    parser.add_argument(
        "--schedules",
        type=str,
        help="Specify which schedules to show (e.g., '1', '1-5', '1,3,5'). Use -1 for all schedules.",
        default="-1",
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


def parse_schedule_selection(selection_str, total_schedules):
    """Parse the schedule selection string."""
    if selection_str == "-1":
        # Return all schedules
        return list(range(total_schedules))

    selected_schedules = set()

    # Split by comma
    parts = selection_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range of schedules
            try:
                start, end = map(int, part.split("-"))
                # Convert to 0-based indexing
                start_idx = start - 1
                end_idx = end - 1

                if start_idx < 0 or end_idx >= total_schedules or start_idx > end_idx:
                    print(f"Warning: Invalid range '{part}'. Using valid portion.")
                    start_idx = max(0, start_idx)
                    end_idx = min(total_schedules - 1, end_idx)

                selected_schedules.update(range(start_idx, end_idx + 1))
            except ValueError:
                print(f"Warning: Could not parse range '{part}'. Skipping.")
        else:
            # Single schedule
            try:
                idx = int(part) - 1  # Convert to 0-based indexing
                if 0 <= idx < total_schedules:
                    selected_schedules.add(idx)
                else:
                    print(f"Warning: Schedule {part} is out of range. Skipping.")
            except ValueError:
                print(f"Warning: Could not parse schedule number '{part}'. Skipping.")

    return sorted(list(selected_schedules))


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


def print_schedule_header(
    schedule_index, frequency, schedule_annotations, max_chunk_id
):
    """Print a clean header for the schedule."""
    schedule_number = schedule_index + 1

    # Create a box around the schedule number
    header_width = 70
    box_line = "+" + "-" * (header_width - 2) + "+"

    print(box_line)
    print(f"| SCHEDULE {schedule_number:<61} |")
    print(box_line)

    # Print frequency
    print(f"| Frequency: {frequency/1e6:.2f} MHz{' ' * (header_width - 24)} |")

    # Print cores and their chunks
    print(f"| Core Assignments:{' ' * (header_width - 18)} |")
    for core_type, chunks in sorted(schedule_annotations.items()):
        chunk_str = ", ".join(map(str, chunks))
        print(f"|   {core_type}: {chunk_str:<{header_width - 7 - len(core_type)}} |")

    # Print number of chunks
    print(f"| Total Chunks: {max_chunk_id + 1:<{header_width - 16}} |")
    print(box_line)


def print_task_summary(tasks_in_region, args):
    """Print a summary of tasks in the selected region."""
    sorted_tasks = sorted(tasks_in_region)
    num_tasks = len(sorted_tasks)

    print(f"Tasks in region {args.start_time:.2f}-{args.end_time:.2f}: {num_tasks}")

    # Format task IDs in rows
    if sorted_tasks:
        task_str = ""
        for i, task_id in enumerate(sorted_tasks):
            task_str += f"{task_id:4}"
            if (i + 1) % 15 == 0:  # 15 tasks per line
                print(f"  {task_str}")
                task_str = ""
        if task_str:  # Print any remaining tasks
            print(f"  {task_str}")


def print_chunk_summary(
    chunk_total_durations,
    chunk_avg_durations_ms,
    chunk_types,
    schedule_annotations,
    CYCLES_TO_MS,
):
    """Print a summary of chunk performance."""
    if not chunk_total_durations:
        print("No data available for this schedule.")
        return

    # Calculate total execution time
    total_execution_cycles = sum(chunk_total_durations.values())
    total_execution_ms = total_execution_cycles * CYCLES_TO_MS

    # Print total execution time
    print(f"\nTotal Execution Time: {total_execution_ms:.6f} ms")

    # Print chunk execution times sorted by duration (descending)
    print("\n  CHUNK EXECUTION BREAKDOWN")
    print("  +-------------------------------------------------+")
    print("  | Chunk | Processor | Core   | Duration (ms) | %  |")
    print("  +-------------------------------------------------+")

    for chunk_id, duration in sorted(
        chunk_total_durations.items(), key=lambda x: x[1], reverse=True
    ):
        # Find core type
        c_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if chunk_id in chunks:
                c_type = core
                break

        p_type = chunk_types.get(chunk_id, "Unknown")
        duration_ms = duration * CYCLES_TO_MS
        percentage = (duration / total_execution_cycles) * 100

        # Add average duration if available
        avg_ms = chunk_avg_durations_ms.get(chunk_id, 0)

        print(
            f"  | {chunk_id:5} | {p_type:8} | {c_type:6} | {duration_ms:11.6f} | {percentage:3.0f} |"
        )

    print("  +-------------------------------------------------+")

    # Group by processor type
    print("\n  PROCESSOR TYPE SUMMARY")
    print("  +-------------------------------------+")
    print("  | Processor | Duration (ms) | % Total |")
    print("  +-------------------------------------+")

    processor_totals = defaultdict(float)
    for chunk_id, duration in chunk_total_durations.items():
        p_type = chunk_types.get(chunk_id, "Unknown")
        processor_totals[p_type] += duration

    for p_type, duration in sorted(
        processor_totals.items(), key=lambda x: x[1], reverse=True
    ):
        duration_ms = duration * CYCLES_TO_MS
        percentage = (duration / total_execution_cycles) * 100
        print(f"  | {p_type:9} | {duration_ms:11.6f} | {percentage:7.1f} |")

    print("  +-------------------------------------+")

    # Group by core type
    print("\n  CORE TYPE SUMMARY")
    print("  +----------------------------------+")
    print("  | Core    | Duration (ms) | % Total |")
    print("  +----------------------------------+")

    core_totals = defaultdict(float)
    for chunk_id, duration in chunk_total_durations.items():
        c_type = "Unknown"
        for core, chunks in schedule_annotations.items():
            if chunk_id in chunks:
                c_type = core
                break
        core_totals[c_type] += duration

    for c_type, duration in sorted(
        core_totals.items(), key=lambda x: x[1], reverse=True
    ):
        duration_ms = duration * CYCLES_TO_MS
        percentage = (duration / total_execution_cycles) * 100
        print(f"  | {c_type:7} | {duration_ms:11.6f} | {percentage:7.1f} |")

    print("  +----------------------------------+")


def calculate_gapness_metrics(chunks_data, time_range, CYCLES_TO_MS):
    """Calculate metrics for 'gapness' in the schedule execution."""
    # First, organize data by chunk
    chunk_timelines = {}
    for chunk_id, tasks in chunks_data.items():
        if tasks:
            # Convert to sorted list of (start_ms, end_ms) pairs
            task_intervals = [
                (t["start_ms"], t["start_ms"] + t["duration_ms"]) for t in tasks
            ]
            task_intervals.sort()
            chunk_timelines[chunk_id] = task_intervals

    # Calculate metrics for each chunk
    chunk_metrics = {}
    for chunk_id, intervals in chunk_timelines.items():
        if not intervals:
            continue

        # Calculate total time span for this chunk
        chunk_start = intervals[0][0]
        chunk_end = intervals[-1][1]
        chunk_span = chunk_end - chunk_start

        # Calculate total execution time (sum of all task durations)
        execution_time = sum(end - start for start, end in intervals)

        # Calculate total gap time
        last_end = intervals[0][0]
        gap_time = 0

        for start, end in intervals:
            if start > last_end:
                gap_time += start - last_end
            last_end = max(last_end, end)

        # Calculate metrics
        if chunk_span > 0:
            gap_percentage = (gap_time / chunk_span) * 100
            utilization = (execution_time / chunk_span) * 100
        else:
            gap_percentage = 0
            utilization = 100

        # Calculate average gap size
        num_gaps = 0
        avg_gap = 0
        last_end = intervals[0][1]

        for i in range(1, len(intervals)):
            start, _ = intervals[i]
            if start > last_end:
                num_gaps += 1
                avg_gap += start - last_end
            last_end = max(last_end, intervals[i][1])

        if num_gaps > 0:
            avg_gap /= num_gaps

        # Store metrics
        chunk_metrics[chunk_id] = {
            "span_ms": chunk_span,
            "execution_ms": execution_time,
            "gap_ms": gap_time,
            "gap_percentage": gap_percentage,
            "utilization": utilization,
            "num_gaps": num_gaps,
            "avg_gap_ms": avg_gap,
        }

    # Calculate overall schedule metrics
    if chunk_metrics:
        overall_span = 0
        overall_execution = 0
        overall_gap = 0

        # Find global start and end times across all chunks
        all_starts = []
        all_ends = []

        for chunk_id, intervals in chunk_timelines.items():
            if intervals:
                all_starts.append(intervals[0][0])
                all_ends.append(intervals[-1][1])

        if all_starts and all_ends:
            global_start = min(all_starts)
            global_end = max(all_ends)
            overall_span = global_end - global_start

            # Calculate overall execution and gap time
            for chunk_id, metrics in chunk_metrics.items():
                overall_execution += metrics["execution_ms"]
                overall_gap += metrics["gap_ms"]

            # Adjust for parallel execution in different chunks
            overall_execution = min(overall_execution, overall_span)

            # Calculate "density" - higher is better (less gaps)
            schedule_density = (
                overall_execution / overall_span if overall_span > 0 else 1.0
            )

            # Calculate cross-chunk gaps (overlaps between chunks)
            # This is a measure of how well the chunks are utilizing the overall timeline
            merged_timeline = []
            for chunk_id, intervals in chunk_timelines.items():
                merged_timeline.extend(intervals)

            merged_timeline.sort()
            merged_execution = 0
            last_end = merged_timeline[0][0]

            for start, end in merged_timeline:
                if start > last_end:
                    # There's a gap
                    pass
                # Extend the current interval if needed
                merged_execution += max(0, end - max(start, last_end))
                last_end = max(last_end, end)

            cross_chunk_utilization = (
                merged_execution / (last_end - merged_timeline[0][0])
                if (last_end - merged_timeline[0][0]) > 0
                else 1.0
            )

            return chunk_metrics, {
                "span_ms": overall_span,
                "execution_ms": overall_execution,
                "gap_ms": overall_gap,
                "density": schedule_density * 100,  # Convert to percentage
                "cross_chunk_utilization": cross_chunk_utilization
                * 100,  # Convert to percentage
            }

    return chunk_metrics, {}


def print_gapness_metrics(chunk_metrics, overall_metrics):
    """Print the gapness metrics for chunks and the overall schedule."""
    if not overall_metrics:
        print("No gapness metrics available.")
        return

    # Print overall metrics first
    print("\n  SCHEDULE GAPNESS METRICS")
    print("  +--------------------------------------------------+")
    print("  | Metric                    | Value                |")
    print("  +--------------------------------------------------+")
    print(
        f"  | Total Time Span           | {overall_metrics['span_ms']:.6f} ms       |"
    )
    print(
        f"  | Schedule Density          | {overall_metrics['density']:.2f}%                |"
    )
    print(
        f"  | Cross-Chunk Utilization   | {overall_metrics['cross_chunk_utilization']:.2f}%                |"
    )
    print("  +--------------------------------------------------+")
    print("  | Higher density/utilization values indicate fewer gaps |")
    print("  +--------------------------------------------------+")

    # Print chunk metrics
    print("\n  CHUNK GAPNESS METRICS")
    print(
        "  +-------------------------------------------------------------------------+"
    )
    print(
        "  | Chunk | Time Span (ms) | Utilization (%) | Gaps | Avg Gap (ms) | Gap % |"
    )
    print(
        "  +-------------------------------------------------------------------------+"
    )

    # Sort chunks by utilization (ascending - worst first)
    for chunk_id, metrics in sorted(
        chunk_metrics.items(), key=lambda x: x[1]["utilization"]
    ):
        print(
            f"  | {chunk_id:5} | {metrics['span_ms']:13.6f} | {metrics['utilization']:14.2f} | {metrics['num_gaps']:4} | "
            f"{metrics['avg_gap_ms']:11.6f} | {metrics['gap_percentage']:5.1f} |"
        )

    print(
        "  +-------------------------------------------------------------------------+"
    )


def process_schedule(section, schedule_index, args):
    """Process a single schedule section."""
    # Parse the task data
    frequency = extract_frequency(section)
    schedule_annotations = extract_schedule_annotations(section)

    tasks = parse_task_data(section)
    if not tasks:
        print(f"No task data found in Schedule {schedule_index+1}, skipping.")
        return

    # Find the maximum chunk ID in the data
    max_chunk_id = max(
        [chunk_id for task_data in tasks.values() for chunk_id in task_data.keys()]
    )

    # Print the schedule header
    print_schedule_header(schedule_index, frequency, schedule_annotations, max_chunk_id)

    # Collect all chunk types
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

    # Calculate gapness metrics
    chunk_gapness, overall_gapness = calculate_gapness_metrics(
        chunks_data, time_range, CYCLES_TO_MS
    )

    # Print task summary
    print_task_summary(tasks_in_region, args)

    # Print chunk summary
    print_chunk_summary(
        chunk_total_durations,
        chunk_avg_durations_ms,
        chunk_types,
        schedule_annotations,
        CYCLES_TO_MS,
    )

    # Print gapness metrics
    print_gapness_metrics(chunk_gapness, overall_gapness)

    # Print a separator
    print("\n" + "=" * 72)


def main():
    """Main function to process all schedules in the input file."""
    args = parse_arguments()

    # Read the input file
    with open(args.input_file, "r") as f:
        content = f.read()

    # Extract Python sections
    python_sections = extract_python_sections(content)

    # Parse the schedules to display
    selected_schedules = parse_schedule_selection(args.schedules, len(python_sections))

    if not selected_schedules:
        print("No valid schedules selected. Exiting.")
        return

    # Show how many schedules were selected
    print(
        f"Processing {len(selected_schedules)} out of {len(python_sections)} schedules."
    )

    # Process each selected schedule
    for idx in selected_schedules:
        process_schedule(python_sections[idx], idx, args)

    print(f"\nFinished processing {len(selected_schedules)} schedule(s).")


if __name__ == "__main__":
    main()
