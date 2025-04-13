import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

base_dir = os.path.dirname(os.path.abspath(__file__))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file", help="Path to the input file containing task timing data"
)
parser.add_argument(
    "--output",
    "-o",
    help="Output filename (with or without extension)",
    default="chunk_execution_timeline",
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

with open(args.input_file, "r") as f:
    lines = f.readlines()

# Parse the input data
tasks = {}
current_task = None
current_chunk = None

for line in lines:
    line = line.strip()

    # Match task line
    task_match = re.match(r"Task (\d+):", line)
    if task_match:
        current_task = int(task_match.group(1))
        tasks[current_task] = {}
        continue

    # Match chunk line with type (e.g., "Chunk 0 (Big):")
    chunk_match = re.match(r"Chunk (\d+) \(([^)]+)\):", line)
    if chunk_match:
        current_chunk = int(chunk_match.group(1))
        chunk_type = chunk_match.group(2)
        tasks[current_task][current_chunk] = {"type": chunk_type}
        continue

    # Match time data with cycles
    start_match = re.match(r"Start: (\d+) cycles", line)
    if start_match:
        tasks[current_task][current_chunk]["start"] = int(start_match.group(1))
        continue

    end_match = re.match(r"End: (\d+) cycles", line)
    if end_match:
        tasks[current_task][current_chunk]["end"] = int(end_match.group(1))
        continue

    # Match duration in cycles
    duration_match = re.match(r"Duration: (\d+) cycles", line)
    if duration_match:
        tasks[current_task][current_chunk]["duration_cycles"] = int(
            duration_match.group(1)
        )
        continue

# Find the maximum chunk ID in the data
max_chunk_id = max(
    [chunk_id for task_data in tasks.values() for chunk_id in task_data.keys()]
)
print(f"Detected {max_chunk_id + 1} chunks in the data")

# Collect all chunk types for the legend
chunk_types = {}
for task_id in tasks:
    for chunk_id in tasks[task_id]:
        if chunk_id not in chunk_types and "type" in tasks[task_id][chunk_id]:
            chunk_types[chunk_id] = tasks[task_id][chunk_id]["type"]

# Adjust timescale: subtract the minimum time to make visualization easier
min_time = min(
    [
        chunk_data["start"]
        for task_data in tasks.values()
        for chunk_data in task_data.values()
        if "start" in chunk_data
    ]
)

# Find the maximum end time for percentage calculation
max_time = max(
    [
        chunk_data["end"]
        for task_data in tasks.values()
        for chunk_data in task_data.values()
        if "end" in chunk_data
    ]
)

# Calculate total time span
time_span = max_time - min_time

# Calculate the actual start and end times based on percentages
filter_start_time = min_time + (args.start_time * time_span)
filter_end_time = min_time + (args.end_time * time_span)

print(f"Total time span: {time_span} cycles")
print(f"Filtering time range: {filter_start_time} to {filter_end_time} cycles")

# Calculate durations and normalize time
for task_id in tasks:
    for chunk_id in tasks[task_id]:
        if "start" in tasks[task_id][chunk_id] and "end" in tasks[task_id][chunk_id]:
            # Calculate relative time from start (in cycles)
            tasks[task_id][chunk_id]["start_norm"] = (
                tasks[task_id][chunk_id]["start"] - min_time
            )
            tasks[task_id][chunk_id]["end_norm"] = (
                tasks[task_id][chunk_id]["end"] - min_time
            )

            # Use provided duration if available, otherwise calculate it
            if "duration_cycles" in tasks[task_id][chunk_id]:
                duration_cycles = tasks[task_id][chunk_id]["duration_cycles"]
            else:
                duration_cycles = (
                    tasks[task_id][chunk_id]["end"] - tasks[task_id][chunk_id]["start"]
                )

            # Store the duration in cycles
            tasks[task_id][chunk_id]["duration_cycles"] = duration_cycles

# Frequency: 24576000 Hz
# 1 cycle = 1e6 / 24576000 us
# 1 cycle = 1e3 / 24576000 ms
CYCLES_TO_MS = 1e3 / 24576000.0  # Convert cycles to milliseconds

# Create a dictionary to organize tasks by chunk
chunks_data = {chunk_id: [] for chunk_id in range(max_chunk_id + 1)}

# Data structures for analysis
tasks_in_region = set()
task_durations_by_chunk = defaultdict(lambda: defaultdict(list))
chunk_total_durations = defaultdict(float)

# Fill in the chunks_data dictionary and filter by time range
for task_id in tasks:
    for chunk_id in tasks[task_id]:
        if (
            "start" in tasks[task_id][chunk_id]
            and "duration_cycles" in tasks[task_id][chunk_id]
        ):
            # Only include chunks that overlap with our time range
            start_time = tasks[task_id][chunk_id]["start"]
            end_time = tasks[task_id][chunk_id]["end"]

            # Skip chunks entirely outside our time range
            if end_time < filter_start_time or start_time > filter_end_time:
                continue

            # Track tasks in this region
            tasks_in_region.add(task_id)

            # Track durations for each task in each chunk
            duration_cycles = tasks[task_id][chunk_id]["duration_cycles"]
            task_durations_by_chunk[chunk_id][task_id].append(duration_cycles)

            # Track total duration for each chunk
            chunk_total_durations[chunk_id] += duration_cycles

            chunks_data[chunk_id].append(
                {
                    "task_id": task_id,
                    "start_ms": (tasks[task_id][chunk_id]["start_norm"]) * CYCLES_TO_MS,
                    "duration_ms": tasks[task_id][chunk_id]["duration_cycles"]
                    * CYCLES_TO_MS,
                    "type": tasks[task_id][chunk_id].get("type", ""),
                }
            )

# Sort chunk data by start time
for chunk_id in chunks_data:
    chunks_data[chunk_id].sort(key=lambda x: x["start_ms"])

# Calculate display range in milliseconds
display_start_ms = (filter_start_time - min_time) * CYCLES_TO_MS
display_end_ms = (filter_end_time - min_time) * CYCLES_TO_MS

# Create a wider Gantt chart
fig, ax = plt.subplots(figsize=(30, 8))  # Increased width to 30 inches

# Task colors - use a color map for different tasks
task_color_map = plt.colormaps["tab20"]

# Define chunk names for y-axis labels based on their types
chunk_names = [
    f"Chunk {i} ({chunk_types.get(i, 'Unknown')})" for i in range(max_chunk_id + 1)
]

# Draw the chunks
y_ticks = []
y_labels = []

for i, chunk_id in enumerate(range(max_chunk_id + 1)):
    y_pos = max_chunk_id - i  # Reverse order to have Chunk 0 at the bottom
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

# Set chart properties
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xlabel("Time (ms, converted from cycles)", fontsize=12)
ax.set_title(
    f"Chunk Execution Timeline ({args.start_time*100:.0f}%-{args.end_time*100:.0f}% of execution)",
    fontsize=14,
)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)

# Set the x-axis limits to our desired time range
ax.set_xlim(display_start_ms, display_end_ms)

# Save the figure with specified output name
output_filename = args.output
if not output_filename.lower().endswith(".png"):
    output_filename += ".png"

# Check if output path has a directory component
if os.path.dirname(output_filename):
    # Use the specified path directly
    output_path = output_filename
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
else:
    # Only a filename was given, put it in the base_dir
    output_path = os.path.join(base_dir, output_filename)

plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Chunk execution timeline saved to {output_path}")
print(f"Time range displayed: {display_start_ms:.2f}ms to {display_end_ms:.2f}ms")

# ----- ANALYSIS OUTPUT -----

# 1. Print all tasks in the region
print("\n======= TASKS IN SELECTED REGION =======")
sorted_tasks = sorted(tasks_in_region)
print(
    f"Found {len(sorted_tasks)} tasks in the region {args.start_time:.2f}-{args.end_time:.2f}"
)
print(f"Tasks: {', '.join(map(str, sorted_tasks))}")

# 2. Calculate and print average duration for each task in each chunk
print("\n======= AVERAGE TASK DURATIONS BY CHUNK =======")
for chunk_id in range(max_chunk_id + 1):
    if chunk_id not in task_durations_by_chunk:
        continue

    chunk_type = chunk_types.get(chunk_id, "Unknown")
    print(f"Chunk {chunk_id} ({chunk_type}):")

    # Calculate average duration for each task
    for task_id, durations in sorted(task_durations_by_chunk[chunk_id].items()):
        avg_duration_cycles = sum(durations) / len(durations)
        avg_duration_ms = avg_duration_cycles * CYCLES_TO_MS
        print(
            f"  Task {task_id}: {avg_duration_cycles:.2f} cycles ({avg_duration_ms:.6f} ms)"
        )

    # Print average across all tasks for this chunk
    all_durations = [
        d
        for task_durations in task_durations_by_chunk[chunk_id].values()
        for d in task_durations
    ]
    if all_durations:
        avg_all_tasks_cycles = sum(all_durations) / len(all_durations)
        avg_all_tasks_ms = avg_all_tasks_cycles * CYCLES_TO_MS
        print(
            f"  All Tasks Average: {avg_all_tasks_cycles:.2f} cycles ({avg_all_tasks_ms:.6f} ms)"
        )
    print()

# 3. Find the widest chunk (most execution time) in the region
print("\n======= WIDEST CHUNK IN REGION =======")
if chunk_total_durations:
    widest_chunk_id = max(chunk_total_durations, key=chunk_total_durations.get)
    widest_chunk_duration_cycles = chunk_total_durations[widest_chunk_id]
    widest_chunk_duration_ms = widest_chunk_duration_cycles * CYCLES_TO_MS
    widest_chunk_type = chunk_types.get(widest_chunk_id, "Unknown")

    print(f"Widest chunk: Chunk {widest_chunk_id} ({widest_chunk_type})")
    print(
        f"Total execution time: {widest_chunk_duration_cycles:.2f} cycles ({widest_chunk_duration_ms:.6f} ms)"
    )

    # Calculate percentage of time spent in this chunk
    total_execution_cycles = sum(chunk_total_durations.values())
    percentage = (widest_chunk_duration_cycles / total_execution_cycles) * 100
    print(f"Percentage of selected region execution time: {percentage:.2f}%")

    # Show execution breakdown for all chunks
    print("\nExecution time breakdown for all chunks in region:")
    for chunk_id, duration in sorted(
        chunk_total_durations.items(), key=lambda x: x[1], reverse=True
    ):
        chunk_type = chunk_types.get(chunk_id, "Unknown")
        duration_ms = duration * CYCLES_TO_MS
        chunk_percentage = (duration / total_execution_cycles) * 100
        print(
            f"  Chunk {chunk_id} ({chunk_type}): {duration:.2f} cycles ({duration_ms:.6f} ms), {chunk_percentage:.2f}%"
        )
else:
    print("No chunk execution data found in the selected region.")
