import os
import re
import matplotlib.pyplot as plt
import numpy as np

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
    default="task_execution_timeline_wide",
)
parser.add_argument(
    "--task-start",
    type=int,
    help="Starting task ID to include (inclusive)",
    default=None,
)
parser.add_argument(
    "--task-end", type=int, help="Ending task ID to include (inclusive)", default=None
)
args = parser.parse_args()

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

# Filter tasks based on command-line arguments
if args.task_start is not None or args.task_end is not None:
    start_task = args.task_start if args.task_start is not None else min(tasks.keys())
    end_task = args.task_end if args.task_end is not None else max(tasks.keys())

    tasks = {
        task_id: task_data
        for task_id, task_data in tasks.items()
        if start_task <= task_id <= end_task
    }

    print(f"Filtered to tasks {start_task} through {end_task}")

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

# Sort task IDs numerically
sorted_task_ids = sorted(tasks.keys())

# Colors for different chunks based on their type, not just ID
chunk_type_colors = {"Big": "red", "GPU": "green", "Medium": "blue", "Little": "purple"}

# Fallback to a color map if a type doesn't have a predefined color
color_map = plt.colormaps["tab10"]
chunk_colors = {}
for i in range(max_chunk_id + 1):
    if i in chunk_types:
        chunk_type = chunk_types[i]
        chunk_colors[i] = chunk_type_colors.get(chunk_type, color_map(i % 10))
    else:
        chunk_colors[i] = color_map(i % 10)

# Create a wider Gantt chart
fig, ax = plt.subplots(figsize=(30, 10))  # Increased width to 30 inches

# Draw the tasks
y_ticks = []
y_labels = []

# # Convert cycles to ms for better visualization (assuming 1 cycle = 1ns for display)
# # This scale factor can be adjusted based on the actual clock frequency
# CYCLES_TO_MS = 1.0 / 1000000.0  # 1M cycles = 1ms (rough approximation)

# Frequency: 24576000 Hz
# 1 cycle = 1e6 / 24576000 us
# 1 cycle = 1e3 / 24576000 ms
CYCLES_TO_MS = 1.0 / 24576000.0  # 1M cycles = 1ms (rough approximation)

for i, task_id in enumerate(sorted_task_ids):
    y_pos = i
    y_ticks.append(y_pos)
    y_labels.append(f"Task {task_id}")

    for chunk_id in range(max_chunk_id + 1):
        if chunk_id in tasks[task_id] and "duration_cycles" in tasks[task_id][chunk_id]:
            chunk_data = tasks[task_id][chunk_id]
            start_ms = chunk_data["start_norm"] * CYCLES_TO_MS
            duration_ms = chunk_data["duration_cycles"] * CYCLES_TO_MS

            # Draw the bar
            bar = ax.barh(
                y_pos,
                duration_ms,
                left=start_ms,
                height=0.5,
                color=chunk_colors[chunk_id],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # Add text label showing the chunk number and type
            text_x = start_ms + duration_ms / 2
            chunk_type = chunk_data.get("type", "")
            ax.text(
                text_x,
                y_pos,
                f"C{chunk_id}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

# Set chart properties
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xlabel("Time (ms, converted from cycles)", fontsize=12)
ax.set_title("Task Execution Timeline (Wide View)", fontsize=14)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)

# Add legend for chunks with their types
legend_elements = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        color=chunk_colors[i],
        label=f"Chunk {i} ({chunk_types.get(i, 'Unknown')})",
    )
    for i in range(max_chunk_id + 1)
    if i in chunk_types
]
ax.legend(handles=legend_elements, loc="upper right")

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

print(f"Wide chart saved to {output_path}")
