import os
import re
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Path to the input file containing task timing data")
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

    # Match chunk line
    chunk_match = re.match(r"Chunk (\d+):", line)
    if chunk_match:
        current_chunk = int(chunk_match.group(1))
        tasks[current_task][current_chunk] = {}
        continue

    # Match time data
    start_match = re.match(r"Start: (\d+) us", line)
    if start_match:
        tasks[current_task][current_chunk]["start"] = int(start_match.group(1))
        continue

    end_match = re.match(r"End: (\d+) us", line)
    if end_match:
        tasks[current_task][current_chunk]["end"] = int(end_match.group(1))
        continue

# Find the maximum chunk ID in the data
max_chunk_id = max(
    [chunk_id for task_data in tasks.values() for chunk_id in task_data.keys()]
)
print(f"Detected {max_chunk_id + 1} chunks in the data")

# Adjust timescale: subtract the minimum time to make visualization easier
min_time = min(
    [
        chunk_data["start"]
        for task_data in tasks.values()
        for chunk_data in task_data.values()
    ]
)
max_time = max(
    [
        chunk_data["end"]
        for task_data in tasks.values()
        for chunk_data in task_data.values()
    ]
)
time_range = max_time - min_time

for task_id in tasks:
    for chunk_id in tasks[task_id]:
        tasks[task_id][chunk_id]["start"] = (
            tasks[task_id][chunk_id]["start"] - min_time
        ) / 1000  # convert to ms
        tasks[task_id][chunk_id]["end"] = (
            tasks[task_id][chunk_id]["end"] - min_time
        ) / 1000  # convert to ms
        tasks[task_id][chunk_id]["duration"] = (
            tasks[task_id][chunk_id]["end"] - tasks[task_id][chunk_id]["start"]
        )

# Sort task IDs numerically
sorted_task_ids = sorted(tasks.keys())

# Colors for different chunks - dynamically generate based on max_chunk_id
color_map = plt.cm.get_cmap("tab10", max_chunk_id + 1)
chunk_colors = [color_map(i) for i in range(max_chunk_id + 1)]

# Create a wider Gantt chart
fig, ax = plt.subplots(figsize=(30, 10))  # Increased width to 30 inches

# Draw the tasks
y_ticks = []
y_labels = []

for i, task_id in enumerate(sorted_task_ids):
    y_pos = i
    y_ticks.append(y_pos)
    y_labels.append(f"Task {task_id}")

    for chunk_id in range(
        max_chunk_id + 1
    ):  # Use detected max chunk ID instead of hardcoded 4
        if chunk_id in tasks[task_id]:
            chunk_data = tasks[task_id][chunk_id]
            start = chunk_data["start"]
            duration = chunk_data["duration"]

            # Draw the bar
            ax.barh(
                y_pos,
                duration,
                left=start,
                height=0.5,
                color=chunk_colors[chunk_id],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # Add text label showing the chunk number
            text_x = start + duration / 2
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
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_title("Task Execution Timeline (Wide View)", fontsize=14)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)

# Add legend for chunks
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=chunk_colors[i], label=f"Chunk {i}")
    for i in range(max_chunk_id + 1)
]
ax.legend(handles=legend_elements, loc="upper right")

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "task_execution_timeline_wide.png"))
plt.close()

print(
    f"Wide chart saved to {os.path.join(base_dir, 'task_execution_timeline_wide.png')}"
)
