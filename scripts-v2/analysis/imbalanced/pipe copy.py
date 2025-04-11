import os
import re
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, "in.txt"), "r") as f:
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

# Create a wider chunk-centric view
fig, ax = plt.subplots(figsize=(30, 10))  # Increased width to 30 inches

# Create a chunk-centric view
chunk_tasks = {i: [] for i in range(max_chunk_id + 1)}

for task_id in sorted_task_ids:
    for chunk_id in range(max_chunk_id + 1):
        if chunk_id in tasks[task_id]:
            chunk_tasks[chunk_id].append(
                {
                    "task_id": task_id,
                    "start": tasks[task_id][chunk_id]["start"],
                    "end": tasks[task_id][chunk_id]["end"],
                    "duration": tasks[task_id][chunk_id]["duration"],
                }
            )

# Sort tasks within each chunk by start time
for chunk_id in chunk_tasks:
    chunk_tasks[chunk_id].sort(key=lambda x: x["start"])

# Calculate total number of tasks across all chunks for proper spacing
total_tasks = sum(len(tasks_list) for tasks_list in chunk_tasks.values())
chunk_heights = {i: len(chunk_tasks[i]) for i in range(max_chunk_id + 1)}
max_chunk_height = max(chunk_heights.values()) if chunk_heights else 0

# Determine fixed spacing between chunks
chunk_spacing = 2
section_height = max_chunk_height + chunk_spacing

# Draw the tasks by chunk
chunk_positions = []
y_ticks = []
y_labels = []

for chunk_id in range(max_chunk_id + 1):
    # Calculate the base position for this chunk
    base_y_pos = chunk_id * section_height
    chunk_positions.append(base_y_pos)
    
    # Add chunk label
    y_ticks.append(base_y_pos + chunk_heights[chunk_id] / 2)
    y_labels.append(f"Chunk {chunk_id}")
    
    # Draw horizontal line to separate chunks
    if chunk_id > 0:
        ax.axhline(base_y_pos - chunk_spacing/2, color='gray', linestyle='-', alpha=0.3)
    
    # Add tasks for this chunk
    for i, task_data in enumerate(chunk_tasks[chunk_id]):
        bar_pos = base_y_pos + i
        
        # Draw task bar
        ax.barh(
            bar_pos,
            task_data["duration"],
            left=task_data["start"],
            height=0.6,
            color=chunk_colors[chunk_id],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        
        # Add text label
        text_x = task_data["start"] + task_data["duration"] / 2
        ax.text(
            text_x,
            bar_pos,
            f"T{task_data['task_id']}",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

# Set chart properties
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_title("Chunk Execution Timeline (Grouped by Chunk, Wide View)", fontsize=14)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)

# Adjust y-axis limits to provide some padding
max_y = (max_chunk_id + 1) * section_height
ax.set_ylim(-1, max_y)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chunk_execution_timeline_wide.png"))
plt.close()

print(
    f"Chunk-centric wide chart saved to {os.path.join(base_dir, 'chunk_execution_timeline_wide.png')}"
)

# Create a CPU pipeline style diagram with overlapping tasks
fig, ax = plt.subplots(figsize=(30, 12))  # Wider and taller

# Create task rows - one row per task
for i, task_id in enumerate(sorted_task_ids):
    y_pos = len(sorted_task_ids) - i - 1  # Reverse order so Task 0 is at the bottom

    # Add task label on the left
    ax.text(
        -5,
        y_pos,
        f"Task {task_id}",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Draw each chunk as a colored rectangle
    for chunk_id in range(max_chunk_id + 1):
        if chunk_id in tasks[task_id]:
            chunk_data = tasks[task_id][chunk_id]
            start = chunk_data["start"]
            end = chunk_data["end"]
            duration = chunk_data["duration"]

            # Draw rectangle
            rect = plt.Rectangle(
                (start, y_pos - 0.3),
                duration,
                0.6,
                color=chunk_colors[chunk_id],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Add text label inside the rectangle
            text_x = start + duration / 2
            ax.text(
                text_x,
                y_pos,
                f"C{chunk_id}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=9,
            )

# Set chart properties
ax.set_yticks(range(len(sorted_task_ids)))
ax.set_yticklabels([])  # Hide the default y-ticks
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_title("CPU Pipeline Style Visualization", fontsize=14)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)
ax.set_xlim(-10, (max_time - min_time) / 1000 + 10)  # Add some padding on both sides
ax.set_ylim(-1, len(sorted_task_ids))

# Add legend for chunks
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=chunk_colors[i], label=f"Chunk {i}")
    for i in range(max_chunk_id + 1)
]
ax.legend(handles=legend_elements, loc="upper right")

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "pipeline_visualization.png"))
plt.close()

print(
    f"Pipeline visualization saved to {os.path.join(base_dir, 'pipeline_visualization.png')}"
)

# Create a more compact overlapping visualization
# This visualization places all chunks of the same type in the same row
fig, ax = plt.subplots(figsize=(30, 8))

# Create a row for each chunk type
for chunk_id in range(max_chunk_id + 1):
    y_pos = max_chunk_id - chunk_id  # Position chunks with 0 at the top

    # Add chunk label on the left
    ax.text(
        -5,
        y_pos,
        f"Chunk {chunk_id}",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Draw each task's instance of this chunk type
    for task_id in sorted_task_ids:
        if chunk_id in tasks[task_id]:
            chunk_data = tasks[task_id][chunk_id]
            start = chunk_data["start"]
            duration = chunk_data["duration"]

            # Draw rectangle with unique color per task
            task_color = plt.cm.tab20(task_id % 20)
            rect = plt.Rectangle(
                (start, y_pos - 0.4),
                duration,
                0.8,
                color=task_color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(rect)

            # Add text label inside the rectangle
            text_x = start + duration / 2
            ax.text(
                text_x,
                y_pos,
                f"T{task_id}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=9,
            )

# Set chart properties
ax.set_yticks(range(max_chunk_id + 1))
ax.set_yticklabels([f"Chunk {max_chunk_id-i}" for i in range(max_chunk_id + 1)])
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_title("Overlapping Chunks Visualization (Textbook Style)", fontsize=14)
ax.grid(True, axis="x", linestyle="--", alpha=0.7)
ax.set_xlim(-10, (max_time - min_time) / 1000 + 10)  # Add some padding on both sides

# Create custom legend for tasks
task_legend_elements = [
    plt.Rectangle(
        (0, 0), 1, 1, color=plt.cm.tab20(task_id % 20), label=f"Task {task_id}"
    )
    for task_id in sorted_task_ids
]

# Use a legendHandler with 5 columns to make the legend more compact
legend = ax.legend(
    handles=task_legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
)

# Save the figure
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend
plt.savefig(os.path.join(base_dir, "textbook_style_pipeline.png"))
plt.close()

print(
    f"Textbook-style pipeline visualization saved to {os.path.join(base_dir, 'textbook_style_pipeline.png')}"
)
