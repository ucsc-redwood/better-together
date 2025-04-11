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

# # Create a wider chunk-centric view
# fig, ax = plt.subplots(figsize=(30, 10))  # Increased width to 30 inches

# # Create a chunk-centric view
# chunk_tasks = {i: [] for i in range(max_chunk_id + 1)}

# for task_id in sorted_task_ids:
#     for chunk_id in range(max_chunk_id + 1):
#         if chunk_id in tasks[task_id]:
#             chunk_tasks[chunk_id].append(
#                 {
#                     "task_id": task_id,
#                     "start": tasks[task_id][chunk_id]["start"],
#                     "end": tasks[task_id][chunk_id]["end"],
#                     "duration": tasks[task_id][chunk_id]["duration"],
#                 }
#             )

# # Sort tasks within each chunk by start time
# for chunk_id in chunk_tasks:
#     chunk_tasks[chunk_id].sort(key=lambda x: x["start"])

# # Pack blocks efficiently by assigning rows based on non-overlapping time intervals
# all_tasks_flat = []
# for chunk_id in range(max_chunk_id + 1):
#     for task_data in chunk_tasks[chunk_id]:
#         all_tasks_flat.append({
#             "chunk_id": chunk_id,
#             "task_id": task_data["task_id"],
#             "start": task_data["start"],
#             "end": task_data["end"],
#             "duration": task_data["duration"],
#             "row": None  # Will be assigned during packing
#         })

# # Sort all tasks by start time
# all_tasks_flat.sort(key=lambda x: x["start"])

# # Assign rows to tasks using a greedy approach
# rows = []  # List of lists, each inner list represents tasks in a row
# for task in all_tasks_flat:
#     # Try to find a row where this task can fit (no overlap)
#     placed = False
#     for i, row in enumerate(rows):
#         # Check if the task can be placed in this row (no overlap with any task in row)
#         can_place = True
#         for existing_task in row:
#             # If new task starts before existing task ends and ends after existing task starts
#             if not (task["start"] >= existing_task["end"] or task["end"] <= existing_task["start"]):
#                 can_place = False
#                 break
        
#         if can_place:
#             # Place task in this row
#             row.append(task)
#             task["row"] = i
#             placed = True
#             break
    
#     # If no suitable row found, create a new row
#     if not placed:
#         rows.append([task])
#         task["row"] = len(rows) - 1

# # Draw the packed blocks
# y_ticks = []
# y_labels = []

# # Task colors - use a different colormap for tasks
# task_color_map = plt.cm.get_cmap("tab20", len(sorted_task_ids))
# task_colors = [task_color_map(i % 20) for i in range(len(sorted_task_ids))]

# # Draw each task at its assigned row
# for task in all_tasks_flat:
#     y_pos = task["row"]
#     chunk_id = task["chunk_id"]
#     task_id = task["task_id"]
    
#     # Draw rectangle with color based on chunk type
#     ax.barh(
#         y_pos,
#         task["duration"],
#         left=task["start"],
#         height=0.6,
#         color=chunk_colors[chunk_id],
#         alpha=0.8,
#         edgecolor="black",
#         linewidth=1,
#     )
    
#     # Add text label
#     text_x = task["start"] + task["duration"] / 2
#     ax.text(
#         text_x,
#         y_pos,
#         f"T{task_id}/C{chunk_id}",
#         ha="center",
#         va="center",
#         color="black",
#         fontweight="bold",
#         fontsize=9,
#     )

# # Set chart properties
# ax.set_yticks(range(len(rows)))
# ax.set_yticklabels([f"Row {i+1}" for i in range(len(rows))])
# ax.set_xlabel("Time (ms)", fontsize=12)
# ax.set_title("Chunk Execution Timeline (Packed View)", fontsize=14)
# ax.grid(True, axis="x", linestyle="--", alpha=0.7)

# # Add legend for chunks
# chunk_legend_elements = [
#     plt.Rectangle((0, 0), 1, 1, color=chunk_colors[i], label=f"Chunk {i}")
#     for i in range(max_chunk_id + 1)
# ]

# # Add task legend
# task_legend_elements = [
#     plt.Rectangle((0, 0), 1, 1, color=task_color_map(task_id % 20), alpha=0.4, label=f"Task {task_id}")
#     for task_id in sorted_task_ids
# ]

# # Create combined legend
# all_legend_elements = chunk_legend_elements + task_legend_elements
# ax.legend(
#     handles=all_legend_elements,
#     loc="upper right",
#     title="Legend",
#     ncol=2
# )

# # Adjust y-axis limits to provide some padding
# ax.set_ylim(-1, len(rows))

# # Save the figure
# plt.tight_layout()
# plt.savefig(os.path.join(base_dir, "chunk_execution_timeline_wide.png"))
# plt.close()

# print(
#     f"Chunk-centric packed chart saved to {os.path.join(base_dir, 'chunk_execution_timeline_wide.png')}"
# )

# # Create a CPU pipeline style diagram with overlapping tasks
# fig, ax = plt.subplots(figsize=(30, 12))  # Wider and taller

# # Create task rows - one row per task
# for i, task_id in enumerate(sorted_task_ids):
#     y_pos = len(sorted_task_ids) - i - 1  # Reverse order so Task 0 is at the bottom

#     # Add task label on the left
#     ax.text(
#         -5,
#         y_pos,
#         f"Task {task_id}",
#         ha="right",
#         va="center",
#         fontsize=10,
#         fontweight="bold",
#     )

#     # Draw each chunk as a colored rectangle
#     for chunk_id in range(max_chunk_id + 1):
#         if chunk_id in tasks[task_id]:
#             chunk_data = tasks[task_id][chunk_id]
#             start = chunk_data["start"]
#             end = chunk_data["end"]
#             duration = chunk_data["duration"]

#             # Draw rectangle
#             rect = plt.Rectangle(
#                 (start, y_pos - 0.3),
#                 duration,
#                 0.6,
#                 color=chunk_colors[chunk_id],
#                 alpha=0.8,
#                 edgecolor="black",
#                 linewidth=1,
#             )
#             ax.add_patch(rect)

#             # Add text label inside the rectangle
#             text_x = start + duration / 2
#             ax.text(
#                 text_x,
#                 y_pos,
#                 f"C{chunk_id}",
#                 ha="center",
#                 va="center",
#                 color="black",
#                 fontweight="bold",
#                 fontsize=9,
#             )

# # Set chart properties
# ax.set_yticks(range(len(sorted_task_ids)))
# ax.set_yticklabels([])  # Hide the default y-ticks
# ax.set_xlabel("Time (ms)", fontsize=12)
# ax.set_title("CPU Pipeline Style Visualization", fontsize=14)
# ax.grid(True, axis="x", linestyle="--", alpha=0.7)
# ax.set_xlim(-10, (max_time - min_time) / 1000 + 10)  # Add some padding on both sides
# ax.set_ylim(-1, len(sorted_task_ids))

# # Add legend for chunks
# legend_elements = [
#     plt.Rectangle((0, 0), 1, 1, color=chunk_colors[i], label=f"Chunk {i}")
#     for i in range(max_chunk_id + 1)
# ]
# ax.legend(handles=legend_elements, loc="upper right")

# # Save the figure
# plt.tight_layout()
# plt.savefig(os.path.join(base_dir, "pipeline_visualization.png"))
# plt.close()

# print(
#     f"Pipeline visualization saved to {os.path.join(base_dir, 'pipeline_visualization.png')}"
# )

# # Create a more compact overlapping visualization
# # This visualization places all chunks of the same type in the same row
# fig, ax = plt.subplots(figsize=(30, 8))

# # Create a row for each chunk type
# for chunk_id in range(max_chunk_id + 1):
#     y_pos = max_chunk_id - chunk_id  # Position chunks with 0 at the top

#     # Add chunk label on the left
#     ax.text(
#         -5,
#         y_pos,
#         f"Chunk {chunk_id}",
#         ha="right",
#         va="center",
#         fontsize=12,
#         fontweight="bold",
#     )

#     # Draw each task's instance of this chunk type
#     for task_id in sorted_task_ids:
#         if chunk_id in tasks[task_id]:
#             chunk_data = tasks[task_id][chunk_id]
#             start = chunk_data["start"]
#             duration = chunk_data["duration"]

#             # Draw rectangle with unique color per task
#             task_color = plt.cm.tab20(task_id % 20)
#             rect = plt.Rectangle(
#                 (start, y_pos - 0.4),
#                 duration,
#                 0.8,
#                 color=task_color,
#                 alpha=0.8,
#                 edgecolor="black",
#                 linewidth=1,
#             )
#             ax.add_patch(rect)

#             # Add text label inside the rectangle
#             text_x = start + duration / 2
#             ax.text(
#                 text_x,
#                 y_pos,
#                 f"T{task_id}",
#                 ha="center",
#                 va="center",
#                 color="black",
#                 fontweight="bold",
#                 fontsize=9,
#             )

# # Set chart properties
# ax.set_yticks(range(max_chunk_id + 1))
# ax.set_yticklabels([f"Chunk {max_chunk_id-i}" for i in range(max_chunk_id + 1)])
# ax.set_xlabel("Time (ms)", fontsize=12)
# ax.set_title("Overlapping Chunks Visualization (Textbook Style)", fontsize=14)
# ax.grid(True, axis="x", linestyle="--", alpha=0.7)
# ax.set_xlim(-10, (max_time - min_time) / 1000 + 10)  # Add some padding on both sides

# # Create custom legend for tasks
# task_legend_elements = [
#     plt.Rectangle(
#         (0, 0), 1, 1, color=plt.cm.tab20(task_id % 20), label=f"Task {task_id}"
#     )
#     for task_id in sorted_task_ids
# ]

# # Use a legendHandler with 5 columns to make the legend more compact
# legend = ax.legend(
#     handles=task_legend_elements,
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.15),
#     ncol=5,
# )

# # Save the figure
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.2)  # Make room for the legend
# plt.savefig(os.path.join(base_dir, "textbook_style_pipeline.png"))
# plt.close()

# print(
#     f"Textbook-style pipeline visualization saved to {os.path.join(base_dir, 'textbook_style_pipeline.png')}"
# )
