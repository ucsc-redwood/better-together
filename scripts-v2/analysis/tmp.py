#!/usr/bin/env python3
import re
import statistics


def parse_file(filename):
    """
    Parse the input file "in.txt" and return a list of tasks.
    Each task is a dictionary containing a 'task_id' and a list of 'chunks'.
    Each chunk is represented as a dictionary with chunk_id, start time, end time, and duration.
    """
    tasks = []
    current_task = None
    current_chunk = None

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            # Detect a new task.
            task_match = re.match(r"^Task\s+(\d+):", line)
            if task_match:
                if current_task is not None:
                    tasks.append(current_task)
                current_task = {"task_id": int(task_match.group(1)), "chunks": []}
                continue

            # Detect a new chunk.
            chunk_match = re.match(r"^Chunk\s+(\d+):", line)
            if chunk_match:
                current_chunk = {"chunk_id": int(chunk_match.group(1))}
                continue

            # Extract start time.
            start_match = re.match(r"^Start:\s+(\d+)\s+us", line)
            if start_match and current_chunk is not None:
                current_chunk["start"] = int(start_match.group(1))
                continue

            # Extract end time.
            end_match = re.match(r"^End:\s+(\d+)\s+us", line)
            if end_match and current_chunk is not None:
                current_chunk["end"] = int(end_match.group(1))
                continue

            # Extract duration.
            duration_match = re.match(r"^Duration:\s+(\d+)\s+us", line)
            if duration_match and current_chunk is not None:
                current_chunk["duration"] = int(duration_match.group(1))
                # Finished reading one chunk; add it to the current task.
                current_task["chunks"].append(current_chunk)
                current_chunk = None

    if current_task is not None:
        tasks.append(current_task)
    return tasks


def compute_statistics(tasks):
    """
    Compute global statistics: total number of tasks/chunks,
    global start/end times and overall duration,
    and aggregate statistics based on task durations and chunk durations.
    """
    num_tasks = len(tasks)
    num_chunks = sum(len(task["chunks"]) for task in tasks)
    task_durations = []  # Total duration per task (sum of its chunks' durations)
    all_chunk_durations = []  # All individual chunk durations

    global_start = None
    global_end = None

    for task in tasks:
        total_duration = sum(chunk["duration"] for chunk in task["chunks"])
        task_durations.append(total_duration)

        for chunk in task["chunks"]:
            duration = chunk["duration"]
            all_chunk_durations.append(duration)
            start_time = chunk["start"]
            end_time = chunk["end"]
            if global_start is None or start_time < global_start:
                global_start = start_time
            if global_end is None or end_time > global_end:
                global_end = end_time

    global_duration = global_end - global_start

    stats = {
        "num_tasks": num_tasks,
        "num_chunks": num_chunks,
        "global_start": global_start,
        "global_end": global_end,
        "global_duration": global_duration,
        "task_avg": statistics.mean(task_durations),
        "task_min": min(task_durations),
        "task_max": max(task_durations),
        "task_std": statistics.stdev(task_durations) if len(task_durations) > 1 else 0,
        "chunk_avg": statistics.mean(all_chunk_durations),
        "chunk_min": min(all_chunk_durations),
        "chunk_max": max(all_chunk_durations),
        "chunk_std": (
            statistics.stdev(all_chunk_durations) if len(all_chunk_durations) > 1 else 0
        ),
    }
    return stats


def compute_chunk_type_stats(tasks):
    """
    For each chunk type (0, 1, 2, 3), compute statistical measures:
    average, minimum, maximum, standard deviation, and count.
    """
    chunk_data = {0: [], 1: [], 2: [], 3: []}
    for task in tasks:
        for chunk in task["chunks"]:
            cid = chunk["chunk_id"]
            chunk_data[cid].append(chunk["duration"])

    chunk_type_stats = {}
    for cid, durations in chunk_data.items():
        if durations:
            chunk_type_stats[cid] = {
                "count": len(durations),
                "avg": statistics.mean(durations),
                "min": min(durations),
                "max": max(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0,
            }
        else:
            chunk_type_stats[cid] = {"count": 0, "avg": 0, "min": 0, "max": 0, "std": 0}
    return chunk_type_stats


def analyze_tasks(tasks):
    """
    For each task, compute the spread (range) in chunk durations.
    Returns a list with tuple (task_id, range, min_chunk, max_chunk)
    and identifies the "widest" task (largest range).
    """
    task_ranges = []  # List of tuples: (task_id, range, min_duration, max_duration)
    for task in tasks:
        durations = [chunk["duration"] for chunk in task["chunks"]]
        task_min = min(durations)
        task_max = max(durations)
        duration_range = task_max - task_min
        task_ranges.append((task["task_id"], duration_range, task_min, task_max))
    widest_task = max(task_ranges, key=lambda x: x[1]) if task_ranges else None
    return task_ranges, widest_task


def main():
    filename = "in.txt"
    tasks = parse_file(filename)

    # Compute global statistics.
    stats = compute_statistics(tasks)

    # Compute statistics for each chunk type.
    chunk_type_stats = compute_chunk_type_stats(tasks)

    # Analyze each task for the range (spread) in chunk durations.
    task_ranges, widest_task = analyze_tasks(tasks)

    # Convert microsecond values to milliseconds.
    us_to_ms = lambda us: us / 1000.0
    global_start_ms = us_to_ms(stats["global_start"])
    global_end_ms = us_to_ms(stats["global_end"])
    global_duration_ms = us_to_ms(stats["global_duration"])

    task_avg_ms = us_to_ms(stats["task_avg"])
    task_min_ms = us_to_ms(stats["task_min"])
    task_max_ms = us_to_ms(stats["task_max"])
    task_std_ms = us_to_ms(stats["task_std"])

    chunk_avg_ms = us_to_ms(stats["chunk_avg"])
    chunk_min_ms = us_to_ms(stats["chunk_min"])
    chunk_max_ms = us_to_ms(stats["chunk_max"])
    chunk_std_ms = us_to_ms(stats["chunk_std"])

    print("=== GLOBAL STATISTICS ===")
    print(f"Number of tasks: {stats['num_tasks']}")
    print(f"Total number of chunks: {stats['num_chunks']}\n")

    print("Global Timing Information:")
    print(f"  Global start time: {global_start_ms:.3f} ms")
    print(f"  Global end time: {global_end_ms:.3f} ms")
    print(f"  Overall duration: {global_duration_ms:.3f} ms\n")

    print("Task Duration Statistics (each task is the sum of its chunks):")
    print(f"  Average task duration: {task_avg_ms:.3f} ms")
    print(f"  Min task duration: {task_min_ms:.3f} ms")
    print(f"  Max task duration: {task_max_ms:.3f} ms")
    print(f"  Standard deviation: {task_std_ms:.3f} ms\n")

    print("Chunk Duration Statistics (all chunks):")
    print(f"  Average chunk duration: {chunk_avg_ms:.3f} ms")
    print(f"  Min chunk duration: {chunk_min_ms:.3f} ms")
    print(f"  Max chunk duration: {chunk_max_ms:.3f} ms")
    print(f"  Standard deviation: {chunk_std_ms:.3f} ms\n")

    print("=== CHUNK TYPE STATISTICS ===")
    for cid in sorted(chunk_type_stats.keys()):
        s = chunk_type_stats[cid]
        print(f"Chunk Type {cid}:")
        print(f"  Count: {s['count']}")
        print(f"  Average duration: {us_to_ms(s['avg']):.3f} ms")
        print(f"  Min duration: {us_to_ms(s['min']):.3f} ms")
        print(f"  Max duration: {us_to_ms(s['max']):.3f} ms")
        print(f"  Standard deviation: {us_to_ms(s['std']):.3f} ms")
        print()

    print("=== TASK ANALYSIS (Chunk Duration Spread) ===")
    print("Task ID |   Range (ms)  |  Min chunk (ms)  |  Max chunk (ms)")
    for task_id, drange, tmin, tmax in sorted(task_ranges, key=lambda x: x[0]):
        print(
            f"{task_id:7d} | {us_to_ms(drange):12.3f} | {us_to_ms(tmin):15.3f} | {us_to_ms(tmax):15.3f}"
        )

    if widest_task:
        wt_id, wt_range, wt_min, wt_max = widest_task
        print("\nWidest Task:")
        print(f"  Task ID: {wt_id}")
        print(
            f"  Range of chunk durations: {us_to_ms(wt_range):.3f} ms (min: {us_to_ms(wt_min):.3f} ms, max: {us_to_ms(wt_max):.3f} ms)"
        )


if __name__ == "__main__":
    main()
