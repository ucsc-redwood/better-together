"""Solution analysis and output formatting for schedule optimization."""

import hashlib
import json
import os


def print_stage_assignments_v2(m, x, num_stages, core_types, stage_timings):
    """Print the assignment of stages to core types."""
    # Group stages by core type
    core_stages = {}
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                if c not in core_stages:
                    core_stages[c] = []
                core_stages[c].append(i)

    # Print stages grouped by core type
    print("\nStage assignments:")
    for core_type, stages in core_stages.items():
        print(f"{core_type} = {stages}")


def print_chunk_summary(m, x, num_stages, core_types, stage_timings):
    """Print a summary of chunks with their core types and times."""
    print("\nMath model summary:")
    current_chunk = 0
    current_core_type = None
    chunk_time = 0.0
    chunk_times = []
    chunk_details = []

    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                if current_core_type is None:
                    current_core_type = c
                    chunk_time = stage_timings[i][core_types.index(c)]
                elif c == current_core_type:
                    chunk_time += stage_timings[i][core_types.index(c)]
                else:
                    print(
                        f"chunk {current_chunk} ({current_core_type}): {chunk_time:.5f} ms"
                    )
                    chunk_times.append(chunk_time)
                    chunk_details.append((current_chunk, current_core_type, chunk_time))
                    current_chunk += 1
                    current_core_type = c
                    chunk_time = stage_timings[i][core_types.index(c)]
                break

    # Print the last chunk
    if current_core_type is not None:
        print(f"chunk {current_chunk} ({current_core_type}): {chunk_time:.5f} ms")
        chunk_times.append(chunk_time)
        chunk_details.append((current_chunk, current_core_type, chunk_time))

    # Calculate load balancing metrics
    if chunk_times:
        max_time = max(chunk_times)
        max_chunk_index = chunk_times.index(max_time)
        max_chunk_details = chunk_details[max_chunk_index]

        min_time = min(chunk_times)
        min_chunk_index = chunk_times.index(min_time)
        min_chunk_details = chunk_details[min_chunk_index]

        avg_time = sum(chunk_times) / len(chunk_times)
        load_balance_ratio = min_time / max_time
        load_imbalance_pct = (1 - load_balance_ratio) * 100
        time_variance = sum((t - avg_time) ** 2 for t in chunk_times) / len(chunk_times)

        print(f"\nChunk Time Highlights:")
        print(
            f"Widest chunk: chunk {max_chunk_details[0]} ({max_chunk_details[1]}) with {max_time:.5f} ms"
        )
        print(
            f"Shortest chunk: chunk {min_chunk_details[0]} ({min_chunk_details[1]}) with {min_time:.5f} ms"
        )
        print(f"Gapness (max-min): {max_time - min_time:.5f} ms")

        print(f"\nLoad Balancing Metrics:")
        print(f"Load balance ratio: {load_balance_ratio:.5f}")
        print(f"Load imbalance percentage: {load_imbalance_pct:.5f}%")
        print(f"Time variance: {time_variance:.5f}")

    return chunk_times if chunk_times else []


def get_solution_representation(m, x, num_stages, core_types):
    """Get a representation of the solution for storage."""
    solution = []
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                solution.append((i, c))
                break
    return solution


def get_detailed_solution(m, x, num_stages, core_types, stage_timings):
    """
    Extract a detailed representation of the solution including stage assignments,
    core types, and timing information.
    """
    # Get assignment of stages to core types
    stage_assignments = {}
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                stage_assignments[i] = {
                    "core_type": c,
                    "time": stage_timings[i][core_types.index(c)],
                }
                break

    # Extract chunk information
    chunks = []
    current_chunk = 0
    current_core_type = None
    chunk_stages = []
    chunk_time = 0.0

    for i in range(num_stages):
        stage_core_type = stage_assignments[i]["core_type"]
        stage_time = stage_assignments[i]["time"]

        if current_core_type is None:
            # First stage
            current_core_type = stage_core_type
            chunk_stages.append(i)
            chunk_time = stage_time
        elif stage_core_type == current_core_type:
            # Continuing the current chunk
            chunk_stages.append(i)
            chunk_time += stage_time
        else:
            # New chunk starts
            chunks.append(
                {
                    "id": current_chunk,
                    "core_type": current_core_type,
                    "stages": chunk_stages.copy(),
                    "time": chunk_time,
                }
            )
            current_chunk += 1
            current_core_type = stage_core_type
            chunk_stages = [i]
            chunk_time = stage_time

    # Add the last chunk
    if chunk_stages:
        chunks.append(
            {
                "id": current_chunk,
                "core_type": current_core_type,
                "stages": chunk_stages.copy(),
                "time": chunk_time,
            }
        )

    # Calculate load balancing metrics
    chunk_times = [chunk["time"] for chunk in chunks]
    if chunk_times:
        max_time = max(chunk_times)
        min_time = min(chunk_times)
        avg_time = sum(chunk_times) / len(chunk_times)
        load_balance_ratio = min_time / max_time
        load_imbalance_pct = (1 - load_balance_ratio) * 100
        time_variance = sum((t - avg_time) ** 2 for t in chunk_times) / len(chunk_times)

        metrics = {
            "max_time": max_time,
            "min_time": min_time,
            "gapness": max_time - min_time,
            "avg_time": avg_time,
            "load_balance_ratio": load_balance_ratio,
            "load_imbalance_pct": load_imbalance_pct,
            "time_variance": time_variance,
        }
    else:
        metrics = {}

    # Generate a readable UID for the solution
    # Format: SCH-{cores_summary}-G{gapness:.2f}
    cores_summary = ""
    for chunk in chunks:
        if chunk["core_type"] == "Little":
            cores_summary += "L"
        elif chunk["core_type"] == "Medium":
            cores_summary += "M"
        elif chunk["core_type"] == "Big":
            cores_summary += "B"
        elif chunk["core_type"] == "GPU":
            cores_summary += "G"
        cores_summary += str(len(chunk["stages"]))

    # Add gapness and unique hash to ensure uniqueness
    gapness_str = f"{metrics.get('gapness', 0):.2f}".replace(".", "")
    unique_hash = hashlib.md5(str(chunks).encode()).hexdigest()[:4]
    uid = f"SCH-{cores_summary}-G{gapness_str}-{unique_hash}"

    return {
        "uid": uid,
        "stage_assignments": stage_assignments,
        "chunks": chunks,
        "metrics": metrics,
    }


def dump_solutions_as_json(
    solutions, baseline_data, output_format="pretty", output_file=None
):
    """
    Dump solutions in a format that can be easily parsed by Python.

    Args:
        solutions: List of solution dictionaries
        baseline_data: Dictionary containing baseline timing data
        output_format: 'pretty' for formatted JSON or 'compact' for compact JSON
        output_file: Path to a file to write the JSON output to. If None, output to console only.
    """
    # Add baseline data to each solution's metrics
    for solution in solutions:
        if baseline_data and "metrics" in solution:
            # Add baseline values to metrics
            for key, value in baseline_data.items():
                solution["metrics"][key] = value

            # Calculate speedups
            if "avg_time" in solution["metrics"]:
                avg_time = solution["metrics"]["avg_time"]
                if "omp" in baseline_data:
                    solution["metrics"]["speedup_over_cpu"] = (
                        baseline_data["omp"] / avg_time
                    )

                gpu_key = next(
                    (k for k in baseline_data.keys() if k not in ["omp", "fastest"]),
                    None,
                )
                if gpu_key:
                    solution["metrics"]["speedup_over_gpu"] = (
                        baseline_data[gpu_key] / avg_time
                    )

    # print("\n\n=== MACHINE PARSABLE OUTPUT START ===")

    if output_format == "pretty":
        json_str = json.dumps(solutions, indent=2)
    else:
        json_str = json.dumps(solutions)

    # print("=== MACHINE PARSABLE OUTPUT END ===")

    # Write to file if path is specified
    if output_file:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            with open(output_file, "w") as f:
                f.write(json_str)
            print(f"\nSolutions written to {output_file}")
        except Exception as e:
            print(f"\nError writing to file {output_file}: {str(e)}")
