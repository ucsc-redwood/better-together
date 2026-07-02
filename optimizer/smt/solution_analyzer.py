"""Solution analysis and output formatting for schedule optimization."""

import hashlib
import json
import os

# Repo-root-relative path to the schedule contract schema (this file lives at
# optimizer/smt/, so root is two levels up).
_SCHEDULE_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "schemas", "schedule.schema.json"
)


def validate_against_schema(solutions):
    """Validate a list of schedules against schemas/schedule.schema.json.

    The producer must never emit a schedule the C++ consumer would reject, so we
    fail loud here rather than ship a malformed contract.
    """
    import jsonschema  # local import: only needed when actually writing schedules

    with open(_SCHEDULE_SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.validate(instance=solutions, schema=schema)


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
                    print(f"chunk {current_chunk} ({current_core_type}): {chunk_time:.5f} ms")
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


def get_detailed_solution(
    m, x, num_stages, core_types, stage_timings, gpu_backend=None, overhead=None
):
    """
    Extract a detailed representation of the solution including stage assignments,
    core types, and timing information.

    gpu_backend: the GPU backend token ("gpu_cuda"/"gpu_vulkan", per vocab.json
    backends[].hardware) to stamp onto GPU chunks as the schema-required "hardware"
    field. Default None keeps the legacy shape (no "hardware" key) for callers that
    patch it downstream.

    overhead: optional {core_type: (per_chunk_ms, per_stage_ms)} framework-overhead
    constants. When given, each chunk's predicted "time" includes them, matching the
    cost the solver optimized (so predicted-vs-measured comparisons stay honest).
    """

    def chunk_cost(core_type, kernel_sum, n_stages):
        oh_chunk, oh_stage = (overhead or {}).get(core_type, (0.0, 0.0))
        return kernel_sum + oh_chunk + n_stages * oh_stage

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
            chunk = {
                "id": current_chunk,
                "core_type": current_core_type,
                # 1-based, inclusive [start_stage, end_stage] -- the schedule
                # contract base (schemas/schedule.schema.json). chunk_stages is
                # 0-based and contiguous (z3 contiguity constraint), so +1 once here.
                "start_stage": chunk_stages[0] + 1,
                "end_stage": chunk_stages[-1] + 1,
                "time": chunk_cost(current_core_type, chunk_time, len(chunk_stages)),
            }
            # GPU chunks need the schema-required "hardware" token; CPU chunks omit it.
            if current_core_type == "GPU" and gpu_backend is not None:
                chunk["hardware"] = gpu_backend
            chunks.append(chunk)
            current_chunk += 1
            current_core_type = stage_core_type
            chunk_stages = [i]
            chunk_time = stage_time

    # Add the last chunk
    if chunk_stages:
        chunk = {
            "id": current_chunk,
            "core_type": current_core_type,
            "start_stage": chunk_stages[0] + 1,
            "end_stage": chunk_stages[-1] + 1,
            "time": chunk_cost(current_core_type, chunk_time, len(chunk_stages)),
        }
        if current_core_type == "GPU" and gpu_backend is not None:
            chunk["hardware"] = gpu_backend
        chunks.append(chunk)

    # stage_assignments stays a local (it builds chunks above) but is NOT exported:
    # it was a dead field on the consumer side (zero C++ readers) that duplicated and
    # drifted from chunks. The schedule contract is chunks alone.
    return finalize_solution(chunks)


def finalize_solution(chunks):
    """Metrics + readable UID for a chunk list (shared by the solver path and
    reprice_solution, so re-priced candidates stay indistinguishable in shape)."""
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
        cores_summary += str(chunk["end_stage"] - chunk["start_stage"] + 1)

    # Add gapness and unique hash to ensure uniqueness
    gapness_str = f"{metrics.get('gapness', 0):.2f}".replace(".", "")
    unique_hash = hashlib.md5(str(chunks).encode()).hexdigest()[:4]
    uid = f"SCH-{cores_summary}-G{gapness_str}-{unique_hash}"

    return {
        "uid": uid,
        "chunks": chunks,
        "metrics": metrics,
    }


def reprice_solution(solution, core_types, stage_timings, overhead):
    """Re-express a solution under the overhead cost model: rebuild every chunk's
    predicted "time" from the stage timings + overhead constants, then regenerate
    metrics and uid to match. Used by the union-candidate sweep in
    02_gen_schedule_merged so candidates discovered by the PLAIN model carry the
    same prediction semantics as the overhead-model candidates in the same file."""
    new_chunks = []
    for chunk in solution["chunks"]:
        col = core_types.index(chunk["core_type"])
        n = chunk["end_stage"] - chunk["start_stage"] + 1
        kernel = sum(
            stage_timings[k][col] for k in range(chunk["start_stage"] - 1, chunk["end_stage"])
        )
        oh_chunk, oh_stage = (overhead or {}).get(chunk["core_type"], (0.0, 0.0))
        new_chunk = dict(chunk)
        new_chunk["time"] = kernel + oh_chunk + n * oh_stage
        new_chunks.append(new_chunk)
    return finalize_solution(new_chunks)


def dump_solutions_as_json(solutions, baseline_data, output_format="pretty", output_file=None):
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

            # Calculate speedups against the pipeline MAKESPAN (max chunk time =
            # steady-state per-task throughput), NOT avg chunk time. avg <= max always,
            # so avg inflated every speedup and favored imbalanced schedules.
            if "max_time" in solution["metrics"]:
                makespan = solution["metrics"]["max_time"]
                if "omp" in baseline_data:
                    solution["metrics"]["speedup_over_cpu"] = baseline_data["omp"] / makespan

                gpu_key = next(
                    (k for k in baseline_data.keys() if k not in ["omp", "fastest"]),
                    None,
                )
                if gpu_key:
                    solution["metrics"]["speedup_over_gpu"] = baseline_data[gpu_key] / makespan

    # Fail before writing if the produced schedules don't match the contract the
    # C++ consumer reads (schemas/schedule.schema.json).
    validate_against_schema(solutions)

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
