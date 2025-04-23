from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And
import pandas as pd
import argparse
import numpy as np
import json
import uuid
import hashlib
import os


def load_csv_and_compute_averages(csv_path):
    """
    Load data from a CSV file and compute average timings for each stage across all runs.

    Args:
        csv_path: Path to the CSV file containing stage timing data

    Returns:
        A list of lists containing average timing data for each stage and core type
    """
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Compute average for each stage across all runs
    # Group by stage and calculate mean for each core type
    avg_df = df.groupby("stage")[["little", "medium", "big", "vulkan", "cuda"]].mean()

    # Print the average table
    print("\n=== Average Stage Timings ===")
    print(avg_df)
    print()

    # Convert to list of lists format expected by the solver
    # Each inner list is [little_time, medium_time, big_time, vulkan_time]
    avg_timings = []
    for stage in range(1, 10):  # Assuming 9 stages, numbered 1-9
        if stage in avg_df.index:
            row = avg_df.loc[stage]
            avg_timings.append(
                [row["little"], row["medium"], row["big"], row["vulkan"]]
            )
        else:
            print(f"Warning: Stage {stage} not found in data")
            # Use zeros as fallback
            avg_timings.append([0.0, 0.0, 0.0, 0.0])

    return avg_timings


def define_data(stage_timings=None):
    """Define the problem data."""
    num_stages = 9
    core_types = ["Little", "Medium", "Big", "GPU"]

    # Use provided stage timings if available, otherwise use default values
    if stage_timings is not None:
        return num_stages, core_types, stage_timings

    # Default timings if no CSV is provided
    default_stage_timings = []

    return num_stages, core_types, default_stage_timings


def create_decision_variables(num_stages, core_types):
    """Create decision variables x[i, c] where x[i, c] is True if stage i is assigned to core type c."""
    x = {}
    for i in range(num_stages):
        for c in core_types:
            x[(i, c)] = Bool(f"x_{i}_{c}")
    return x


def add_assignment_constraints(opt, x, num_stages, core_types):
    """Add constraints to ensure each stage is assigned exactly one processing unit."""
    for i in range(num_stages):

        # At least one PU must be chosen.
        opt.add(Or([x[(i, c)] for c in core_types]))

        # And at most one PU can be chosen.
        for j in range(len(core_types)):
            for k in range(j + 1, len(core_types)):
                opt.add(Or(Not(x[(i, core_types[j])]), Not(x[(i, core_types[k])])))


def add_chunk_time_constraint(opt, x, core_types, num_stages, stage_timings):
    """Add constraints for the chunk times and optimize for minimal gap between max and min chunk times."""
    # Define the maximum and minimum chunk time variables
    T_max = Real("T_max")
    T_min = Real("T_min")
    Gapness = Real("Gapness")

    opt.add(T_max > 0)
    opt.add(T_min > 0)

    # For each PU and every contiguous segment of stages, if the entire segment is handled by PU c,
    # then the sum of timings for that segment must be <= T_max and >= T_min (if it's assigned).
    for c in core_types:
        for i in range(num_stages):
            for j in range(i, num_stages):
                # Build the condition: all stages k in [i, j] are assigned PU c.
                segment_assigned = And([x[(k, c)] for k in range(i, j + 1)])
                # Compute the sum over the segment.
                seg_sum = Sum(
                    [
                        RealVal(stage_timings[k][core_types.index(c)])
                        for k in range(i, j + 1)
                    ]
                )
                # Add the implication: if the segment is uniformly assigned c, then seg_sum <= T_max.
                opt.add(Implies(segment_assigned, seg_sum <= T_max))

                # Also, if this is an actual chunk (i.e., it's a maximal contiguous segment assigned to c),
                # then its time should be >= T_min
                # Check if this is a maximal segment (no stages before i or after j assigned to c)
                is_start = i == 0 or Not(x[(i - 1, c)])
                is_end = j == num_stages - 1 or Not(x[(j + 1, c)])
                is_maximal_segment = And(segment_assigned, is_start, is_end)

                # If this is a maximal segment, its time should be >= T_min
                opt.add(Implies(is_maximal_segment, seg_sum >= T_min))

    # Define Gapness as the difference between max and min chunk times
    opt.add(Gapness == T_max - T_min)

    # Optimize only for minimal gap
    opt.minimize(Gapness)

    # Optimize only for T_max
    # opt.minimize(T_max)

    return T_max, T_min, Gapness


def add_contiguity_constraints(opt, x, core_types, num_stages):
    """Add constraints to ensure each PU appears in one continuous block."""
    for c in core_types:
        for i in range(num_stages):
            for j in range(i + 1, num_stages):
                for k in range(j + 1, num_stages):
                    opt.add(Implies(And(x[(i, c)], x[(k, c)]), x[(j, c)]))


def add_no_big_cores_constraint(opt, x, num_stages, core_types):
    """Add constraints to prevent using big cores."""
    for i in range(num_stages):
        opt.add(Not(x[(i, "Big")]))
    print("Adding constraint: No Big cores allowed")


def add_no_little_cores_constraint(opt, x, num_stages, core_types):
    """Add constraints to prevent using little cores."""
    for i in range(num_stages):
        opt.add(Not(x[(i, "Little")]))
    print("Adding constraint: No Little cores allowed")


def add_no_gpu_constraint(opt, x, num_stages, core_types):
    """Add constraints to prevent using GPU."""
    for i in range(num_stages):
        opt.add(Not(x[(i, "GPU")]))
    print("Adding constraint: No GPU allowed")


def block_solution(opt, x, num_stages, core_types, model):
    """Add constraint to block the current solution, so we can find a new one."""
    block = []
    for i in range(num_stages):
        for c in core_types:
            if model.evaluate(x[(i, c)]):
                block.append(Not(x[(i, c)]))
            else:
                block.append(x[(i, c)])
    opt.add(Or(block))


def print_stage_assignments(m, x, num_stages, core_types, stage_timings):
    """Print the assignment of stages to core types."""
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                print(
                    f"Stage {i}: core type {c} with time {stage_timings[i][core_types.index(c)]}"
                )


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


def solve_optimization_problem(stage_timings, num_solutions=30):
    """Solve the optimization problem and display the solution."""
    # Initialize data
    num_stages, core_types, fully_stage_timings = define_data(stage_timings)

    # Prepare a list to hold up to num_solutions solutions
    top_solutions = []
    detailed_solutions = []

    # Create optimizer
    opt = Optimize()

    # Create decision variables
    x = create_decision_variables(num_stages, core_types)

    # Add constraints
    add_assignment_constraints(opt, x, num_stages, core_types)
    T_max, T_min, Gapness = add_chunk_time_constraint(
        opt, x, core_types, num_stages, fully_stage_timings
    )
    add_contiguity_constraints(opt, x, core_types, num_stages)

    # Enable/disable optional constraints here
    # Uncomment the constraints you want to apply

    # add_no_big_cores_constraint(opt, x, num_stages, core_types)
    # add_no_little_cores_constraint(opt, x, num_stages, core_types)
    # add_no_gpu_constraint(opt, x, num_stages, core_types)

    print("\nOptimization approach: Minimizing the gap between max and min chunk times")
    print("---------------------------------------------------------------------")

    # Find up to num_solutions solutions
    solution_count = 0

    while solution_count < num_solutions and opt.check() == sat:
        m = opt.model()
        max_time = float(m[T_max].as_fraction())
        min_time = float(m[T_min].as_fraction())
        gapness_value = float(m[Gapness].as_fraction())

        print(f"\n=== Solution {solution_count + 1} ===")
        print(f"Gap between max and min: {gapness_value:.2f} ms")
        print(f"Max chunk time: {max_time:.2f} ms")
        print(f"Min chunk time: {min_time:.2f} ms")

        # Print details
        # print_stage_assignments(m, x, num_stages, core_types, fully_stage_timings)
        print_stage_assignments_v2(m, x, num_stages, core_types, fully_stage_timings)
        chunk_times = print_chunk_summary(
            m, x, num_stages, core_types, fully_stage_timings
        )

        # Get detailed solution for JSON output
        detailed_solution = get_detailed_solution(
            m, x, num_stages, core_types, fully_stage_timings
        )
        detailed_solution["solution_id"] = solution_count + 1
        detailed_solution["metrics"]["max_time"] = max_time
        detailed_solution["metrics"]["min_time"] = min_time
        detailed_solution["metrics"]["gapness"] = gapness_value

        # Print UID for reference
        print(f"Solution UID: {detailed_solution['uid']}")

        detailed_solutions.append(detailed_solution)

        # Store solution
        solution_repr = get_solution_representation(m, x, num_stages, core_types)
        top_solutions.append((gapness_value, max_time, solution_repr))

        # Block this solution to find the next one
        block_solution(opt, x, num_stages, core_types, m)

        solution_count += 1

    if solution_count == 0:
        print("No solution found.")
    else:
        # Print a summary of all solutions
        print("\n=== Summary of All Solutions ===")
        for i, (gapness, max_time, _) in enumerate(top_solutions):
            solution_uid = detailed_solutions[i]["uid"]
            print(
                f"Solution {i + 1}: Gap = {gapness:.2f} ms, Max time = {max_time:.2f} ms, UID: {solution_uid}"
            )

    return detailed_solutions


def dump_solutions_as_json(solutions, output_format="pretty", output_file=None):
    """
    Dump solutions in a format that can be easily parsed by Python.

    Args:
        solutions: List of solution dictionaries
        output_format: 'pretty' for formatted JSON or 'compact' for compact JSON
        output_file: Path to a file to write the JSON output to. If None, output to console only.
    """
    print("\n\n=== MACHINE PARSABLE OUTPUT START ===")
    if output_format == "pretty":
        json_str = json.dumps(solutions, indent=2)
    else:
        json_str = json.dumps(solutions)

    # print(json_str)
    print("=== MACHINE PARSABLE OUTPUT END ===")

    # Write to file if path is specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write(json_str)
            print(f"\nSolutions written to {output_file}")
        except Exception as e:
            print(f"\nError writing to file {output_file}: {str(e)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve scheduling optimization problem using data from a CSV file."
    )
    parser.add_argument(
        "--csv_folder",
        type=str,
        help="Path to the CSV folder with stage timing data",
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--backend", required=True, choices=["vk", "cu"])
    parser.add_argument(
        "-n",
        "--num_solutions",
        type=int,
        help="Number of solutions to find",
        default=20,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to write the JSON output file (optional)",
        default=None,
    )
    # parser.add_argument(
    #     "--compact",
    #     action="store_true",
    #     help="Write JSON in compact format instead of pretty format",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.csv_folder:
        backend = args.backend
        device = args.device
        app = args.app
        csv_path = os.path.join(args.csv_folder, f"{device}_{app}_{backend}_fully.csv")
        out_path = os.path.join(args.output_folder, f"{device}_{app}_{backend}_fully_schedules.json")

        print(f"Loading data from CSV file: {csv_path}")
        stage_timings = load_csv_and_compute_averages(csv_path)
        solutions = solve_optimization_problem(stage_timings, args.num_solutions)

        # Dump solutions in a machine-parsable format
        dump_solutions_as_json(solutions, "pretty", out_path)
