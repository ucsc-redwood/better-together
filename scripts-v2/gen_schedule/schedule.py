from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And


def define_data():
    """Define the problem data."""
    num_stages = 9
    type_value = {"Little": 0, "Medium": 1, "Big": 2, "GPU": 3}
    core_types = ["Little", "Medium", "Big", "GPU"]

    # this is a avg table of Stages x Core Types
    # Stage 1: 28.94 7.69 10.68 5.28
    # Stage 2: 55.03 27.91 33.87 3.41
    # Stage 3: 25.15 6.53 6.73 3.09
    # Stage 4: 27.07 14.11 17.14 2.43
    # Stage 5: 13.78 3.85 3.87 1.93
    # Stage 6: 13.79 3.96 3.89 1.99
    # Stage 7: 14.34 4.03 3.94 2.15
    # Stage 8: 15.57 7.46 10.15 1.43
    # Stage 9: 0.08 0.03 0.03 0.40
    stage_timings = [
        [28.94, 7.69, 10.68, 5.28],
        [55.03, 27.91, 33.87, 3.41],
        [25.15, 6.53, 6.73, 3.09],
        [27.07, 14.11, 17.14, 2.43],
        [13.78, 3.85, 3.87, 1.93],
        [13.79, 3.96, 3.89, 1.99],
        [14.34, 4.03, 3.94, 2.15],
        [15.57, 7.46, 10.15, 1.43],
        [0.08, 0.03, 0.03, 0.40],
    ]

    return num_stages, core_types, type_value, stage_timings


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
    """Add constraints for the chunk time T_chunk."""
    # Define the "chunk time" variable (to be minimized) as a real number.
    T_chunk = Real("T_chunk")
    opt.add(T_chunk > 0)

    # For each PU and every contiguous segment of stages, if the entire segment is handled by PU c,
    # then the sum of timings for that segment must be <= T_chunk.
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
                # Add the implication: if the segment is uniformly assigned c, then seg_sum <= T_chunk.
                opt.add(Implies(segment_assigned, seg_sum <= T_chunk))

    # Objective: Minimize T_chunk.
    opt.minimize(T_chunk)

    return T_chunk


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
                    current_chunk += 1
                    current_core_type = c
                    chunk_time = stage_timings[i][core_types.index(c)]
                break

    # Print the last chunk
    if current_core_type is not None:
        print(f"chunk {current_chunk} ({current_core_type}): {chunk_time:.5f} ms")
        chunk_times.append(chunk_time)

    # Calculate load balancing metrics
    if chunk_times:
        max_time = max(chunk_times)
        min_time = min(chunk_times)
        avg_time = sum(chunk_times) / len(chunk_times)
        load_balance_ratio = min_time / max_time
        load_imbalance_pct = (1 - load_balance_ratio) * 100
        time_variance = sum((t - avg_time) ** 2 for t in chunk_times) / len(chunk_times)

        print(f"\nLoad Balancing Metrics:")
        print(f"Load balance ratio: {load_balance_ratio:.5f}")
        print(f"Load imbalance percentage: {load_imbalance_pct:.5f}%")
        print(f"Time variance: {time_variance:.5f}")


def get_solution_representation(m, x, num_stages, core_types):
    """Get a representation of the solution for storage."""
    solution = []
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                solution.append((i, c))
                break
    return solution


def solve_optimization_problem():
    """Solve the optimization problem and display the solution."""
    # Initialize data
    num_stages, core_types, type_value, stage_timings = define_data()

    # Number of solutions to find
    num_solutions = 10

    # Prepare a list to hold up to num_solutions solutions
    top_solutions = []

    # Create optimizer
    opt = Optimize()

    # Create decision variables
    x = create_decision_variables(num_stages, core_types)

    # Add constraints
    add_assignment_constraints(opt, x, num_stages, core_types)
    T_chunk = add_chunk_time_constraint(opt, x, core_types, num_stages, stage_timings)
    add_contiguity_constraints(opt, x, core_types, num_stages)

    # Enable/disable optional constraints here
    # Uncomment the constraints you want to apply

    # add_no_big_cores_constraint(opt, x, num_stages, core_types)
    # add_no_little_cores_constraint(opt, x, num_stages, core_types)
    # add_no_gpu_constraint(opt, x, num_stages, core_types)

    # Find up to num_solutions solutions
    solution_count = 0
    while solution_count < num_solutions and opt.check() == sat:
        m = opt.model()
        solution_time = float(m[T_chunk].as_fraction())

        # Print solution header
        print(f"\n=== Solution {solution_count + 1} ===")
        print(f"Optimal chunk time: {m[T_chunk]} = {solution_time} ms")

        # Print details
        # print_stage_assignments(m, x, num_stages, core_types, stage_timings)
        print_stage_assignments_v2(m, x, num_stages, core_types, stage_timings)
        print_chunk_summary(m, x, num_stages, core_types, stage_timings)

        # Store solution
        solution_repr = get_solution_representation(m, x, num_stages, core_types)
        top_solutions.append((solution_time, solution_repr))

        # Block this solution to find the next one
        block_solution(opt, x, num_stages, core_types, m)

        solution_count += 1

    if solution_count == 0:
        print("No solution found.")
    else:
        # Print a summary of all solutions
        print("\n=== Summary of All Solutions ===")
        for i, (time, _) in enumerate(top_solutions):
            print(f"Solution {i + 1}: Chunk time = {time} ms")


if __name__ == "__main__":
    solve_optimization_problem()
