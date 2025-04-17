from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And


def define_data():
    """Define the problem data."""
    num_stages = 9
    type_value = {"Little": 0, "Medium": 1, "Big": 2, "GPU": 3}
    core_types = ["Little", "Medium", "Big", "GPU"]

    #         little   medium      big  vulkan
    # stage
    # 1      19.5931   5.6027   4.4183  7.7635
    # 2      23.2424  12.4853  10.7541  6.2564
    # 3      10.1525   2.9634   2.3622  4.4883
    # 4      12.5767   6.5304   5.7360  3.3440
    # 5       5.7409   1.6797   1.4564  2.6495
    # 6       5.7451   1.6710   1.4679  2.2917
    # 7       5.7631   1.6806   1.5271  2.3128
    # 8       6.9995   3.4611   3.1758  2.1264
    # 9       0.0253   0.0118   0.0111  0.4699
    fully_stage_timings = [
        [23.6234, 5.8916, 6.0479, 4.7451],
        [29.3516, 13.2806, 14.9687, 2.7761],
        [15.0454, 3.1998, 3.8297, 2.4621],
        [16.8728, 6.9002, 8.2633, 1.6953],
        [9.0700, 1.8227, 2.3823, 1.7382],
        [9.2635, 1.9144, 2.5132, 1.6730],
        [8.8400, 1.9034, 2.4333, 1.7638],
        [9.0580, 3.9778, 4.6411, 0.9440],
        [0.0536, 0.0175, 0.0167, 0.2801],
    ]

    normal_stage_timings = [
        [19.5931, 5.6027, 4.4183, 7.7635],
        [23.2424, 12.4853, 10.7541, 6.2564],
        [10.1525, 2.9634, 2.3622, 4.4883],
        [12.5767, 6.5304, 5.7360, 3.3440],
        [5.7409, 1.6797, 1.4564, 2.6495],
        [5.7451, 1.6710, 1.4679, 2.2917],
        [5.7631, 1.6806, 1.5271, 2.3128],
        [6.9995, 3.4611, 3.1758, 2.1264],
        [0.0253, 0.0118, 0.0111, 0.4699],
    ]

    return num_stages, core_types, type_value, fully_stage_timings, normal_stage_timings


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


def solve_optimization_problem():
    """Solve the optimization problem and display the solution."""
    # Initialize data
    num_stages, core_types, type_value, fully_stage_timings, normal_stage_timings = (
        define_data()
    )

    # Number of solutions to find
    num_solutions = 40

    # Prepare a list to hold up to num_solutions solutions
    top_solutions = []

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
        print(f"Gap between max and min: {gapness_value} ms")
        print(f"Max chunk time: {max_time} ms")
        print(f"Min chunk time: {min_time} ms")

        # Print details
        # print_stage_assignments(m, x, num_stages, core_types, stage_timings)
        print_stage_assignments_v2(m, x, num_stages, core_types, fully_stage_timings)
        chunk_times = print_chunk_summary(
            m, x, num_stages, core_types, fully_stage_timings
        )

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
            print(f"Solution {i + 1}: Gap = {gapness} ms, Max time = {max_time} ms")


if __name__ == "__main__":
    solve_optimization_problem()
