from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And


def define_data():
    """Define the problem data."""
    num_stages = 9
    core_types = ["GPU", "Big", "Medium", "Little"]
    type_value = {"GPU": 0, "Big": 1, "Medium": 2, "Little": 3}

    stage_timings = [
        [8.87154, 7.89115, 5.6832, 19.7066],
        [4.95056, 19.1689, 12.4102, 23.2933],
        [4.36582, 3.94475, 2.99475, 10.3515],
        [2.85043, 9.75395, 6.49812, 11.7059],
        [2.3015, 2.6454, 1.66993, 5.86698],
        [2.24134, 2.4999, 1.69134, 5.80243],
        [2.32418, 2.40803, 1.68515, 5.88757],
        [1.68159, 5.2261, 3.5159, 6.29989],
        [0.247123, 0.0579614, 0.0276213, 0.182615],
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


def print_stage_assignments(m, x, num_stages, core_types, stage_timings):
    """Print the assignment of stages to core types."""
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                print(
                    f"Stage {i}: core type {c} with time {stage_timings[i][core_types.index(c)]}"
                )


def print_chunk_summary(m, x, num_stages, core_types, stage_timings):
    """Print a summary of chunks with their core types and times."""
    print("\nMath model summary:")
    current_chunk = 0
    current_core_type = None
    chunk_time = 0.0

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
                    current_chunk += 1
                    current_core_type = c
                    chunk_time = stage_timings[i][core_types.index(c)]
                break

    # Print the last chunk
    if current_core_type is not None:
        print(f"chunk {current_chunk} ({current_core_type}): {chunk_time:.5f} ms")


def solve_optimization_problem():
    """Solve the optimization problem and display the solution."""
    # Initialize data
    num_stages, core_types, type_value, stage_timings = define_data()

    # Create optimizer
    opt = Optimize()

    # Create decision variables
    x = create_decision_variables(num_stages, core_types)

    # Add constraints
    add_assignment_constraints(opt, x, num_stages, core_types)
    T_chunk = add_chunk_time_constraint(opt, x, core_types, num_stages, stage_timings)
    add_contiguity_constraints(opt, x, core_types, num_stages)

    # Solve and display results
    if opt.check() == sat:
        m = opt.model()
        print(
            f"Optimal chunk time: {m[T_chunk]} = {float(m[T_chunk].as_fraction())} ms"
        )
        print_stage_assignments(m, x, num_stages, core_types, stage_timings)
        print_chunk_summary(m, x, num_stages, core_types, stage_timings)
    else:
        print("No solution found.")


if __name__ == "__main__":
    solve_optimization_problem()
