"""SMT constraints and optimization setup for schedule optimization."""

from z3 import And, Bool, Implies, Not, Optimize, Or, Real, RealVal, Sum


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


def add_availability_constraints(opt, x, core_types, num_stages, stage_timings, unavailable):
    """Forbid assigning a stage to a PU the device physically lacks (cost == the
    ``UNAVAILABLE`` sentinel). This makes absent hardware *structurally* impossible to
    select, protecting every objective. A cost penalty alone does NOT: a load-balancing
    objective (gapness) finds a zero-gap schedule on an absent tier just as cheaply as on
    a real one, then z3 emits an unrunnable schedule (review #2)."""
    if not stage_timings:
        return
    for i in range(num_stages):
        for c in core_types:
            if stage_timings[i][core_types.index(c)] >= unavailable:
                opt.add(Not(x[(i, c)]))


def add_chunk_time_constraint(
    opt, x, core_types, num_stages, stage_timings, minimize_mode="gapness"
):
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
                    [RealVal(stage_timings[k][core_types.index(c)]) for k in range(i, j + 1)]
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

    if minimize_mode == "gapness":
        # Makespan is the PRIMARY objective; the gap is only a lexicographic tie-breaker
        # among equal-makespan schedules (z3 Optimize is lexicographic in objective order).
        # Minimizing the gap ALONE is degenerate: a slow single-PU chunk has gap=0 and would
        # win over any pipelined schedule, so z3 systematically picked the slowest single-PU
        # assignment and ignored pipelining (review #1).
        opt.minimize(T_max)
        opt.minimize(Gapness)
    elif minimize_mode == "max_time":
        opt.minimize(T_max)
    else:
        raise ValueError(f"Invalid minimize mode: {minimize_mode}")

    return T_max, T_min, Gapness


def add_contiguity_constraints(opt, x, core_types, num_stages):
    """Add constraints to ensure each PU appears in one continuous block."""
    for c in core_types:
        for i in range(num_stages):
            for j in range(i + 1, num_stages):
                for k in range(j + 1, num_stages):
                    opt.add(Implies(And(x[(i, c)], x[(k, c)]), x[(j, c)]))


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


def create_optimizer():
    """Create and return a Z3 optimizer instance."""
    return Optimize()
