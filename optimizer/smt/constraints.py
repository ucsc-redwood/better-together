"""SMT constraints and optimization setup for schedule optimization."""

from z3 import And, Bool, If, Implies, Not, Optimize, Or, Real, RealVal, Sum


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


COST_MODELS = ("per-stage", "range-hot", "range-idle")


def idle_price(t0, t1, period):
    """A GPU chunk's time under range-idle (design D3) when the pipeline period is
    ``period``: the chunk idles g = period - t, and t = t0 + k * min(g, t0) with
    k = (t1 - t0) / t0, so t = min(t1, (t0 + k * period) / (1 + k)) -- t0 at the bottleneck
    (no idle), rising to t1 with a gap of t0 or more. t1 < t0 (noise) prices t0."""
    k = max(t1 - t0, 0.0) / t0 if t0 > 0 else 0.0
    return min(t0 * (1 + k), (t0 + k * period) / (1 + k))


def add_chunk_time_constraint(
    opt,
    x,
    core_types,
    num_stages,
    stage_timings,
    minimize_mode="gapness",
    overhead=None,
    shared_core_types=(),
    cost_model="per-stage",
    range_costs=None,
):
    """Add constraints for the chunk times and optimize for minimal gap between max and min chunk times.

    overhead: optional {core_type: (per_chunk_ms, per_stage_ms)} framework-overhead
    constants (smt.overhead). A chunk's cost is then
        sum(stage kernel times) + per_chunk_ms + n_stages * per_stage_ms.
    Adding the constants to EVERY segment (not just maximal chunks) is sound for the
    <= T_max side: a sub-segment costs no more than its enclosing chunk (stage times
    and the constants are >= 0), so the binding constraint is still the maximal chunk.

    shared_core_types: PUs allowed in several chunks (allow_gpu_multi_chunk). One device
    serves all of them, so their chunks time-share it: the SUM over the PU's maximal
    segments, each with its per-chunk overhead, must also be <= T_max -- a lower bound on
    the pipeline period (runtime/gpu-chunks spec "Optimizer models GPU time-sharing").

    cost_model / range_costs (optimizer/cost-model spec; design D3/D4): with "range-hot" or
    "range-idle" and range_costs {(i, j): (t0, t1)} (0-based, inclusive; measured whole GPU
    chunks), a GPU chunk covering exactly [i, j] costs its measured range time, without
    stage sums or GPU overhead: t0 for range-hot; for range-idle t0 interpolated toward t1
    by the chunk's idle time per period (idle_price), at t0 when the GPU has several chunks.
    A measured time need not be monotone in the range, so the GPU's constraints then bind
    MAXIMAL segments (the actual chunks) only; a GPU range without a record keeps its
    per-stage price there. The GPU's time-sharing bound becomes the sum of its chunks' t0.
    CPU tiers, and every PU under "per-stage" or without range_costs, are encoded as before.

    Returns (T_max, T_min, Gapness, gpu_prices): gpu_prices {(i, j): z3 term} gives the
    price of each range-priced GPU segment, to evaluate in a model ({} when not used).
    """
    if cost_model not in COST_MODELS:
        raise ValueError(f"Invalid cost model: {cost_model}")
    range_costs = range_costs if cost_model != "per-stage" else None
    # Define the maximum and minimum chunk time variables
    T_max = Real("T_max")
    T_min = Real("T_min")
    Gapness = Real("Gapness")

    opt.add(T_max > 0)
    opt.add(T_min > 0)

    def chunk_starts(c):  # the number of chunks PU c has
        return Sum(
            [
                If(And(x[(k, c)], True if k == 0 else Not(x[(k - 1, c)])), 1, 0)
                for k in range(num_stages)
            ]
        )

    range_pu = "GPU" if range_costs and "GPU" in core_types else None
    # Range-priced GPU segments: t0, the price to report (a term to evaluate in a model), and
    # the constraint "price >= T_min". The price is never asserted as such: under range-idle
    # min(t1, (t0 + k*T)/(1 + k)) <= T holds exactly when t0 <= T, and min(a, b) >= T_min
    # when both are, so every constraint stays linear (an If on T_max made z3 3-8x slower).
    gpu_hot, gpu_prices, gpu_above_min = {}, {}, {}
    several = False
    if range_pu and cost_model == "range-idle" and range_pu in shared_core_types:
        several = chunk_starts(range_pu) > 1
    for (i, j), (t0, t1) in (range_costs or {}).items() if range_pu else ():
        hot = RealVal(t0)
        gpu_hot[(i, j)] = hot
        if cost_model == "range-hot":
            gpu_prices[(i, j)], gpu_above_min[(i, j)] = hot, hot >= T_min
            continue
        k = max(t1 - t0, 0.0) / t0 if t0 > 0 else 0.0
        lin, top = (hot + RealVal(k) * T_max) / RealVal(1 + k), RealVal(t0 * (1 + k))
        idle = If(lin < top, lin, top)
        idle_above = And(lin >= T_min, top >= T_min)
        gpu_prices[(i, j)] = If(several, hot, idle)
        gpu_above_min[(i, j)] = If(several, hot >= T_min, idle_above)

    # For each PU and every contiguous segment of stages, if the entire segment is handled by PU c,
    # then the sum of timings for that segment must be <= T_max and >= T_min (if it's assigned).
    for c in core_types:
        oh_chunk, oh_stage = (overhead or {}).get(c, (0.0, 0.0))
        for i in range(num_stages):
            for j in range(i, num_stages):
                # Build the condition: all stages k in [i, j] are assigned PU c.
                segment_assigned = And([x[(k, c)] for k in range(i, j + 1)])

                # Segment cost = kernel sums + the framework-overhead terms.
                n_seg = j - i + 1
                seg_sum = Sum(
                    [RealVal(stage_timings[k][core_types.index(c)]) for k in range(i, j + 1)]
                    + [RealVal(oh_chunk + n_seg * oh_stage)]
                )

                # Check if this is a maximal segment (no stages before i or after j assigned to c)
                is_start = i == 0 or Not(x[(i - 1, c)])
                is_end = j == num_stages - 1 or Not(x[(j + 1, c)])
                is_maximal_segment = And(segment_assigned, is_start, is_end)

                if c == range_pu:
                    # Range models: the actual chunk alone is priced (design D4).
                    if (i, j) in gpu_hot:
                        opt.add(Implies(is_maximal_segment, gpu_hot[(i, j)] <= T_max))
                        opt.add(Implies(is_maximal_segment, gpu_above_min[(i, j)]))
                    else:  # no record: its per-stage price
                        opt.add(Implies(is_maximal_segment, seg_sum <= T_max))
                        opt.add(Implies(is_maximal_segment, seg_sum >= T_min))
                    continue

                # Add the implication: if the segment is uniformly assigned c, then seg_sum <= T_max.
                opt.add(Implies(segment_assigned, seg_sum <= T_max))

                # Also, if this is an actual chunk (i.e., it's a maximal contiguous segment assigned to c),
                # then its time should be >= T_min
                # If this is a maximal segment, its time should be >= T_min
                opt.add(Implies(is_maximal_segment, seg_sum >= T_min))

    for c in shared_core_types:
        oh_chunk, oh_stage = (overhead or {}).get(c, (0.0, 0.0))
        col = core_types.index(c)
        if c == range_pu:
            # The sum over the PU's chunks (maximal segments) of their t0 (design D4); a
            # range without a record adds its per-stage price.
            chunk_cost = []
            for i in range(num_stages):
                for j in range(i, num_stages):
                    maximal = And(
                        [x[(k, c)] for k in range(i, j + 1)]
                        + ([Not(x[(i - 1, c)])] if i > 0 else [])
                        + ([Not(x[(j + 1, c)])] if j < num_stages - 1 else [])
                    )
                    fallback = RealVal(
                        sum(stage_timings[k][col] for k in range(i, j + 1))
                        + oh_chunk
                        + (j - i + 1) * oh_stage
                    )
                    chunk_cost.append(If(maximal, gpu_hot.get((i, j), fallback), RealVal(0)))
            opt.add(Sum(chunk_cost) <= T_max)
            continue
        stage_cost = Sum(
            [
                If(x[(k, c)], RealVal(stage_timings[k][col] + oh_stage), RealVal(0))
                for k in range(num_stages)
            ]
        )
        opt.add(stage_cost + RealVal(oh_chunk) * chunk_starts(c) <= T_max)

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

    return T_max, T_min, Gapness, gpu_prices


def add_contiguity_constraints(opt, x, core_types, num_stages, exempt_core_types=()):
    """Add constraints to ensure each PU appears in one continuous block. PUs in
    exempt_core_types (allow_gpu_multi_chunk: the GPU) may appear in several."""
    for c in core_types:
        if c in exempt_core_types:
            continue
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
