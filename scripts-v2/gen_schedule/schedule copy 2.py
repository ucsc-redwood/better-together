from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat, Implies, And

# Data
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

# Create an optimizer instance.
opt = Optimize()

# ---- Decision Variables ----
# Decision Variables: x[i, c] is True if stage i is assigned to core type c.
# ----------------------------

x = {}
for i in range(num_stages):
    for c in core_types:
        x[(i, c)] = Bool(f"x_{i}_{c}")


# ---- Constraints ----

# ------------------------------------------------------------
# Each stage must be assigned exactly one PU.
# ------------------------------------------------------------

for i in range(num_stages):
    # At least one PU must be chosen.
    opt.add(Or([x[(i, c)] for c in core_types]))
    # And at most one PU can be chosen.
    for j in range(len(core_types)):
        for k in range(j + 1, len(core_types)):
            opt.add(Or(Not(x[(i, core_types[j])]), Not(x[(i, core_types[k])])))

# Define the "chunk time" variable (to be minimized) as a real number.
T_chunk = Real("T_chunk")
opt.add(T_chunk > 0)

# ---- Enforce the chunk time constraint for contiguous segments ----
# ------------------------------------------------------------
# For each PU and every contiguous segment of stages, if the entire segment is handled by PU c,
# then the sum of timings for that segment must be <= T_chunk.
# ------------------------------------------------------------
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

# (Optional) You might keep the contiguity constraints from before if you want to ensure that if a PU is used,
# it appears in one continuous block. For that, one common pattern is:
for c in core_types:
    for i in range(num_stages):
        for j in range(i + 1, num_stages):
            for k in range(j + 1, num_stages):
                opt.add(Implies(And(x[(i, c)], x[(k, c)]), x[(j, c)]))


# Prepare a list to hold up to 10 solutions.
top_solutions = []
num_solutions = 10


# Solve and display the solution.

if opt.check() == sat:
    m = opt.model()
    print(f"Optimal chunk time: {m[T_chunk]} = {float(m[T_chunk].as_fraction())} ms")
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                print(
                    f"Stage {i}: core type {c} with time {stage_timings[i][core_types.index(c)]}"
                )

    # Print chunk summary
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


else:
    print("No solution found.")
