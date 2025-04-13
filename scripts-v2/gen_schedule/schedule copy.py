from z3 import Optimize, Bool, Real, Sum, If, Or, Not, RealVal, sat

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

# Create an optimization solver instance
opt = Optimize()

# Decision variables: x[i][c] indicating whether stage i uses core type c
x = {}
for i in range(num_stages):
    for c in core_types:
        x[(i, c)] = Bool(f"x_{i}_{c}")

# Cycle time variable T_cycle must be positive.
T_cycle = Real("T_cycle")
opt.add(T_cycle > 0)

# Constraint: Each stage must be assigned to exactly one core type.
for i in range(num_stages):
    # At least one core must be selected
    opt.add(Or([x[(i, c)] for c in core_types]))
    # And no two core types can be selected simultaneously.
    for j in range(len(core_types)):
        for k in range(j + 1, len(core_types)):
            opt.add(Or(Not(x[(i, core_types[j])]), Not(x[(i, core_types[k])])))

# For each stage, the processing time should be less than or equal to T_cycle.
# This effectively makes T_cycle the maximum processing time over all stages.
for i in range(num_stages):
    # Compute the processing time for stage i as a linear combination.
    stage_time = Sum(
        [
            RealVal(stage_timings[i][core_types.index(c)]) * If(x[(i, c)], 1, 0)
            for c in core_types
        ]
    )
    opt.add(stage_time <= T_cycle)

# Set the objective to minimize T_cycle.
opt.minimize(T_cycle)

# Check for the optimal solution.
if opt.check() == sat:
    m = opt.model()
    print("Optimal cycle time:", m[T_cycle])
    for i in range(num_stages):
        for c in core_types:
            if m.evaluate(x[(i, c)]):
                print(
                    f"Stage {i}: core type {c} with time {stage_timings[i][core_types.index(c)]}"
                )
else:
    print("No solution found.")
