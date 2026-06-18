"""Main SMT solver for schedule optimization."""

from z3 import sat

from .constraints import (
    create_decision_variables,
    add_assignment_constraints,
    add_chunk_time_constraint,
    add_contiguity_constraints,
    block_solution,
    create_optimizer,
)
from .data_loader import define_data
from .solution_analyzer import (
    get_solution_representation,
    get_detailed_solution,
)


# baseline_data,
def solve_optimization_problem(
    stage_timings, num_solutions=30, app_name=None, minimize_mode="gapness"
):
    """Solve the optimization problem and return the solutions.

    minimize_mode: "gapness" (minimize T_max - T_min, load balance) or "max_time"
    (minimize T_max, the pipeline makespan). Was hardcoded to gapness -- the caller's
    choice never reached the solver, so every "tmax" schedule was a gapness clone.
    """
    # Initialize data
    num_stages, core_types, stage_timings_data = define_data(stage_timings, app_name)

    # Prepare a list to hold up to num_solutions solutions
    top_solutions = []
    detailed_solutions = []

    # Create optimizer
    opt = create_optimizer()

    # Create decision variables
    x = create_decision_variables(num_stages, core_types)

    # Add constraints
    add_assignment_constraints(opt, x, num_stages, core_types)

    T_max, T_min, Gapness = add_chunk_time_constraint(
        opt, x, core_types, num_stages, stage_timings_data, minimize_mode
    )

    add_contiguity_constraints(opt, x, core_types, num_stages)

    objective = (
        "T_max (the pipeline makespan)"
        if minimize_mode == "max_time"
        else "the gap between max and min chunk times"
    )
    print(f"\nOptimization approach: Minimizing {objective}")
    print("---------------------------------------------------------------------")

    # Find up to num_solutions solutions
    solution_count = 0

    while solution_count < num_solutions and opt.check() == sat:
        m = opt.model()
        max_time = float(m[T_max].as_fraction())
        min_time = float(m[T_min].as_fraction())
        gapness_value = float(m[Gapness].as_fraction())

        # Get detailed solution for JSON output
        detailed_solution = get_detailed_solution(
            m, x, num_stages, core_types, stage_timings_data
        )
        detailed_solution["solution_id"] = solution_count + 1
        detailed_solution["metrics"]["max_time"] = max_time
        detailed_solution["metrics"]["min_time"] = min_time
        detailed_solution["metrics"]["gapness"] = gapness_value

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

        # Print the solutions again but sorted by max time
        print("\n=== Summary of All Solutions Sorted by Max Time ===")
        sorted_solutions = sorted(top_solutions, key=lambda x: x[1], reverse=False)
        for i, (gapness, max_time, _) in enumerate(sorted_solutions):
            solution_uid = detailed_solutions[i]["uid"]
            print(
                f"Solution {i + 1}: Gap = {gapness:.2f} ms, Max time = {max_time:.2f} ms, UID: {solution_uid}"
            )

    return detailed_solutions
