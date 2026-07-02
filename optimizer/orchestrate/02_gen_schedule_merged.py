#!/usr/bin/env python3
"""
Simplified schedule generation script that uses modular SMT components.

This script orchestrates the schedule optimization process by:
1. Loading and processing CSV data
2. Getting baseline information
3. Running the SMT solver
4. Outputting results
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smt.baselines import get_baseline_for_config
from smt.bt_vocab import CORE_TYPES
from smt.data_loader import load_stage_timings
from smt.overhead import load_overhead, resolve_for_solver
from smt.solution_analyzer import dump_solutions_as_json, reprice_solution
from smt.solver import solve_optimization_problem

from orchestrate.case import Case, table_to_scenario


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve scheduling optimization problem using data from a CSV file."
    )

    # Input folder: the canonical JSONL profiling store, structured as
    #   <root>/<device>/<app>/<backend_long>/<scenario>/run-*.jsonl
    # e.g. data/profiling/jetson/cifar-dense/cuda/isolated/run-001.jsonl
    # (scenario = isolated for --table_type isolated, interference for btpm).
    # Legacy layout, kept for reference:
    #
    # data/bm_logs/
    # ├── 3A021JEHN02756
    # │   ├── cifar-dense
    # │   │   └── vk
    # │   │       ├── ...
    # │   │       ├── btpm.csv
    # │   │       └── isolated.csv
    #
    parser.add_argument(
        "--profiling_root",
        type=str,
        help="Root of the canonical JSONL profiling store "
        "(<root>/<device>/<app>/<backend>/<scenario>/run-*.jsonl)",
        required=True,
    )

    # Basic target information
    #
    parser.add_argument("--device", required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--backend", required=True, choices=["vk", "cu"])

    # which table and optimization mode to use
    #
    parser.add_argument(
        "--table_type",
        type=str,
        choices=["isolated", "btpm"],
        required=True,
        help="Mode to select CSV file: 'isolated' for isolated.csv or 'btpm' for btpm.csv",
    )
    parser.add_argument(
        "--minimize_mode",
        type=str,
        choices=["gapness", "tmax"],
        required=True,
        help="Mode to minimize: 'gapness' for minimizing the gap between max and min chunk times, 'tmax' for minimizing the max chunk time",
    )

    parser.add_argument(
        "-n",
        "--num_solutions",
        type=int,
        help="Number of solutions to find",
        required=True,
    )

    # Output folder
    # The output JSON files are in the following structure:
    #
    # data/schedules/
    # ├── 3A021JEHN02756
    # │   ├── cifar-dense
    # │   │   └── vk
    # │   │       ├── schedules_btpm_gapness.json
    # │   │       ├── schedules_btpm_tmax.json
    # │   │       ├── schedules_isolated_gapness.json
    # │   │       └── schedules_isolated_tmax.json
    #
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Root path to write the JSON output file (optional)",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
        default=False,
    )
    parser.add_argument(
        "--dvfs-floor",
        action="store_true",
        default=False,
        help="conservative guard: clamp a too-cheap-under-load GPU interference cell up to "
        "its isolated value. OFF by default -- the chaotic real environment (incl. the GPU "
        "boosting when kept busy) is what we want to capture, not sanitize.",
    )
    parser.add_argument(
        "--no-overhead",
        action="store_true",
        default=False,
        help="solve WITHOUT the fitted per-chunk framework-overhead constants "
        "(<profiling_root>/<device>/overhead.json, fitted by analysis/fit_overhead.py). "
        "Default applies them when the file exists.",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    backend = args.backend
    device = args.device
    app = args.app
    table_type = args.table_type
    minimize_mode = args.minimize_mode
    verbose = args.verbose
    case = Case(device, app, backend)  # the data-layout single source of truth

    # Get baseline data for this configuration (derived from the same profiling store).
    baseline_data = get_baseline_for_config(device, app, backend, args.profiling_root)
    if baseline_data:
        if verbose:
            print(f"Baseline data for {device}/{app}/{backend}:")
            print(f"  CPU (OpenMP): {baseline_data.get('omp')} ms")
            print(f"  GPU ({backend}): {baseline_data.get(backend)} ms")
            print(f"  Fastest: {baseline_data.get('fastest')} ms")
    else:
        print(f"No baseline data available for {device}/{app}/{backend}")

    # Profiling-store cell for this table_type (isolated.json -> isolated, btpm -> interference)
    scenario = table_to_scenario(table_type)
    prof_dir = case.profiling_dir(args.profiling_root, scenario)

    # Skip a combination with no profiling data (rather than failing the whole sweep).
    if not os.path.isdir(prof_dir):
        print(f"Warning: no profiling data at {prof_dir}. Skipping this combination.")
        sys.exit(0)
    else:
        print(f"Loading profiling data from: {prof_dir}")
        print(f"Using table type: {table_type}")
        print(f"Using minimize mode: {minimize_mode}")

    # Output path for schedule JSON
    if args.output_folder:
        out_path = case.schedule_path(args.output_folder, table_type, minimize_mode)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        print("No output folder specified, skipping")
        sys.exit(0)

    try:
        stage_timings, use_cuda = load_stage_timings(
            args.profiling_root,
            device,
            app,
            backend,
            scenario,
            verbose=verbose,
            dvfs_floor=args.dvfs_floor,
        )

        # Store which GPU backend was used
        gpu_backend = "gpu_cuda" if use_cuda else "gpu_vulkan"

        # Fitted per-chunk framework-overhead constants (see smt/overhead.py). Missing
        # file (never fitted) or --no-overhead -> the plain kernel-sum cost model.
        overhead = None
        if not args.no_overhead:
            raw_overhead = load_overhead(args.profiling_root, device)
            if raw_overhead:
                overhead = resolve_for_solver(raw_overhead, CORE_TYPES, gpu_backend)
                print(
                    "Applying framework-overhead constants (per-chunk, per-stage ms): "
                    + ", ".join(f"{c}=({v[0]:.3f},{v[1]:.3f})" for c, v in overhead.items())
                )

        # Solve the optimization problem. Map the CLI token "tmax" to the solver's
        # "max_time" (constraints.py uses the latter); "gapness" passes through.
        solver_mode = "max_time" if minimize_mode == "tmax" else minimize_mode
        # GPU chunks get their schema-required "hardware" stamped inside the solver
        # (threaded through to get_detailed_solution), so the dump is self-validating.
        solutions = solve_optimization_problem(
            stage_timings, args.num_solutions, app, solver_mode, gpu_backend, overhead
        )

        # Union-candidate hedge: ALSO solve with the plain kernel-sum model and append
        # any assignments the overhead model didn't propose, re-priced under the
        # overhead model so the file carries one consistent prediction semantics. The
        # measured top-K sweep (03) then picks the true winner -- robustness against
        # either model's blind spots (the fitted constants come mostly from large
        # chunks and can over-penalize tiny ones, e.g. phone x tree).
        if overhead is not None:

            def signature(sol):
                return tuple(
                    (c["core_type"], c["start_stage"], c["end_stage"]) for c in sol["chunks"]
                )

            seen = {signature(s) for s in solutions}
            plain = solve_optimization_problem(
                stage_timings, args.num_solutions, app, solver_mode, gpu_backend, None
            )
            added = 0
            for sol in plain:
                if signature(sol) not in seen:
                    seen.add(signature(sol))
                    solutions.append(
                        reprice_solution(sol, CORE_TYPES, stage_timings, overhead)
                    )
                    added += 1
            solutions.sort(key=lambda s: s["metrics"].get("max_time", float("inf")))
            for i, sol in enumerate(solutions):
                sol["solution_id"] = i + 1
            print(f"Union sweep: +{added} plain-model candidates (re-priced), "
                  f"{len(solutions)} total, sorted by predicted makespan")

        # Output the solutions
        dump_solutions_as_json(solutions, baseline_data, "pretty", out_path)

    except Exception as e:
        print(f"Error processing {prof_dir}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
