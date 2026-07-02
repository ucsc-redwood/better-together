#!/usr/bin/env python3
"""Tests for the per-chunk framework-overhead term (P2 cost-model fix).

Without the term, z3 sums per-stage kernel times only, so a many-stage GPU chunk
looks free of dispatch cost and the solver picks pipelines that lose on tiny apps
(tree x VK in the 2026-07-02 baseline). With the term, a chunk on PU c costs
    sum(stage times) + per_chunk_ms(c) + n_stages * per_stage_ms(c).

Run:  uv run python -m pytest optimizer/tests/test_overhead.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smt.overhead import resolve_for_solver  # noqa: E402
from smt.solution_analyzer import reprice_solution  # noqa: E402
from smt.solver import solve_optimization_problem  # noqa: E402

CORE_TYPES = ["Little", "Medium", "Big", "GPU"]
HUGE = 1000.0

# 9 stages (app="cifar-dense"): GPU kernels are 1 ms, Big is 2 ms, others absent-ish.
STAGES = [[HUGE, HUGE, 2.0, 1.0] for _ in range(9)]


def _best(overhead):
    sols = solve_optimization_problem(
        STAGES, num_solutions=1, app_name="cifar-dense", minimize_mode="max_time",
        gpu_backend="gpu_vulkan", overhead=overhead,
    )
    return sols[0]


def test_overhead_shifts_the_optimum():
    """A large per-stage GPU dispatch tax must push work OFF the GPU: with zero
    overhead the all-GPU schedule wins (9 ms); with a 5 ms/stage GPU tax an all-GPU
    chunk costs 9 + 5*9 = 54 ms, so the optimum must abandon GPU-only."""
    free = _best(None)
    assert free["metrics"]["max_time"] <= 9.001  # all-GPU, kernel sums only

    taxed = _best({"Little": (0, 0), "Medium": (0, 0), "Big": (0, 0), "GPU": (5.0, 5.0)})
    gpu_stages = sum(
        c["end_stage"] - c["start_stage"] + 1
        for c in taxed["chunks"]
        if c["core_type"] == "GPU"
    )
    assert gpu_stages < 9, "solver kept the all-GPU schedule despite the dispatch tax"


def test_predicted_chunk_time_includes_overhead():
    """The emitted chunk 'time' must match the cost the solver optimized (kernel sum
    + per-chunk + n*per-stage), so predicted-vs-measured comparisons stay honest."""
    oh = {"Little": (0, 0), "Medium": (0, 0), "Big": (0, 0), "GPU": (3.0, 0.5)}
    sol = _best(oh)
    for c in sol["chunks"]:
        n = c["end_stage"] - c["start_stage"] + 1
        kernel = {"Big": 2.0, "GPU": 1.0}[c["core_type"]] * n
        oh_chunk, oh_stage = oh[c["core_type"]]
        assert abs(c["time"] - (kernel + oh_chunk + n * oh_stage)) < 1e-6


def test_reprice_matches_overhead_model():
    """A plain-model candidate re-priced under the overhead model must carry the same
    chunk times the solver itself would have predicted (union-sweep consistency)."""
    oh = {"Little": (0, 0), "Medium": (0, 0), "Big": (0, 0), "GPU": (3.0, 0.5)}
    plain = _best(None)  # all-GPU under the plain model
    repriced = reprice_solution(plain, CORE_TYPES, STAGES, oh)
    for c in repriced["chunks"]:
        n = c["end_stage"] - c["start_stage"] + 1
        kernel = {"Big": 2.0, "GPU": 1.0}[c["core_type"]] * n
        oh_chunk, oh_stage = oh[c["core_type"]]
        assert abs(c["time"] - (kernel + oh_chunk + n * oh_stage)) < 1e-6
    # Metrics and uid regenerate to match the new pricing.
    assert repriced["metrics"]["max_time"] > plain["metrics"]["max_time"]
    assert repriced["uid"] != plain["uid"]
    # The assignment itself is untouched.
    assert [(c["core_type"], c["start_stage"], c["end_stage"]) for c in repriced["chunks"]] == [
        (c["core_type"], c["start_stage"], c["end_stage"]) for c in plain["chunks"]
    ]


def test_resolver_maps_classes_and_defaults_to_zero():
    raw = {"cpu": {"per_chunk_ms": 1.0, "per_stage_ms": 0.25}, "gpu_vulkan": {"per_chunk_ms": 2.0}}
    r = resolve_for_solver(raw, CORE_TYPES, "gpu_vulkan")
    assert r["Big"] == (1.0, 0.25) and r["Little"] == (1.0, 0.25)
    assert r["GPU"] == (2.0, 0.0)
    # Solving for the OTHER backend with no fitted class -> zero overhead.
    r_cu = resolve_for_solver(raw, CORE_TYPES, "gpu_cuda")
    assert r_cu["GPU"] == (0.0, 0.0)


if __name__ == "__main__":
    test_overhead_shifts_the_optimum()
    test_predicted_chunk_time_includes_overhead()
    test_resolver_maps_classes_and_defaults_to_zero()
    print("PASS: overhead term shifts optima and is reflected in predicted chunk times")
