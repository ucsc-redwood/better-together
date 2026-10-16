#!/usr/bin/env python3
"""Tests for the opt-in allow_gpu_multi_chunk solver mode (runtime/gpu-chunks spec,
"Optimizer models GPU time-sharing"; design C4).

Toy profile (11 stages, app="cifar-dense"): the GPU is fast on stages 1-2 and 6-11, the
Big tier only on 3-5. One contiguous GPU block cannot take both GPU-friendly runs, so the
default mode's best period is far above 8 ms. Letting the GPU take two chunks (1-2 and
6-11) around Big 3-5 is better -- but only if their summed GPU time (2 + 6 = 8 ms) is what
bounds the period, since one device serves both.

Run:  uv run python -m pytest optimizer/tests/test_gpu_multi_chunk.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smt.solution_analyzer import validate_against_schema  # noqa: E402
from smt.solver import solve_optimization_problem  # noqa: E402

HUGE = 1000.0
# [Little, Medium, Big, GPU] per stage.
STAGES = [
    [HUGE, HUGE, 1.0 if 2 <= s <= 4 else 10.0, 10.0 if 2 <= s <= 4 else 1.0] for s in range(11)
]


def _solve(allow, overhead=None, n=20):
    return solve_optimization_problem(
        STAGES,
        num_solutions=n,
        app_name="cifar-dense",
        minimize_mode="max_time",
        gpu_backend="gpu_vulkan",
        overhead=overhead,
        allow_gpu_multi_chunk=allow,
    )


def _gpu_chunks(sol):
    return [c for c in sol["chunks"] if c["core_type"] == "GPU"]


def test_default_mode_never_emits_two_gpu_chunks():
    sols = _solve(allow=False)
    assert sols
    for sol in sols:
        per_pu = [c["core_type"] for c in sol["chunks"]]
        assert len(per_pu) == len(set(per_pu)), f"a PU repeats in {per_pu}"
        assert len(_gpu_chunks(sol)) <= 1
    assert sols[0]["metrics"]["max_time"] > 8.0 + 1e-6  # cannot reach the 2-GPU-chunk period


def test_multi_chunk_period_is_at_least_the_gpu_sum():
    sols = _solve(allow=True)
    best = sols[0]
    gpu = _gpu_chunks(best)
    assert [(c["start_stage"], c["end_stage"]) for c in gpu] == [(1, 2), (6, 11)]
    assert abs(best["metrics"]["max_time"] - 8.0) < 1e-6
    for sol in sols:
        gpu_sum = sum(c["time"] for c in _gpu_chunks(sol))
        assert sol["metrics"]["max_time"] >= gpu_sum - 1e-6, sol["uid"]
    validate_against_schema(sols)  # repeated GPU chunks are schema-valid


def test_multi_chunk_cost_counts_every_chunk_overhead():
    """With a 5 ms per-chunk GPU overhead, every GPU chunk carries its own 5 ms, and the
    predicted period covers the GPU's summed time, overheads included."""
    oh = {"Little": (0, 0), "Medium": (0, 0), "Big": (0, 0), "GPU": (5.0, 0.0)}
    sols = _solve(allow=True, overhead=oh)
    best = sols[0]
    assert len(_gpu_chunks(best)) == 2
    for sol in sols:
        gpu = _gpu_chunks(sol)
        for c in gpu:
            kernel = sum(STAGES[k][3] for k in range(c["start_stage"] - 1, c["end_stage"]))
            assert abs(c["time"] - (kernel + 5.0)) < 1e-6
        assert sol["metrics"]["max_time"] >= sum(c["time"] for c in gpu) - 1e-6, sol["uid"]


if __name__ == "__main__":
    test_default_mode_never_emits_two_gpu_chunks()
    test_multi_chunk_period_is_at_least_the_gpu_sum()
    test_multi_chunk_cost_counts_every_chunk_overhead()
    print("PASS: GPU multi-chunk mode is opt-in and prices time-sharing")
