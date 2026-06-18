#!/usr/bin/env python3
"""Regression tests for the z3 objective fixes (code-review 2026-06-18, #1 and #2).

#1 -- the `gapness` objective was degenerate: minimizing `T_max - T_min` alone, a slow
single-PU chunk has gap=0 and wins over any pipelined schedule, so z3 systematically
picked the SLOWEST single-PU assignment. Fix: makespan (`T_max`) is now the primary
objective, gap only a lexicographic tie-breaker -- so `gapness` must now also find the
fast split, not the 108 ms single-PU degenerate optimum.

#2 -- a CPU tier the device lacks is encoded as `UNAVAILABLE`; under the old gap-only
objective z3 could still pick it (gap=0 on an all-absent-tier chunk) and emit an
unrunnable schedule. Fix: absent tiers are now structurally forbidden, so NO objective
can select them.

Run:  uv run python -m pytest optimizer/tests/test_minimize_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smt.solver import solve_optimization_problem  # noqa: E402
from smt.data_loader import UNAVAILABLE  # noqa: E402

HUGE = 1000.0
# [Little, Medium, Big, GPU] per stage (app="cifar-dense" -> 9 stages). Little/Medium are
# huge (never picked); Big is cheap except stage 0; GPU is cheap except stage 8. So any
# single-PU schedule costs ~108 ms (gap 0), while the makespan-optimal SPLIT (GPU 0..7,
# Big 8) costs ~8 ms (gap ~7). The degenerate old gapness loved the 108 ms gap-0 answer.
STAGES = [
    [HUGE, HUGE, (100.0 if s == 0 else 1.0), (100.0 if s == 8 else 1.0)]
    for s in range(9)
]


def _primary_makespan(mode):
    # solutions[0] is the optimizer's OPTIMUM for the chosen objective.
    sols = solve_optimization_problem(STAGES, num_solutions=20, app_name="cifar-dense",
                                      minimize_mode=mode)
    return sols[0]["metrics"]["max_time"]


def test_max_time_finds_split():
    """max_time mode reaches the ~8 ms split, not a ~108 ms single-PU schedule."""
    assert _primary_makespan("max_time") < 20.0


def test_gapness_is_not_degenerate():
    """gapness mode must ALSO reach the fast split now that makespan is primary -- it must
    NOT return the slow single-PU gap-0 optimum (~108 ms) the degenerate objective picked."""
    gap_makespan = _primary_makespan("gapness")
    assert gap_makespan < 20.0, (
        f"gapness still degenerate: primary makespan {gap_makespan:.1f} ms "
        f"(a single-PU gap-0 schedule); expected the ~8 ms split"
    )


def test_absent_tier_never_assigned():
    """A tier encoded as UNAVAILABLE must never appear in any chunk, under EITHER objective
    (structural forbid, not just a cost penalty)."""
    # Little is absent (UNAVAILABLE) on every stage; Medium/Big/GPU are real and cheap.
    stages = [[UNAVAILABLE, 2.0, 3.0, 1.0] for _ in range(9)]
    for mode in ("gapness", "max_time"):
        sols = solve_optimization_problem(stages, num_solutions=10,
                                          app_name="cifar-dense", minimize_mode=mode)
        for sol in sols:
            cores = [chunk["core_type"] for chunk in sol["chunks"]]
            assert "Little" not in cores, (
                f"{mode}: schedule assigned the absent 'Little' tier: {cores}"
            )


if __name__ == "__main__":
    test_max_time_finds_split()
    test_gapness_is_not_degenerate()
    test_absent_tier_never_assigned()
    print("PASS: gapness non-degenerate + absent tiers structurally forbidden")
