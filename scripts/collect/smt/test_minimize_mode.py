#!/usr/bin/env python3
"""Regression test: minimize_mode must actually change the z3 objective.

Before the wiring fix, `solve_optimization_problem` ignored minimize_mode and always
minimized gapness, so `tmax` and `gapness` produced identical schedules. This builds a
fixture where the two objectives DISAGREE and asserts they now differ.

Fixture (9 stages, [Little, Medium, Big, GPU] ms): Little/Medium are huge (never
picked); Big is cheap on every stage except stage 0; GPU is cheap on every stage
except stage 8. So:
  - any single-PU assignment costs ~108 ms (gap 0, what gapness loves)
  - the makespan-optimal answer is a SPLIT (GPU does 0..7, Big does 8) at ~8 ms, gap 7
Hence tmax (minimize T_max) must reach ~8 ms while gapness (minimize T_max - T_min)
stays at ~108 ms.  Run:  uv run python scripts/collect/smt/test_minimize_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smt.solver import solve_optimization_problem  # noqa: E402

HUGE = 1000.0
# [Little, Medium, Big, GPU] per stage (app="cifar-dense" -> 9 stages)
STAGES = [
    [HUGE, HUGE, (100.0 if s == 0 else 1.0), (100.0 if s == 8 else 1.0)]
    for s in range(9)
]


def primary_makespan(mode):
    # solutions[0] is the optimizer's OPTIMUM for the chosen objective (later solutions
    # are blocked alternatives, so a min-over-all would hide the objective difference).
    sols = solve_optimization_problem(STAGES, num_solutions=20, app_name="cifar-dense",
                                      minimize_mode=mode)
    return sols[0]["metrics"]["max_time"]


def main():
    tmax = primary_makespan("max_time")
    gap = primary_makespan("gapness")
    print(f"primary-solution makespan: tmax={tmax:.2f} ms  gapness={gap:.2f} ms")

    # The tmax optimum minimizes makespan (~5 ms split); the gapness optimum minimizes
    # the gap and lands on a single-PU assignment (~108 ms). Before the wiring fix both
    # returned the gapness optimum, so this assertion failed.
    assert tmax < gap - 1.0, f"minimize_mode not effective: tmax={tmax} not < gapness={gap}"
    assert tmax < 20.0, f"tmax did not find the split makespan: {tmax}"
    assert gap > 50.0, f"gapness optimum unexpectedly low: {gap}"
    print("PASS: minimize_mode changes the objective (tmax makespan < gapness makespan)")


if __name__ == "__main__":
    main()
