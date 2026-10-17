#!/usr/bin/env python3
"""The solver's GPU cost models (optimizer/cost-model spec; design D3/D4).

  per-stage   every chunk = sum of its stages + fitted overhead (today's model, the default)
  range-hot   a GPU chunk with a range record = its back-to-back whole-chunk time t0
  range-idle  ... t0 raised toward the after-idle time t1 by its idle time per period

Toy tables use app "tree" (7 stages); columns are [Little, Medium, Big, GPU].

Run:  uv run pytest -q optimizer/tests/test_cost_model.py
"""

import json
import os
import subprocess
import sys

import pytest
from smt.constraints import idle_price
from smt.data_loader import UNAVAILABLE
from smt.solution_analyzer import reprice_solution
from smt.solver import solve_optimization_problem
from test_range_store import APP, DEVICE, add_ranges, store  # noqa: F401  (store: fixture)

HUGE = 1000.0
CORE_TYPES = ["Little", "Medium", "Big", "GPU"]
GEN = os.path.join(os.path.dirname(__file__), "..", "orchestrate", "02_gen_schedule_merged.py")


def solve(stages, cost_model="per-stage", ranges=None, n=10, overhead=None, multi=False):
    return solve_optimization_problem(
        stages,
        num_solutions=n,
        app_name="tree",
        minimize_mode="max_time",
        gpu_backend="gpu_vulkan",
        overhead=overhead,
        allow_gpu_multi_chunk=multi,
        cost_model=cost_model,
        range_costs=ranges,
    )


def shape(sol):
    return [(c["core_type"], c["start_stage"], c["end_stage"]) for c in sol["chunks"]]


def gpu_chunks(sol):
    return [c for c in sol["chunks"] if c["core_type"] == "GPU"]


# --- 3.1: per-stage stays today's model -------------------------------------------------


def candidates(sols):
    """Solutions as a set: z3 orders equal-cost candidates differently from one solve to the
    next within a process, so in-process comparisons ignore that order (the byte-level
    check below runs each solve in a fresh process)."""
    return {
        (tuple(shape(s)), tuple(c["time"] for c in s["chunks"]), s["metrics"]["max_time"])
        for s in sols
    }


def test_per_stage_ignores_range_records():
    # Big and the GPU only: 14 schedules, all enumerated, so no tie straddles the cutoff.
    stages = [[UNAVAILABLE, UNAVAILABLE, 10.0, 2.0] for _ in range(7)]
    ranges = {(i, j): (1.0, 2.0) for i in range(7) for j in range(i, 7)}
    plain = candidates(solve(stages, n=20))
    assert len(plain) == 14
    assert candidates(solve(stages, "per-stage", ranges, n=20)) == plain
    # A range model without range records prices per stage, exactly as before.
    assert candidates(solve(stages, "range-hot", {}, n=20)) == plain
    assert candidates(solve(stages, "range-idle", None, n=20)) == plain


# Today's 02 (before --cost-model): load the per-stage table, solve, dump. No overhead.json
# exists in the test store, so there is no union sweep.
TODAY = """
import sys
sys.path.insert(0, sys.argv[1])
from smt.baselines import get_baseline_for_config
from smt.data_loader import load_stage_timings
from smt.solution_analyzer import dump_solutions_as_json
from smt.solver import solve_optimization_problem
root, device, app, out = sys.argv[2:6]
timings, _ = load_stage_timings(root, device, app, "vk", "isolated")
sols = solve_optimization_problem(timings, 5, app, "max_time", "gpu_vulkan", None, False)
dump_solutions_as_json(sols, get_baseline_for_config(device, app, "vk", root), "pretty", out)
"""


def test_gen_schedule_per_stage_is_unchanged_on_a_store_with_ranges(store, tmp_path):  # noqa: F811
    """02 --cost-model per-stage on a store holding range files writes the JSON today's
    02 wrote (its load -> solve -> dump path), apart from metrics.cost_model/table."""
    add_ranges(store, "isolated")
    out = str(tmp_path / "schedules")
    subprocess.run(
        [sys.executable, GEN, "--profiling_root", store, "--device", DEVICE, "--app", APP]
        + ["--backend", "vk", "--table_type", "isolated", "--minimize_mode", "tmax", "-n", "5"]
        + ["-o", out, "--cost-model", "per-stage"],
        check=True,
        capture_output=True,
    )
    path = os.path.join(out, DEVICE, APP, "vk", "schedules_isolated_tmax.json")
    with open(path) as f:
        new = json.load(f)
    assert {s["metrics"]["cost_model"] for s in new} == {"per-stage"}
    assert {s["metrics"]["table"] for s in new} == {"isolated"}
    for s in new:
        del s["metrics"]["cost_model"], s["metrics"]["table"]

    ref = str(tmp_path / "today.json")
    optimizer = os.path.join(os.path.dirname(__file__), "..")
    subprocess.run(
        [sys.executable, "-c", TODAY, optimizer, store, DEVICE, APP, ref],
        check=True,
        capture_output=True,
    )
    with open(ref) as f:
        assert f.read() == json.dumps(new, indent=2)


# --- 3.2: range-hot ------------------------------------------------------------------------

# GPU per-stage 2 ms, Big 10 ms. Measured ranges: 3 ms per stage, except the whole app
# [1, 7] at 6 ms, cheaper than its own sub-range [1, 3] at 9 ms (non-monotone).
HOT_STAGES = [[HUGE, HUGE, 10.0, 2.0] for _ in range(7)]
HOT_RANGES = {(i, j): (3.0 * (j - i + 1), 4.0 * (j - i + 1)) for i in range(7) for j in range(i, 7)}
HOT_RANGES[(0, 6)] = (6.0, 8.0)
HOT_RANGES[(0, 2)] = (9.0, 12.0)


def test_range_hot_prices_the_chunk_not_its_sub_ranges():
    """All seven stages as one GPU chunk cost [1, 7]'s 6 ms; its dearer sub-range [1, 3]
    (9 ms) must not bound it, as it would if every assigned sub-segment were constrained."""
    best = solve(HOT_STAGES, "range-hot", HOT_RANGES)[0]
    assert shape(best) == [("GPU", 1, 7)]
    assert best["metrics"]["max_time"] == pytest.approx(6.0)
    assert best["chunks"][0]["time"] == pytest.approx(6.0)
    # Per stage, the same table's optimum is one stage on Big and six on the GPU: 12 ms.
    per_stage = solve(HOT_STAGES)[0]
    assert per_stage["metrics"]["max_time"] == pytest.approx(12.0)


def test_range_hot_cpu_chunks_keep_their_per_stage_price_and_gpu_drops_its_overhead():
    oh = {"Little": (0, 0), "Medium": (0, 0), "Big": (1.0, 0.5), "GPU": (3.0, 1.0)}
    sols = solve(HOT_STAGES, "range-hot", HOT_RANGES, n=15, overhead=oh)
    mixed = [s for s in sols if {c["core_type"] for c in s["chunks"]} == {"GPU", "Big"}]
    assert mixed
    for sol in sols:
        for c in sol["chunks"]:
            n = c["end_stage"] - c["start_stage"] + 1
            col = CORE_TYPES.index(c["core_type"])
            if c["core_type"] != "GPU":  # per stage + the CPU overhead, as under per-stage
                kernel = sum(
                    HOT_STAGES[k][col] for k in range(c["start_stage"] - 1, c["end_stage"])
                )
                assert c["time"] == pytest.approx(
                    kernel + oh[c["core_type"]][0] + n * oh[c["core_type"]][1]
                )
            else:  # the measured range alone: no stage sum, no GPU overhead
                t0 = HOT_RANGES[(c["start_stage"] - 1, c["end_stage"] - 1)][0]
                assert c["time"] == pytest.approx(t0)
        assert sol["metrics"]["max_time"] >= max(c["time"] for c in sol["chunks"]) - 1e-9


def test_range_price_does_not_depend_on_per_stage_records():
    """Spec "Range data present": the [4, 7] GPU chunk lies between its t0 and t1 and does
    not move when the per-stage GPU times of stages 4-7 do."""
    stages = [[HUGE, HUGE, 2.0 if s < 3 else HUGE, HUGE if s < 3 else 1.0] for s in range(7)]
    ranges = {(3, 6): (5.0, 9.0)}
    for model in ("range-hot", "range-idle"):
        prices = []
        for gpu_stage in (1.0, 50.0):
            table = [row[:3] + [gpu_stage if s >= 3 else HUGE] for s, row in enumerate(stages)]
            best = solve(table, model, ranges, n=1)[0]
            assert shape(best) == [("Big", 1, 3), ("GPU", 4, 7)]
            prices.append(gpu_chunks(best)[0]["time"])
        assert prices[0] == pytest.approx(prices[1])
        assert 5.0 - 1e-9 <= prices[0] <= 9.0 + 1e-9


# --- 3.3: range-idle -----------------------------------------------------------------------


def idle_case(big_ms):
    """Big runs 1-3 (big_ms in total), the GPU 4-7 (t0 5, t1 10, the only range record);
    every other placement is prohibitive."""
    stages = [
        [HUGE, HUGE, big_ms / 3 if s < 3 else HUGE, HUGE if s < 3 else HUGE] for s in range(7)
    ]
    return solve(stages, "range-idle", {(3, 6): (5.0, 10.0)}, n=1)[0]


def test_range_idle_at_the_bottleneck_is_t0():
    best = idle_case(3.0)  # the GPU chunk is the bottleneck: it never idles
    assert shape(best) == [("Big", 1, 3), ("GPU", 4, 7)]
    assert gpu_chunks(best)[0]["time"] == pytest.approx(5.0)
    assert best["metrics"]["max_time"] == pytest.approx(5.0)


def test_range_idle_with_slack_rises_toward_t1():
    best = idle_case(8.0)  # Big sets an 8 ms period; the GPU idles 8 - t per item
    t = gpu_chunks(best)[0]["time"]
    assert 5.0 < t <= 10.0
    assert t == pytest.approx(idle_price(5.0, 10.0, 8.0)) and t == pytest.approx(6.5)
    assert best["metrics"]["max_time"] == pytest.approx(8.0)
    # A long period caps the price at t1.
    assert gpu_chunks(idle_case(40.0))[0]["time"] == pytest.approx(10.0)


def test_reprice_matches_the_solver_under_both_range_models():
    for model in ("range-hot", "range-idle"):
        for sol in solve(HOT_STAGES, model, HOT_RANGES, n=8):
            again = reprice_solution(sol, CORE_TYPES, HOT_STAGES, None, model, HOT_RANGES)
            assert [c["time"] for c in again["chunks"]] == pytest.approx(
                [c["time"] for c in sol["chunks"]]
            )


# --- 3.4: several GPU chunks ---------------------------------------------------------------


def test_two_gpu_chunks_bound_the_period_by_their_sum():
    """GPU fast on [1, 2] and [5, 7], Big on [3, 4]: GPU[1-2] | Big[3-4] | GPU[5-7] with
    measured ranges of 3 and 4 ms has a period of at least 3 + 4 = 7 ms."""
    stages = [[HUGE, HUGE, 1.0 if s in (2, 3) else 10.0, 10.0] for s in range(7)]
    ranges = {
        (i, j): (10.0 * (j - i + 1), 12.0 * (j - i + 1)) for i in range(7) for j in range(i, 7)
    }
    ranges[(0, 1)] = (3.0, 6.0)
    ranges[(4, 6)] = (4.0, 8.0)
    for model in ("range-hot", "range-idle"):
        sols = solve(stages, model, ranges, n=10, multi=True)
        best = sols[0]
        assert shape(best) == [("GPU", 1, 2), ("Big", 3, 4), ("GPU", 5, 7)]
        assert [c["time"] for c in gpu_chunks(best)] == pytest.approx([3.0, 4.0])  # t0 each
        assert best["metrics"]["max_time"] == pytest.approx(7.0)
        for sol in sols:
            gpu_sum = sum(c["time"] for c in gpu_chunks(sol))
            assert sol["metrics"]["max_time"] >= gpu_sum - 1e-9, sol["uid"]
    # Without the multi-chunk option one GPU chunk is the best it can do.
    assert len(gpu_chunks(solve(stages, "range-hot", ranges, n=1)[0])) == 1
