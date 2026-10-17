#!/usr/bin/env python3
"""GPU stage-range records live beside the per-stage runs without changing them
(profiling/cost-table spec "Range records do not break existing stores"; design D1).

A store is built in a temp dir: per-stage run-*.jsonl for a 3-stage toy cell (app "tree",
whose loaders expect 7 stages, so all 7 are written), then range-*.jsonl beside them.

Run:  uv run pytest -q optimizer/tests/test_range_store.py
"""

import json
import os
import shutil

import pytest
from smt.data_loader import load_range_timings, load_stage_timings
from smt.profiling_loader import load_profiling, load_ranges

DEVICE, APP, BACKEND = "3A021JEHN02756", "tree", "vulkan"
PROV = {
    "git_sha": "abc1234",
    "ts": "2026-10-16T00:00:00Z",
    "host": "localhost",
    "freq_governor": "sched_pixel",
    "freq_locked": False,
    "gpu_clock_mhz": 400,
    "warmup_iters": 20,
    "harness": "bt.prof/v0",
}


def timing(p50, count=10):
    return {
        "unit": "ms",
        "count": count,
        "p50": p50,
        "p95": p50 * 1.1,
        "p99": p50 * 1.2,
        "mean": p50,
        "cv": 0.05,
        "min": p50 * 0.9,
        "max": p50 * 1.3,
    }


def stage_record(scenario, run, stage, pu, p50):
    return {
        "schema": "bt.profiling/v0",
        "device": DEVICE,
        "app": APP,
        "backend": BACKEND,
        "scenario": scenario,
        "run": run,
        "stage": stage,
        "pu": pu,
        "timing": timing(p50),
        "provenance": PROV,
    }


def range_record(scenario, run, a, b, gap_ms, p50, submissions=1):
    return {
        "schema": "bt.profiling-range/v0",
        "device": DEVICE,
        "app": APP,
        "backend": BACKEND,
        "scenario": scenario,
        "run": run,
        "start_stage": a,
        "end_stage": b,
        "pu": "vulkan",
        "gap_ms": gap_ms,
        "submissions": submissions,
        "timing": timing(p50),
        "provenance": {**PROV, "gpu_clock_samples": 12, "harness": "bt.ranges/v0"},
    }


def write(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


@pytest.fixture
def store(tmp_path):
    """Two isolated and two interference per-stage runs, no range files yet."""
    root = str(tmp_path / "profiling")
    for scenario in ("isolated", "interference"):
        for run in (1, 2):
            recs = []
            for s in range(1, 8):
                recs.append(stage_record(scenario, run, s, "vulkan", 1.0 + s + 0.1 * run))
                recs.append(stage_record(scenario, run, s, "big", 2.0 * s + 0.1 * run))
            d = os.path.join(root, DEVICE, APP, BACKEND, scenario)
            write(os.path.join(d, f"run-{run:03d}.jsonl"), recs)
    return root


def add_ranges(root, scenario, runs=(1, 2)):
    d = os.path.join(root, DEVICE, APP, BACKEND, scenario)
    for run in runs:
        recs = []
        for a in range(1, 8):
            for b in range(a, 8):
                t0 = 10.0 * (b - a + 1) + run
                subs = 2 if a <= 3 < b else 1
                recs.append(range_record(scenario, run, a, b, 0, t0, subs))
                recs.append(range_record(scenario, run, a, b, t0, 1.5 * t0, subs))
        write(os.path.join(d, f"range-{run:03d}.jsonl"), recs)


def snapshot(root):
    return {
        sc: (
            load_profiling(root, DEVICE, APP, BACKEND, sc)[0],
            load_stage_timings(root, DEVICE, APP, "vk", sc)[0],
        )
        for sc in ("isolated", "interference")
    }


def test_old_store_loads_unchanged(store):
    """An old store (no range files) loads without error; the range loader yields nothing."""
    before = snapshot(store)
    assert before["isolated"][1][0] == [1e9, 1e9, pytest.approx(2.15), pytest.approx(2.15)]
    ranges, report = load_ranges(store, DEVICE, APP, BACKEND, "isolated")
    assert ranges == {} and report["paths"] == []
    assert load_range_timings(store, DEVICE, APP, "vk", "isolated") == {}


def test_range_files_leave_the_per_stage_table_unchanged(store):
    before = snapshot(store)
    add_ranges(store, "isolated")
    add_ranges(store, "interference")
    assert snapshot(store) == before


def test_range_loader_reads_both_cadences(store):
    add_ranges(store, "isolated")
    ranges, report = load_ranges(store, DEVICE, APP, BACKEND, "isolated")
    assert len(ranges) == 28 and report["n_records"] == 2 * 56
    r = ranges[(4, 7)]
    assert r["t0"] == pytest.approx(41.5)  # mean of runs 1 and 2 (equal counts)
    assert r["t1"] == pytest.approx(1.5 * 41.5)
    assert r["gap_ms"] == pytest.approx(41.5)
    assert ranges[(1, 7)]["submissions"] == 2 and ranges[(4, 7)]["submissions"] == 1
    # Solver view: 0-based (i, j) -> (t0, t1).
    timings = load_range_timings(store, DEVICE, APP, "vk", "isolated")
    assert timings[(3, 6)] == (pytest.approx(41.5), pytest.approx(62.25))


def test_interference_ranges_need_two_runs(store):
    add_ranges(store, "interference", runs=(1,))
    assert load_range_timings(store, DEVICE, APP, "vk", "interference") == {}
    add_ranges(store, "interference", runs=(1, 2))
    assert len(load_range_timings(store, DEVICE, APP, "vk", "interference")) == 28


def test_range_record_is_not_a_per_stage_record(store):
    """A range record misfiled as run-*.jsonl fails the per-stage schema loudly instead of
    being read as stage start_stage (why ranges have their own files and schema)."""
    d = os.path.join(store, DEVICE, APP, BACKEND, "isolated")
    shutil.copy(os.path.join(d, "run-001.jsonl"), os.path.join(d, "run-003.jsonl"))
    with open(os.path.join(d, "run-003.jsonl"), "a") as f:
        f.write(json.dumps(range_record("isolated", 3, 4, 7, 0, 40.0)) + "\n")
    with pytest.raises(ValueError, match="schema validation failed"):
        load_profiling(store, DEVICE, APP, BACKEND, "isolated")
