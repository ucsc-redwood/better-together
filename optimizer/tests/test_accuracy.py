#!/usr/bin/env python3
"""Prediction accuracy of executed schedules (optimizer/cost-model spec "Prediction accuracy
is reported for executed schedules"): analysis/accuracy.py on fixture schedule JSON and
fixture sched_logs, and its section in speedup_summary's Markdown.

Fixture cell (pixel, tree, vk), logs at a 1 kHz tick so ticks are milliseconds:
  per-stage/isolated  A Big 1-3 | GPU 4-7 (predicted 10), B GPU 1-7 (predicted 20)
  range-hot/isolated  A' = A's assignment (predicted 12), C GPU 1-2 | Big 3-4 | GPU 5-7 (8)
  per-stage/btpm      B'' = B's assignment (predicted 25)
  logs                A measured 11 / 12 / 13 ms (3 logs), B 22 / 20 / 24, C once: GPU chunks
                      3 + 5 ms, Big 6 ms -> 8 ms (one device serves both GPU chunks)
These schedules carry no metrics.cost_model / table (as before 02 recorded them), so the file
name names their model. Further tests: gapness schedule files, alone and beside tmax ones,
and 02's own per-stage fallback for a range model on a store without range records.

Run:  uv run pytest -q optimizer/tests/test_accuracy.py
"""

import json
import os
import subprocess
import sys

import pytest
from analysis.accuracy import accuracy_rows, signature, summarize
from analysis.speedup_summary import render_markdown
from test_range_store import store  # noqa: F401  (fixture)

GEN = os.path.join(os.path.dirname(__file__), "..", "orchestrate", "02_gen_schedule_merged.py")

DEV, APP, BE = "3A021JEHN02756", "tree", "vk"


def sched(uid, chunks, predicted):
    return {
        "uid": uid,
        "chunks": [
            {"id": i, "core_type": pu, "start_stage": a, "end_stage": b, "time": 1.0}
            for i, (pu, a, b) in enumerate(chunks)
        ],
        "metrics": {"max_time": predicted},
    }


A = [("Big", 1, 3), ("GPU", 4, 7)]
B = [("GPU", 1, 7)]
C = [("GPU", 1, 2), ("Big", 3, 4), ("GPU", 5, 7)]


def block(uid, chunk_ms, tasks=4):
    lines = ["### Python Begin ###", f"Schedule_UID={uid}", "Frequency=1000 Hz"]
    t = 1000
    for task in range(tasks):
        for chunk, ms in enumerate(chunk_ms):
            lines.append(f"Task={task} Chunk={chunk} Start={t} End={t + ms} Duration={ms}")
            t += ms
    return "\n".join(lines + ["### Python End ###"]) + "\n"


@pytest.fixture
def fleet(tmp_path):
    root, logs = tmp_path / "data", tmp_path / "data" / "sched_logs"
    files = {
        ("isolated", ""): [sched("A", A, 10.0), sched("B", B, 20.0)],
        ("isolated", "_range-hot"): [sched("A2", A, 12.0), sched("C", C, 8.0)],
        ("btpm", ""): [sched("B3", B, 25.0)],
    }
    for (table, suffix), sols in files.items():
        d = root / f"schedules_{table}" / DEV / APP / BE
        d.mkdir(parents=True, exist_ok=True)
        (d / f"schedules_{table}_tmax{suffix}.json").write_text(json.dumps(sols))
    run_dir = logs / f"{DEV}_{APP}_{BE}_isolated"
    run_dir.mkdir(parents=True)
    for i, (a_big, b_gpu) in enumerate([(11, 22), (12, 20), (13, 24)], 1):
        text = block("A", [a_big, 7]) + block("B", [b_gpu])
        if i == 1:
            text += block("C", [3, 6, 5])
        (run_dir / f"schedule_run_{i}.log").write_text(text)
    # A schedule no file proposes (another build's uid) is ignored.
    (run_dir / "schedule_run_4.log").write_text(block("SCH-unknown", [1, 1]))
    return str(root), str(logs)


def by_key(rows):
    return {(r["cost_model"], r["table"], r["uid"]): r for r in rows}


def test_ratios_per_executed_schedule(fleet):
    rows = by_key(accuracy_rows(*fleet))
    assert set(rows) == {
        ("per-stage", "isolated", "A"),
        ("per-stage", "isolated", "B"),
        ("range-hot", "isolated", "A2"),
        ("range-hot", "isolated", "C"),
        ("per-stage", "btpm", "B3"),
    }
    a = rows[("per-stage", "isolated", "A")]
    assert (a["measured"], a["measured_min"], a["measured_max"], a["n_logs"]) == (12, 11, 13, 3)
    assert a["ratio"] == pytest.approx(1.2)
    assert rows[("per-stage", "isolated", "B")]["ratio"] == pytest.approx(22 / 20)
    # Matched by assignment: A2 is A's pipeline under another model's uid.
    assert rows[("range-hot", "isolated", "A2")]["ratio"] == pytest.approx(1.0)
    # C's two GPU chunks share the device: 3 + 5 = 8 ms, not the 6 ms largest chunk.
    c = rows[("range-hot", "isolated", "C")]
    assert c["measured"] == pytest.approx(8.0) and c["ratio"] == pytest.approx(1.0)
    assert rows[("per-stage", "btpm", "B3")]["ratio"] == pytest.approx(22 / 25)


def test_median_abs_error_and_best_per_model_and_table(fleet):
    summary = {(r["cost_model"], r["table"]): r for r in summarize(accuracy_rows(*fleet))}
    assert summary[("per-stage", "isolated")]["n"] == 2
    assert summary[("per-stage", "isolated")]["median_abs_error"] == pytest.approx(0.15)
    assert summary[("per-stage", "isolated")]["best_measured"] == 12
    assert summary[("range-hot", "isolated")]["median_abs_error"] == pytest.approx(0.0)
    assert summary[("range-hot", "isolated")]["best_measured"] == pytest.approx(8.0)
    assert summary[("per-stage", "btpm")]["median_abs_error"] == pytest.approx(0.12)


def test_summary_markdown_lists_ratios_and_medians(fleet):
    md = render_markdown([], accuracy_rows(*fleet))
    assert "## Prediction accuracy" in md
    assert "| pixel | tree | VK | per-stage | isolated | 2 | 0.150 | 12.00 [11.00-13.00] |" in md
    assert "| pixel | tree | VK | range-hot | isolated | 2 | 0.000 | 8.00 [8.00-8.00] |" in md
    assert (
        "| pixel | tree | VK | per-stage | isolated | tmax | A | 10.00 | 12.00 [11.00-13.00] | 3 "
        "| 1.200 |" in md
    )
    assert (
        "| pixel | tree | VK | per-stage | btpm | tmax | B3 | 25.00 | 22.00 [20.00-24.00] | 3 "
        "| 0.880 |" in md
    )


def write_cell(root, files, logs):
    """files {file name: schedules} under schedules_isolated/; logs [log text] in the
    isolated log folder."""
    d = os.path.join(root, "schedules_isolated", DEV, APP, BE)
    os.makedirs(d, exist_ok=True)
    for name, sols in files.items():
        with open(os.path.join(d, name), "w") as f:
            json.dump(sols, f)
    run_dir = os.path.join(root, "sched_logs", f"{DEV}_{APP}_{BE}_isolated")
    os.makedirs(run_dir, exist_ok=True)
    for i, text in enumerate(logs, 1):
        with open(os.path.join(run_dir, f"schedule_run_{i}.log"), "w") as f:
            f.write(text)
    return root, os.path.join(root, "sched_logs")


def test_gapness_schedules_alone(tmp_path):
    fleet = write_cell(
        str(tmp_path),
        {"schedules_isolated_gapness.json": [sched("A", A, 10.0)]},
        [block("A", [ms, 7]) for ms in (11, 12, 13)],
    )
    (row,) = accuracy_rows(*fleet)
    assert (row["cost_model"], row["table"], row["mode"], row["uid"]) == (
        "per-stage",
        "isolated",
        "gapness",
        "A",
    )
    assert row["ratio"] == pytest.approx(1.2)


def test_both_modes_are_kept_and_a_shared_assignment_counts_once(tmp_path):
    """tmax proposes A and B, gapness A (another uid) and C: three schedules, A once
    (from the tmax file), so the summary's n and median are not inflated by the duplicate."""
    fleet = write_cell(
        str(tmp_path),
        {
            "schedules_isolated_tmax.json": [sched("A", A, 10.0), sched("B", B, 20.0)],
            "schedules_isolated_gapness.json": [sched("Ag", A, 10.0), sched("C", C, 8.0)],
        },
        [block("A", [12, 7]) + block("B", [22]) + block("C", [3, 6, 5])] * 3,
    )
    rows = accuracy_rows(*fleet)
    assert sorted((r["mode"], r["uid"]) for r in rows) == [
        ("gapness", "C"),
        ("tmax", "A"),
        ("tmax", "B"),
    ]
    (summary,) = summarize(rows)
    assert summary["n"] == 3
    assert summary["median_abs_error"] == pytest.approx(0.1)  # |1.2-1|, |1.1-1|, |1.0-1|


def test_a_range_model_without_range_records_is_reported_as_per_stage(store, tmp_path):  # noqa: F811
    """02 --cost-model range-hot on a store without range files warns and prices per stage,
    recording metrics.cost_model "per-stage" in schedules_isolated_tmax_range-hot.json. Its
    executed schedules are per-stage ones: reported under per-stage, once each beside the
    per-stage file's same schedules, never under range-hot."""
    root = str(tmp_path / "data")
    out = os.path.join(root, "schedules_isolated")
    for model in ("per-stage", "range-hot"):
        p = subprocess.run(
            [sys.executable, GEN, "--profiling_root", store, "--device", DEV, "--app", APP]
            + ["--backend", "vk", "--table_type", "isolated", "--minimize_mode", "tmax"]
            + ["-n", "5", "-o", out, "--cost-model", model],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "no range records" in p.stdout
    d = os.path.join(out, DEV, APP, BE)
    with open(os.path.join(d, "schedules_isolated_tmax_range-hot.json")) as f:
        fallback = json.load(f)
    assert {s["metrics"]["cost_model"] for s in fallback} == {"per-stage"}
    with open(os.path.join(d, "schedules_isolated_tmax.json")) as f:
        plain = json.load(f)

    executed = fallback[:3]  # run under the fallback file's uids
    logs = ["".join(block(s["uid"], [2] * len(s["chunks"])) for s in executed) for _ in range(3)]
    rows = accuracy_rows(*write_cell(root, {}, logs))
    sigs = {signature(s) for s in executed}
    assert {r["cost_model"] for r in rows} == {"per-stage"}
    assert sorted(r["signature"] for r in rows) == sorted(sigs)  # once each, not per file
    # Kept from the per-stage file, which comes before the fallback file.
    assert {r["uid"] for r in rows} <= {s["uid"] for s in plain}
    (summary,) = summarize(rows)
    assert summary["cost_model"] == "per-stage" and summary["n"] == len(sigs)
