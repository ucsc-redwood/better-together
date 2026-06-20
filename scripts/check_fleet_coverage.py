#!/usr/bin/env python3
"""check_fleet_coverage: fail if a GPU (app x backend x hardware) cell the project
EXPECTS to be exercised on real hardware was never RAN.

The everyday gate (ctest -L omp) gives NO signal on the CUDA/Vulkan backends: off-fleet
those differential tests GTEST_SKIP and skip-exit-0, so a never-run cell is indistinguishable
from a pass. The run-on-*.sh deploy scripts emit a machine-greppable marker per binary --

    BT-CELL <app> <backend> <hardware> <RAN|SKIP|FAIL>

into a coverage log (fleet-coverage.log). This checker diffs the *latest* status of each
cell in that log against the expected cells, so a silently-absent (never-marked) or
SKIP/FAIL cell is a hard failure instead of reading as covered.

The expected cells are DERIVED (no separate fleet-coverage.json to drift): per device in
fleet.json, the gated backends are `coverage_backends` if present else all benchmark
`backends` (mapped short->long via vocab.json), crossed with vocab.json `app_stages`.

Usage:
    scripts/check_fleet_coverage.py [coverage.log]     # default: fleet-coverage.log
Env:
    BT_CELL_LOG       override the coverage-log path (same default the run scripts use)
Exit code: 0 = every expected cell most-recently RAN; 1 = a missing/SKIP/FAIL cell.

See docs/reports-for-human/code-review-2026-06-18.md finding #8.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLEET = ROOT / "fleet.json"
VOCAB = ROOT / "vocab.json"
DEFAULT_LOG = Path(os.environ.get("BT_CELL_LOG", ROOT / "fleet-coverage.log"))

import json


def load_expected():
    """Derive the expected (app, backend, hardware) cells from fleet.json + vocab.json.

    apps come from vocab.json `app_stages`; per device, the gated backends are
    `coverage_backends` if present else all benchmark `backends`, mapped from the short
    name (cu/vk) to the long name (cuda/vulkan) used in the BT-CELL markers via vocab.json.
    """
    fleet = json.loads(FLEET.read_text(encoding="utf-8"))
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    long_name = {b["short"]: b["long"] for b in vocab["backends"]}
    apps = list(vocab["app_stages"])
    cells = []
    for hardware, dev in fleet["devices"].items():
        for short in dev.get("coverage_backends", dev["backends"]):
            for app in apps:
                cells.append((app, long_name[short], hardware))
    return cells


def load_observed(log_path: Path):
    """Latest status wins per (app, backend, hardware) cell."""
    seen: dict[tuple[str, str, str], str] = {}
    if not log_path.exists():
        return seen
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "BT-CELL":
            continue
        _, app, backend, hardware, status = parts
        seen[(app, backend, hardware)] = status
    return seen


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    expected = load_expected()
    observed = load_observed(log_path)

    print(f"fleet coverage check: {len(expected)} expected cells vs {log_path}")
    failures = []
    for cell in expected:
        status = observed.get(cell, "MISSING")
        app, backend, hardware = cell
        tag = "OK  " if status == "RAN" else "FAIL"
        print(f"  [{tag}] {app:<13} {backend:<7} {hardware:<8} {status}")
        if status != "RAN":
            failures.append((cell, status))

    # Markers for cells NOT in the manifest are surfaced (drift), but do not fail.
    extra = sorted(set(observed) - set(expected))
    for cell in extra:
        print(f"  [note] unlisted cell marked: {cell} = {observed[cell]}")

    print()
    if failures:
        print(f"COVERAGE RED -- {len(failures)} expected cell(s) not RAN:")
        for (app, backend, hardware), status in failures:
            hint = (
                "never ran (run the fleet deploy script)"
                if status == "MISSING"
                else f"last status {status}"
            )
            print(f"  - {app} {backend} {hardware}: {hint}")
        print("Run scripts/run-on-{jetson,rocky,android}.sh on the fleet to populate the log.")
        return 1
    print("COVERAGE GREEN -- every expected GPU cell most-recently RAN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
