#!/usr/bin/env python3
"""Gate for baselines.py: the loader returns MEASURED whole-pipeline baselines
derived from the committed isolated.csv store (not hand-coded), including minipc.

    uv run python scripts/collect/smt/test_baselines.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smt.baselines import get_baseline_for_config  # noqa: E402

# Repo root so the default CSV root (data/btpm_export) resolves regardless of cwd.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSV_ROOT = os.path.join(ROOT, "data", "btpm_export")


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def main():
    failures = []

    # jetson/cifar-dense/cu: sum(cuda)=38.45, sum(little)=189.51 (the only CPU tier).
    # The plan's "measured 38.1" cross-checks the cuda sum vs the stale hand-coded 5.48.
    b = get_baseline_for_config("jetson", "cifar-dense", "cu", CSV_ROOT)
    assert b is not None, "jetson/cifar-dense/cu baseline missing"
    if not approx(b["cu"], 38.45):
        failures.append(f"jetson cu={b['cu']} expected ~38.45")
    if not approx(b["omp"], 189.51):
        failures.append(f"jetson omp={b['omp']} expected ~189.51")
    if not approx(b["fastest"], 38.45):
        failures.append(f"jetson fastest={b['fastest']} expected ~38.45")

    # minipc: the device the old hand-coded table omitted entirely (returned None).
    b = get_baseline_for_config("minipc", "tree", "vk", CSV_ROOT)
    assert b is not None, "minipc/tree/vk baseline missing (the old table's gap)"
    if not approx(b["vk"], 2.0646):
        failures.append(f"minipc vk={b['vk']} expected ~2.0646")
    if not approx(b["omp"], 2.1192):  # only 'big' tier populated on the Big-only MiniPC
        failures.append(f"minipc omp={b['omp']} expected ~2.1192")

    # phone: little/medium/big all populated -> omp must pick the FASTEST tier (medium).
    b = get_baseline_for_config("R5CY21Y3VEV", "cifar-sparse", "vk", CSV_ROOT)
    assert b is not None, "R5CY21Y3VEV/cifar-sparse/vk baseline missing"
    medium_sum = 321.679921 + 7.586172 + 872.087344 + 3.888476 + 899.646132 \
        + 1763.532968 + 1765.025312 + 2.065430 + 1.022031
    if not approx(b["omp"], medium_sum):
        failures.append(f"phone omp={b['omp']} expected fastest tier (medium) ~{medium_sum:.1f}")

    # absent config -> None (no crash).
    if get_baseline_for_config("nope", "tree", "vk", CSV_ROOT) is not None:
        failures.append("absent config should return None")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("OK: baselines derived from measured isolated.csv (jetson, minipc, phone)")


if __name__ == "__main__":
    main()
