#!/usr/bin/env python3
"""Gate for baselines.py: whole-pipeline baselines are MEASURED (summed straight from
the canonical JSONL profiling store), not hand-coded, and the selection logic is right.

The expected magnitudes are recomputed from the SAME store via an independent path
(load_profiling + a plain per-PU sum) rather than pinned to specific numbers -- the
store is regenerable/un-versioned, so hard-coded magnitudes go stale on every
re-collection. What this pins is the LOGIC: GPU = backend column sum, OMP = fastest
fully-measured CPU tier, fastest = min(omp, gpu), absent config -> None, and the
Big-only MiniPC vs all-tiers phone tier selection.

    uv run python scripts/collect/smt/test_baselines.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case import Case  # noqa: E402
from profiling_loader import load_profiling  # noqa: E402
from smt.baselines import get_baseline_for_config, get_num_stages_for_app  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STORE = os.path.join(ROOT, "data", "profiling")
_CPU_TIERS = ("little", "medium", "big")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def store_column_sums(device, app, backend):
    """Independent oracle: per-PU isolated whole-pipeline sum (only fully-measured PUs)."""
    case = Case(device, app, backend)
    table, _ = load_profiling(STORE, device, app, case.backend_long, "isolated", max_cv=1.0)
    stages = range(1, get_num_stages_for_app(app) + 1)
    sums = {}
    for pu in _CPU_TIERS + (case.backend_long,):
        vals = [table[(s, pu)]["value"] for s in stages if (s, pu) in table]
        if len(vals) == len(stages):
            sums[pu] = sum(vals)
    return sums, case.backend_long


def check(device, app, backend, failures, *, expect_tiers):
    """Assert get_baseline_for_config matches the independent oracle, and that the OMP
    tier selection is the fastest fully-measured CPU tier (named in expect_tiers)."""
    b = get_baseline_for_config(device, app, backend, STORE)
    assert b is not None, f"{device}/{app}/{backend} baseline missing"
    sums, gpu_pu = store_column_sums(device, app, backend)

    # GPU baseline == backend column sum.
    if not approx(b[backend], sums[gpu_pu]):
        failures.append(f"{device} {backend}={b[backend]} expected {sums[gpu_pu]}")

    # OMP baseline == min over the CPU tiers actually present in the store...
    cpu_sums = {t: sums[t] for t in _CPU_TIERS if t in sums}
    if set(cpu_sums) != set(expect_tiers):
        failures.append(f"{device} present CPU tiers {sorted(cpu_sums)} != {sorted(expect_tiers)}")
    omp_expected = min(cpu_sums.values())
    if not approx(b["omp"], omp_expected):
        failures.append(f"{device} omp={b['omp']} expected fastest tier {omp_expected}")

    # ...and fastest == min(omp, gpu).
    if not approx(b["fastest"], min(omp_expected, sums[gpu_pu])):
        failures.append(f"{device} fastest={b['fastest']} expected {min(omp_expected, sums[gpu_pu])}")


def main():
    failures = []

    # jetson: only the 'little' CPU tier was profiled -> OMP must come from 'little'.
    check("jetson", "cifar-dense", "cu", failures, expect_tiers=("little",))

    # minipc: the device the old hand-coded table omitted; Big-only -> OMP from 'big'.
    check("minipc", "tree", "vk", failures, expect_tiers=("big",))

    # phone: little/medium/big all populated -> OMP must pick the FASTEST of the three.
    check("R5CY21Y3VEV", "cifar-sparse", "vk", failures, expect_tiers=_CPU_TIERS)

    # absent config -> None (no crash).
    if get_baseline_for_config("nope", "tree", "vk", STORE) is not None:
        failures.append("absent config should return None")

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("OK: baselines derived from measured JSONL store (jetson, minipc, phone)")


if __name__ == "__main__":
    main()
