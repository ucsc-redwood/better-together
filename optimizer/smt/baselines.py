"""Baseline data and configuration management for schedule optimization.

Baselines are DERIVED from the measured profiling data, not hand-coded: each
whole-pipeline single-PU baseline is the sum of the per-stage isolated times read
straight from the canonical JSONL store (the same data 02 solves on). The GPU baseline
is the sum of the backend's column (vulkan/cuda); the OMP baseline is the sum of the
*fastest fully-measured* CPU tier (little/medium/big). This keeps the numbers in
lock-step with the measured data and naturally covers every device/app/backend present
in the store (incl. minipc, which the old hand-coded table missed).

Reading the JSONL store directly (via profiling_loader, count-weighted across runs)
also fixes the wide-CSV path's latent multi-run bug: that path summed every (run,stage)
row, so an N-run isolated.csv inflated the baseline N-fold.
"""
from orchestrate.case import Case
from smt.profiling_loader import load_profiling
from .bt_vocab import APP_STAGES  # generated from vocab.json

# Default profiling-store root (overridable by callers, e.g. 02's --profiling_root).
DEFAULT_PROFILING_ROOT = "data/profiling"

_CPU_TIERS = ("little", "medium", "big")


def get_baseline_for_config(device, app, backend, root=DEFAULT_PROFILING_ROOT):
    """Whole-pipeline baselines for a config, derived from its isolated profiling.

    Returns ``{"omp": <ms>, <backend>: <ms>, "fastest": <ms>}`` (the GPU key is the
    backend token "vk"/"cu", matching the callers), or None when the data is absent.
    """
    case = Case(device, app, backend)
    try:
        # max_cv=1.0 mirrors the z3 cost-matrix loader (data_loader.load_stage_timings):
        # keep every measured stage, dropping only explicit thermal-throttle samples, so
        # the baseline covers the same stages the solver sees. (The retired wide-CSV path
        # did no CV filtering at all.)
        table, _ = load_profiling(root, device, app, case.backend_long, "isolated", max_cv=1.0)
    except FileNotFoundError:
        print(f"Warning: no isolated profiling (baseline source) for {device}/{app}/{backend}")
        return None

    num_stages = get_num_stages_for_app(app)
    stages = range(1, num_stages + 1)

    def column_sum(pu):
        """Sum a PU's per-stage isolated time, or None unless every stage is measured
        (a partly-measured tier isn't a runnable whole-pipeline baseline)."""
        vals = [table[(s, pu)]["value"] for s in stages if (s, pu) in table]
        return sum(vals) if len(vals) == num_stages else None

    gpu_time = column_sum(case.backend_long)
    tier_sums = [t for t in (column_sum(tier) for tier in _CPU_TIERS) if t is not None]
    omp_time = min(tier_sums) if tier_sums else None

    if omp_time is None and gpu_time is None:
        print(f"Warning: isolated profiling for {device}/{app}/{backend} has no usable column")
        return None

    present = [t for t in (omp_time, gpu_time) if t is not None]
    result = {"fastest": min(present)}
    if omp_time is not None:
        result["omp"] = omp_time
    if gpu_time is not None:
        result[backend] = gpu_time
    return result


def get_num_stages_for_app(app_name):
    """Get the number of stages for the given application."""
    stage_counts = APP_STAGES  # generated from vocab.json

    # Extract the base app name without backend suffix
    base_app = app_name.split("-")[0] if "-" in app_name else app_name

    # For cifar apps, both dense and sparse variants have the same number of stages
    if base_app == "cifar":
        return stage_counts.get("cifar-dense", 9)

    return stage_counts.get(base_app, 9)  # Default to 9 if not found
