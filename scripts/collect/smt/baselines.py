"""Baseline data and configuration management for schedule optimization.

Baselines are DERIVED from the committed measured profiling data, not hand-coded:
each whole-pipeline single-PU baseline is the sum of the per-stage isolated times in
``<csv_root>/<device>/<app>/<backend>/isolated.csv`` (the same table 02 solves on).
The GPU baseline is the sum of the backend's column (vulkan/cuda); the OMP baseline is
the sum of the *fastest fully-populated* CPU tier (little/medium/big). This keeps the
numbers in lock-step with the measured data and naturally covers every device/app/
backend present in the store (incl. minipc, which the old hand-coded table missed).
"""
import csv
import os

# Default profiling-table root (overridable by callers, e.g. 02's --csv_root_folder).
DEFAULT_CSV_ROOT = "data/btpm_export"

# isolated.csv column name for each backend token used on the CLI / dir layout.
_GPU_COLUMN = {"vk": "vulkan", "cu": "cuda"}
_CPU_TIERS = ("little", "medium", "big")


def _read_isolated(csv_path):
    """Return {column: [per-stage value, ...]} from an isolated.csv, or None."""
    if not os.path.exists(csv_path):
        return None
    columns = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for tier in _CPU_TIERS + tuple(_GPU_COLUMN.values()):
                if tier in row and row[tier] != "":
                    columns.setdefault(tier, []).append(float(row[tier]))
    return columns if columns else None


def get_baseline_for_config(device, app, backend, csv_root=DEFAULT_CSV_ROOT):
    """Whole-pipeline baselines for a config, derived from its isolated.csv.

    Returns ``{"omp": <ms>, <backend>: <ms>, "fastest": <ms>}`` (the GPU key is the
    backend token "vk"/"cu", matching the callers), or None when the data is absent.
    """
    csv_path = os.path.join(csv_root, device, app, backend, "isolated.csv")
    columns = _read_isolated(csv_path)
    if columns is None:
        print(f"Warning: No isolated.csv (baseline source) for {device}/{app}/{backend}")
        return None

    gpu_col = _GPU_COLUMN.get(backend)
    gpu_vals = columns.get(gpu_col)
    gpu_time = sum(gpu_vals) if gpu_vals and any(v > 0 for v in gpu_vals) else None

    # OMP baseline = the fastest CPU tier that is fully populated (every stage > 0);
    # a partly-zero tier isn't a runnable whole-pipeline baseline.
    tier_sums = [
        sum(vals)
        for tier in _CPU_TIERS
        if (vals := columns.get(tier)) and all(v > 0 for v in vals)
    ]
    omp_time = min(tier_sums) if tier_sums else None

    if omp_time is None and gpu_time is None:
        print(f"Warning: isolated.csv for {device}/{app}/{backend} has no usable column")
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
    stage_counts = {
        "tree": 7,
        "cifar-dense": 9,
        "cifar-sparse": 9,
    }

    # Extract the base app name without backend suffix
    base_app = app_name.split("-")[0] if "-" in app_name else app_name

    # For cifar apps, both dense and sparse variants have the same number of stages
    if base_app == "cifar":
        return stage_counts.get("cifar-dense", 9)

    return stage_counts.get(base_app, 9)  # Default to 9 if not found
