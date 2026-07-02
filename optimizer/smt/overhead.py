"""Per-chunk framework-overhead model for the z3 cost function.

The profiling table holds per-stage KERNEL times; a real chunk additionally pays a
per-chunk constant (SPSC handoff, thread wake, first submit) and a per-stage dispatch
tax (GPU submit + fence round-trip per stage in the chunk). Without these terms the
solver under-predicts small/overhead-bound chunks 2-6x and picks pipelines that lose
to a single PU on tiny apps (tree) -- see the 2026-07-02 fresh-start baseline.

The constants are FITTED from measured schedule runs by
optimizer/analysis/fit_overhead.py, which writes

    <profiling_root>/<device>/overhead.json
    { "<class>": {"per_chunk_ms": float, "per_stage_ms": float, ...}, ... }

with classes "cpu", "gpu_cuda", "gpu_vulkan". At solve time the class is resolved per
solver core-type column: CPU tiers -> "cpu", the GPU column -> the backend being
solved ("gpu_cuda"/"gpu_vulkan"). Missing file/class means zero overhead (the old
behavior); 02_gen_schedule_merged exposes --no-overhead to opt out explicitly.
"""

import json
import os


def load_overhead(profiling_root, device):
    """Return the fitted overhead dict for a device, or None if never fitted."""
    path = os.path.join(profiling_root, device, "overhead.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_for_solver(overhead, core_types, gpu_backend):
    """Map solver core-type columns to (per_chunk_ms, per_stage_ms) tuples.

    gpu_backend: "gpu_cuda" | "gpu_vulkan" -- the backend this solve targets.
    Absent classes resolve to (0, 0) so a partial fit degrades gracefully.
    """
    out = {}
    for c in core_types:
        cls = gpu_backend if c == "GPU" else "cpu"
        entry = (overhead or {}).get(cls) or {}
        out[c] = (
            float(entry.get("per_chunk_ms", 0.0)),
            float(entry.get("per_stage_ms", 0.0)),
        )
    return out
