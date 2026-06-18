"""Data loading and processing utilities for schedule optimization."""

from orchestrate.case import Case
from smt.profiling_loader import load_profiling
from .baselines import get_num_stages_for_app

# A CPU tier the device physically lacks -- encode as a huge cost so z3 never assigns a
# stage to hardware that does not exist (0.0 would look infinitely fast in a minimization
# and z3 would pick it, then the executor crashes -- e.g. the Big-only MiniPC has no
# little/medium cores).
UNAVAILABLE = 1e9
_CPU_TIERS = ("little", "medium", "big")


def load_stage_timings(root, device, app, backend, scenario,
                       metric="p50", max_cv=1.0, verbose=False):
    """Build the z3 per-stage cost matrix directly from the canonical JSONL store.

    Reads ``<root>/<device>/<app>/<backend_long>/<scenario>/run-*.jsonl`` via
    ``profiling_loader.load_profiling`` (schema-validated, throttled/high-CV filtered,
    count-weighted across runs) and returns ``(avg_timings, use_cuda)`` where
    ``avg_timings[s]`` is ``[little, medium, big, gpu]`` for stage s+1.

    Presence is by EXISTENCE, not by a 0.0 sentinel: a CPU tier with no measured record
    for this cell is ABSENT and encoded as ``UNAVAILABLE``; the GPU column is the
    REQUESTED backend (``cu``->cuda / ``vk``->vulkan), never sniffed from the data.
    """
    case = Case(device, app, backend)
    gpu_pu = case.backend_long  # "cuda" | "vulkan"
    table, report = load_profiling(root, device, app, gpu_pu, scenario,
                                   metric=metric, max_cv=max_cv)
    num_stages = get_num_stages_for_app(app) if app else 9

    # A CPU tier is present iff it has at least one measured stage in this cell.
    tier_present = {pu: any((s, pu) in table for s in range(1, num_stages + 1))
                    for pu in _CPU_TIERS}
    if verbose:
        absent = [pu for pu, ok in tier_present.items() if not ok]
        print(f"loaded {report['n_records']} records from {len(report['paths'])} run file(s)")
        print(f"GPU backend: {gpu_pu}; absent CPU tiers (UNAVAILABLE): {absent or 'none'}")

    avg_timings = []
    for stage in range(1, num_stages + 1):
        # The requested GPU backend must be measured for every stage -- it is the target.
        if (stage, gpu_pu) not in table:
            raise ValueError(
                f"stage {stage} missing {gpu_pu} timing in "
                f"{case.profiling_dir(root, scenario)} (incomplete profiling data); "
                f"refusing to fabricate a zero-cost stage"
            )
        row = []
        for pu in _CPU_TIERS:
            if not tier_present[pu]:
                row.append(UNAVAILABLE)
            elif (stage, pu) in table:
                row.append(table[(stage, pu)]["value"])
            else:
                # Present for some stages but not this one -> incomplete, not absent.
                raise ValueError(
                    f"stage {stage} missing {pu} timing (tier present on other stages) "
                    f"in {case.profiling_dir(root, scenario)}; incomplete profiling data"
                )
        row.append(table[(stage, gpu_pu)]["value"])
        avg_timings.append(row)

    return avg_timings, backend == "cu"


def define_data(stage_timings=None, app_name=None):
    """Define the problem data."""
    # Get application-specific stage count if available
    num_stages = get_num_stages_for_app(app_name) if app_name else 9
    core_types = ["Little", "Medium", "Big", "GPU"]

    # Use provided stage timings if available, otherwise use default values
    if stage_timings is not None:
        return num_stages, core_types, stage_timings

    # Default timings if no CSV is provided
    default_stage_timings = []

    return num_stages, core_types, default_stage_timings
