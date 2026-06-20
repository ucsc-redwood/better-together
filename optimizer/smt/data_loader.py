"""Data loading and processing utilities for schedule optimization."""

from orchestrate.case import Case

from smt.profiling_loader import load_profiling

from .baselines import get_num_stages_for_app
from .bt_vocab import CORE_TYPES
from .bt_vocab import CPU_TIERS as _CPU_TIERS  # generated from vocab.json

# A CPU tier the device physically lacks -- encode as a huge cost so z3 never assigns a
# stage to hardware that does not exist (0.0 would look infinitely fast in a minimization
# and z3 would pick it, then the executor crashes -- e.g. the Big-only MiniPC has no
# little/medium cores).
UNAVAILABLE = 1e9

# Scenario-aware quality gates (interference-audit rec 6). The interference scenario is
# noisier, so require >=2 runs to AGREE before a cell feeds z3 -- this kills the exact
# failure the audit flagged ("a cv>1 cell surviving on ONE lucky run under the gate").
# The CV gate stays 1.0: it already excludes every cv>1 cell, and lowering it further only
# drops moderately-noisy CPU-tier cells, breaking per-tier completeness and gratuitously
# failing btpm schedules. The real correctness fix for interference is min_runs>=2 + the
# DVFS floor below, not a tighter threshold. Isolated is unchanged.
SCENARIO_GATES = {
    "isolated": {"max_cv": 1.0, "min_runs": 1},
    "interference": {"max_cv": 1.0, "min_runs": 2},
}


def _apply_dvfs_floor(table, root, device, app, gpu_pu, metric, num_stages):
    """DVFS guard (interference-audit rec 1). A GPU stage that measures FASTER under
    interference than in isolation is physically impossible for true contention -- it is a
    clock/DVFS artifact (the background GPU load kept the iGPU boosted while the isolated
    run ran gappy at a low clock). Clamp such cells UP to the isolated value so z3 never
    sees a too-cheap-under-load GPU. Mutates `table`; returns [(stage, itf, iso)] clamped.
    """
    try:
        iso, _ = load_profiling(
            root,
            device,
            app,
            gpu_pu,
            "isolated",
            metric=metric,
            max_cv=SCENARIO_GATES["isolated"]["max_cv"],
            min_runs=SCENARIO_GATES["isolated"]["min_runs"],
        )
    except (FileNotFoundError, ValueError):
        return []  # no usable isolated reference -> nothing to floor against
    clamped = []
    for stage in range(1, num_stages + 1):
        key = (stage, gpu_pu)
        if key in table and key in iso and table[key]["value"] < iso[key]["value"]:
            clamped.append((stage, table[key]["value"], iso[key]["value"]))
            table[key] = {**table[key], "value": iso[key]["value"], "dvfs_clamped": True}
    return clamped


def load_stage_timings(
    root,
    device,
    app,
    backend,
    scenario,
    metric="p50",
    max_cv=None,
    min_runs=None,
    verbose=False,
    dvfs_floor=True,
):
    """Build the z3 per-stage cost matrix directly from the canonical JSONL store.

    Reads ``<root>/<device>/<app>/<backend_long>/<scenario>/run-*.jsonl`` via
    ``profiling_loader.load_profiling`` (schema-validated, throttled/high-CV filtered,
    count-weighted across runs) and returns ``(avg_timings, use_cuda)`` where
    ``avg_timings[s]`` is ``[little, medium, big, gpu]`` for stage s+1.

    Presence is by EXISTENCE, not by a 0.0 sentinel: a CPU tier with no measured record
    for this cell is ABSENT and encoded as ``UNAVAILABLE``; the GPU column is the
    REQUESTED backend (``cu``->cuda / ``vk``->vulkan), never sniffed from the data.

    Interference-scenario guards (see SCENARIO_GATES + _apply_dvfs_floor): a stricter
    CV/min-runs gate, and a DVFS floor that clamps a too-cheap-under-load GPU up to its
    isolated value. ``max_cv``/``min_runs`` default to the scenario gate; pass them
    explicitly to override. ``dvfs_floor=False`` disables the clamp (e.g. for tests).
    """
    gates = SCENARIO_GATES.get(scenario, {"max_cv": 1.0, "min_runs": 1})
    eff_max_cv = gates["max_cv"] if max_cv is None else max_cv
    eff_min_runs = gates["min_runs"] if min_runs is None else min_runs

    case = Case(device, app, backend)
    gpu_pu = case.backend_long  # "cuda" | "vulkan"
    table, report = load_profiling(
        root, device, app, gpu_pu, scenario, metric=metric, max_cv=eff_max_cv, min_runs=eff_min_runs
    )
    num_stages = get_num_stages_for_app(app) if app else 9

    if dvfs_floor and scenario == "interference":
        clamped = _apply_dvfs_floor(table, root, device, app, gpu_pu, metric, num_stages)
        if clamped:
            print(
                f"WARNING [DVFS guard]: {device}/{app}/{backend}: clamped {len(clamped)} GPU "
                f"stage(s) where interference<isolated (clock artifact) up to the isolated floor: "
                + ", ".join(f"s{s}({itf:.4f}->{iso:.4f}ms)" for s, itf, iso in clamped)
            )

    # A CPU tier is present iff it has at least one measured stage in this cell.
    tier_present = {
        pu: any((s, pu) in table for s in range(1, num_stages + 1)) for pu in _CPU_TIERS
    }
    if verbose:
        absent = [pu for pu, ok in tier_present.items() if not ok]
        print(f"loaded {report['n_records']} records from {len(report['paths'])} run file(s)")
        print(f"GPU backend: {gpu_pu}; absent CPU tiers (UNAVAILABLE): {absent or 'none'}")

    avg_timings = []
    demoted = []  # (stage, pu) CPU cells dropped by the gate -> encoded UNAVAILABLE
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
                # Tier present on other stages but its cell here didn't survive the gate
                # (e.g. min_runs>=2 dropped a single-lucky-run interference sample). Encode
                # UNAVAILABLE so z3 simply won't assign THIS stage to THIS tier, rather than
                # failing the whole app's schedule on one flaky cell (was a hard error; that
                # brittleness is exactly what the interference audit's rec 6 would trip).
                row.append(UNAVAILABLE)
                demoted.append((stage, pu))
        row.append(table[(stage, gpu_pu)]["value"])
        avg_timings.append(row)

    if demoted:
        print(
            f"WARNING [gate]: {device}/{app}/{backend}/{scenario}: {len(demoted)} CPU cell(s) "
            f"didn't survive the quality gate -> encoded UNAVAILABLE (z3 won't use them): "
            + ", ".join(f"s{s}/{pu}" for s, pu in demoted)
        )

    return avg_timings, backend == "cu"


def define_data(stage_timings=None, app_name=None):
    """Define the problem data."""
    # Get application-specific stage count if available
    num_stages = get_num_stages_for_app(app_name) if app_name else 9
    core_types = list(CORE_TYPES)

    # Use provided stage timings if available, otherwise use default values
    if stage_timings is not None:
        return num_stages, core_types, stage_timings

    # Default timings if no CSV is provided
    default_stage_timings = []

    return num_stages, core_types, default_stage_timings
