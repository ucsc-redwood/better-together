#!/usr/bin/env python3
"""Coverage matrix for the canonical JSONL profiling store: what (app x device x
backend) cells of the permutation table are collected, and what is missing.

The unit of collection is one GPU-backend binary run -- a single `bm-prof-<app>-<be>`
run measures the GPU PU *and* every present CPU tier in isolation, so the OMP
column is "bundled" (present iff any binary ran for that app on that device).

A cell is one of:
    collected  -- run-*.jsonl present and non-empty
    n/a        -- the hardware lacks that backend (by FLEET below)  -> not a gap
    MISSING    -- supported by the hardware but no data on disk      -> a gap

    uv run python optimizer/analysis/coverage.py            # default data/profiling
    uv run python optimizer/analysis/coverage.py --min-runs 3   # flag < 3 runs as low
"""

import argparse
import glob
import json

APPS = ["tree", "cifar-dense", "cifar-sparse"]

# The test fleet: device dir -> (display name, GPU backends the hardware supports).
# CUDA only where there is an NVIDIA GPU; Vulkan needs an integrated GPU. OMP runs
# everywhere and is measured inside every GPU-binary run. See
# docs/instruction-for-ai/01-hardware.md / 03-unit-testing.md for the source of truth.
FLEET = {
    "duck-stable": ("Jetson (stable)", ["cuda", "vulkan"]),
    "duck-naughty": ("Jetson (naughty)", ["cuda", "vulkan"]),
    "minipc": ("MiniPC", ["vulkan"]),
    "R5CY21Y3VEV": ("Samsung", ["vulkan"]),
}

GPU_PUS = {"cuda", "vulkan"}


def scan(root, scenario, dev, app, be):
    """(runs, cpu_tiers, abandoned_cells) for one (dev, app, be) cell."""
    runs = ab = 0
    cpu_tiers = set()
    for f in sorted(glob.glob(f"{root}/{dev}/{app}/{be}/{scenario}/run-*.jsonl")):
        n = 0
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if r["pu"] not in GPU_PUS:
                cpu_tiers.add(r["pu"])
            if r.get("provenance", {}).get("abandoned"):
                ab += 1
        if n > 0:
            runs += 1
    return runs, cpu_tiers, ab


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="data/profiling")
    ap.add_argument("--scenario", default="isolated")
    ap.add_argument(
        "--min-runs", type=int, default=1, help="cells with fewer collected runs are flagged LOW"
    )
    args = ap.parse_args()

    print(f"coverage of {args.root}/*/*/*/{args.scenario} (permutation: {len(APPS)} apps x fleet)")
    print(
        "legend:  OK=collected (runs)   --=n/a (hw lacks backend)   "
        "MISSING=gap   ab=cells early-abandoned\n"
    )

    total = collected = missing = 0
    gaps = []
    for app in APPS:
        print(f"### {app}")
        print(f"{'device':<8} {'OMP (cpu tiers)':<26} {'CUDA':<16} {'Vulkan':<16}")
        for dev, (name, supported) in FLEET.items():
            omp_tiers, omp_runs = set(), 0
            cols = {}
            for be in ("cuda", "vulkan"):
                runs, tiers, ab = scan(args.root, args.scenario, dev, app, be)
                if be not in supported:
                    cols[be] = "--"
                    continue
                total += 1
                if runs == 0:
                    cols[be] = "MISSING"
                    missing += 1
                    gaps.append(f"{dev}/{app}/{be}")
                    continue
                collected += 1
                tag = "OK" if runs >= args.min_runs else "LOW"
                cols[be] = f"{tag} {runs}run" + (f"/ab{ab}" if ab else "")
                omp_tiers |= tiers
                omp_runs = max(omp_runs, runs)
            omp = f"OK {omp_runs}run [{','.join(sorted(omp_tiers))}]" if omp_tiers else "MISSING"
            print(f"{name:<8} {omp:<26} {cols['cuda']:<16} {cols['vulkan']:<16}")
        print()

    print(f"summary: {collected}/{total} supported GPU-binary cells collected, {missing} missing")
    if gaps:
        print("gaps: " + ", ".join(gaps))


if __name__ == "__main__":
    main()
