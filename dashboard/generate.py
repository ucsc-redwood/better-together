#!/usr/bin/env python3
"""Generate the BetterTogether static analysis dashboard.

Reads the repo's single sources of truth -- devices/*.json, vocab.json, the
per-app kernel manifests (dashboard/manifest/), the kernel source files, and the
canonical JSONL profiling store (data/profiling/) -- and emits a self-contained
static site under dashboard/site/:

  site/  = copy of dashboard/static/  +  bundle.js (window.BT_DATA = {...})

The data is INLINED as bundle.js (not fetched) so the site opens offline by
double-clicking site/index.html -- browsers block fetch() of local files.

Run with the optimizer venv (it carries jsonschema and the reused loaders):
    uv run --project optimizer python dashboard/generate.py
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH_DIR = os.path.join(REPO_ROOT, "dashboard")
STATIC_DIR = os.path.join(DASH_DIR, "static")
SITE_DIR = os.path.join(DASH_DIR, "site")
MANIFEST_DIR = os.path.join(DASH_DIR, "manifest")
PROF_ROOT = os.path.join(REPO_ROOT, "data", "profiling")
SCHED_ROOT = os.path.join(REPO_ROOT, "data", "schedules")
SCHED_LOGS_ROOT = os.path.join(REPO_ROOT, "data", "sched_logs")
SCHED_SUMMARY = os.path.join(SCHED_LOGS_ROOT, "speedup-summary.md")
SCHEMA_DIR = os.path.join(REPO_ROOT, "schemas")

# Reuse the optimizer's loaders (path-layout + JSONL reader/validator) verbatim.
sys.path.insert(0, os.path.join(REPO_ROOT, "optimizer"))
from orchestrate.case import Case, to_short_backend  # noqa: E402
from smt.baselines import get_baseline_for_config  # noqa: E402
from smt.profiling_loader import read_records, validate  # noqa: E402

APPS = ["tree", "cifar-dense", "cifar-sparse"]
SCENARIOS = ["isolated", "interference"]
GPU_DIRS = ["cuda", "vulkan"]  # on-disk backend dir names (== backend_long)
CPU_TIERS = ["little", "medium", "big"]  # 'super' is an orphaned tier, never measured
PU_ORDER = ["cuda", "vulkan", "big", "medium", "little"]
METRICS = ["p50", "p95", "p99", "mean", "cv", "min", "max"]
MAX_CV_LOOSE = 0.5  # dashboard display gate (matches render_isolated_table.py default)
MAX_CV_STRICT = 0.1  # the solver's gate (profiling_loader default)

# schedule store axes (== the z3 CLI tokens / Case.schedule_path filenames)
SCHED_BACKENDS = ["cu", "vk"]
TABLE_TYPES = ["isolated", "btpm"]  # btpm == the interference profiling scenario
MODES = ["gapness", "tmax"]
# speedup-summary.md uses friendly device names; map them to devices/*.json ids.
SUMMARY_DEVICE_ALIAS = {
    "jetson": "jetson",
    "minipc": "minipc",
    "samsung": "R5CY21Y3VEV",
    "pixel": "3A021JEHN02756",
}


# --------------------------------------------------------------------------
# source extraction
# --------------------------------------------------------------------------
def read_text(rel_or_abs):
    path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(REPO_ROOT, rel_or_abs)
    with open(path, encoding="utf-8") as f:
        return f.read()


def infer_lang(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".comp":
        return "glsl"
    if ext in (".cu", ".cuh", ".cpp", ".hpp", ".cc", ".cxx", ".h", ".c"):
        return "cpp"
    return "plaintext"


def _match_braces(text, open_idx):
    """Return the index just past the '}' matching text[open_idx] == '{',
    skipping // /* */ comments and "..." '...' literals."""
    depth, i, n, state = 0, open_idx, len(text), None
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "line":
            if c == "\n":
                state = None
        elif state == "block":
            if c == "*" and nxt == "/":
                state = None
                i += 1
        elif state == "str":
            if c == "\\":
                i += 1
            elif c == '"':
                state = None
        elif state == "char":
            if c == "\\":
                i += 1
            elif c == "'":
                state = None
        else:
            if c == "/" and nxt == "/":
                state = "line"
                i += 1
            elif c == "/" and nxt == "*":
                state = "block"
                i += 1
            elif c == '"':
                state = "str"
            elif c == "'":
                state = "char"
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return n


def slice_symbol(text, symbol, path):
    """Extract the first DEFINITION of `symbol` (its full signature + braced body).
    A leading qualifier (class::, return type) on the same line is kept."""
    for m in re.finditer(r"\b" + re.escape(symbol) + r"\s*\(", text):
        brace = text.find("{", m.end())
        semi = text.find(";", m.end())
        if brace == -1:
            continue
        if semi != -1 and semi < brace:
            continue  # a forward declaration, not the definition
        sig_start = text.rfind("\n", 0, m.start()) + 1
        end = _match_braces(text, brace)
        return text[sig_start:end].rstrip("\n")
    raise SystemExit(f"generate.py: symbol '{symbol}' not found as a definition in {path}")


def slice_lines(text, lines, path):
    a, b = lines
    rows = text.splitlines()
    if a < 1 or b > len(rows) or a > b:
        raise SystemExit(f"generate.py: bad line range {lines} for {path} ({len(rows)} lines)")
    return "\n".join(rows[a - 1 : b])


def extract_source(entry):
    path = entry["path"]
    abspath = os.path.join(REPO_ROOT, path)
    if not os.path.isfile(abspath):
        raise SystemExit(f"generate.py: manifest source path does not exist: {path}")
    text = read_text(path)
    if "symbol" in entry:
        code = slice_symbol(text, entry["symbol"], path)
    elif "lines" in entry:
        code = slice_lines(text, entry["lines"], path)
    else:
        code = text.rstrip("\n")
    out = {"path": path, "lang": infer_lang(path), "code": code}
    if "symbol" in entry:
        out["symbol"] = entry["symbol"]
    return out


# --------------------------------------------------------------------------
# vocab + devices (Section 1)
# --------------------------------------------------------------------------
def load_vocab():
    return json.loads(read_text("vocab.json"))


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_devices(coverage):
    from jsonschema import Draft202012Validator

    schema = json.loads(read_text(os.path.join(SCHEMA_DIR, "device-spec.schema.json")))
    validator = Draft202012Validator(schema)
    freqs = json.loads(read_text(os.path.join(MANIFEST_DIR, "device_freqs.json")))

    devices = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "devices", "*.json"))):
        dev = json.loads(read_text(path))
        errs = sorted(validator.iter_errors(dev), key=lambda e: e.path)
        if errs:
            raise SystemExit(f"generate.py: {path} fails device schema: {errs[0].message}")
        did = dev["id"]
        cores = dev.get("cores", [])
        tiers = {}
        for tier in ["little", "medium", "big", "super"]:
            ids = [c["id"] for c in cores if c["type"] == tier]
            tiers[tier] = {
                "count": len(ids),
                "ids": ids,
                "pinnable": [c["pinnable"] for c in cores if c["type"] == tier],
            }
        gpu = dev.get("gpu")
        backends = ["omp"] + ([gpu["backend"]] if gpu else [])
        dev_cov = coverage.get(did, {})
        scen = sorted({s for app in dev_cov.values() for be in app.values() for s in be})
        devices.append(
            {
                "id": did,
                "description": dev.get("description", ""),
                "cores": cores,
                "cpu_tiers": tiers,
                "gpu": gpu,
                "backends_supported": backends,
                "has_data": bool(dev_cov),
                "scenarios_available": scen,
                "freq": freqs.get(did),
            }
        )
    return devices


# --------------------------------------------------------------------------
# apps (Section 2)
# --------------------------------------------------------------------------
def build_apps(vocab):
    apps = []
    for app in APPS:
        manifest = json.loads(read_text(os.path.join(MANIFEST_DIR, f"kernels.{app}.json")))
        n_stages = vocab["app_stages"][app]
        seen = {s["stage"] for s in manifest["stages"]}
        if seen != set(range(1, n_stages + 1)):
            raise SystemExit(
                f"generate.py: {app} manifest stages {sorted(seen)} != 1..{n_stages} (vocab.json)"
            )
        stages = []
        for s in sorted(manifest["stages"], key=lambda x: x["stage"]):
            kernels, support = {}, {}
            for backend in ["omp", "cuda", "vulkan"]:
                entries = s.get("sources", {}).get(backend, [])
                support[backend] = bool(entries)
                kernels[backend] = [extract_source(e) for e in entries]
            stages.append(
                {
                    "stage": s["stage"],
                    "op": s.get("op", f"Stage {s['stage']}"),
                    "desc": s.get("desc", ""),
                    "support": support,
                    "kernels": kernels,
                    "appdata": s.get("appdata", []),
                }
            )
        apps.append(
            {
                "id": app,
                "title": manifest.get("title", app),
                "input": manifest.get("input", ""),
                "characteristic": manifest.get("characteristic", ""),
                "n_stages": n_stages,
                "stages": stages,
            }
        )
    return apps


# --------------------------------------------------------------------------
# profiling (Section 3)
# --------------------------------------------------------------------------
def _aggregate(records, max_cv):
    kept = [
        r
        for r in records
        if not r["provenance"].get("throttled", False) and r["timing"]["cv"] <= max_cv
    ]
    if not kept:
        return None
    weight = sum(r["timing"]["count"] for r in kept)
    agg = {m: sum(r["timing"][m] * r["timing"]["count"] for r in kept) / weight for m in METRICS}
    agg["n_runs"] = len(kept)
    agg["count"] = weight
    return agg


def _raw_row(r):
    t = r["timing"]
    return {
        "run": r["run"],
        "count": t["count"],
        **{m: t[m] for m in METRICS},
        "backend_dir": r["backend"],
        "provenance": r["provenance"],
    }


def build_profiling(devices_gpu_backend):
    """Walk data/profiling, merge CPU/GPU records across backend dirs per
    (device, app, scenario), aggregate per (stage, pu). Returns (cells, coverage)."""
    coverage = {}
    cells = []
    if not os.path.isdir(PROF_ROOT):
        return cells, coverage

    device_ids = sorted(
        d for d in os.listdir(PROF_ROOT) if os.path.isdir(os.path.join(PROF_ROOT, d))
    )
    for device in device_ids:
        for app in APPS:
            for scenario in SCENARIOS:
                # gather records per present backend dir
                recs_by_dir = {}
                for bd in GPU_DIRS:
                    case = Case(device, app, to_short_backend(bd))
                    # skip 0-byte run files (a profiler may be mid-write) so the
                    # dashboard never crashes just because the store is being refreshed
                    paths = [p for p in sorted(glob.glob(case.profiling_glob(PROF_ROOT, scenario)))
                             if os.path.getsize(p) > 0]
                    if not paths:
                        continue
                    try:
                        recs = read_records(paths)
                        validate(recs)
                    except Exception as e:
                        print(f"Warning: skipping {device}/{app}/{bd}/{scenario} — unreadable "
                              f"(profiler mid-write?): {e}")
                        continue
                    coverage.setdefault(device, {}).setdefault(app, {}).setdefault(bd, {})[
                        scenario
                    ] = len(paths)
                    recs_by_dir[bd] = recs
                if not recs_by_dir:
                    continue
                # GPU-PU records come from their own dir; CPU-tier records from the
                # device's native GPU dir (else any present dir) to avoid double-count.
                merged = []
                for bd, recs in recs_by_dir.items():
                    merged += [r for r in recs if r["pu"] == bd]  # pu name == dir name
                native = devices_gpu_backend.get(device)
                cpu_dir = native if native in recs_by_dir else next(iter(recs_by_dir))
                merged += [r for r in recs_by_dir[cpu_dir] if r["pu"] in CPU_TIERS]

                by_cell = {}
                for r in merged:
                    by_cell.setdefault((r["stage"], r["pu"]), []).append(r)
                for (stage, pu), rs in by_cell.items():
                    rs.sort(key=lambda r: r["run"])
                    cells.append(
                        {
                            "device": device,
                            "app": app,
                            "scenario": scenario,
                            "stage": stage,
                            "pu": pu,
                            "raw": [_raw_row(r) for r in rs],
                            "agg": _aggregate(rs, MAX_CV_LOOSE),
                            "agg_strict_dropped": _aggregate(rs, MAX_CV_STRICT) is None,
                            "flags": {
                                "high_cv": any(r["timing"]["cv"] > MAX_CV_STRICT for r in rs)
                            },
                        }
                    )
    return cells, coverage


# --------------------------------------------------------------------------
# schedules (Section 4: the z3 stage->PU assignment + the measured payoff)
# --------------------------------------------------------------------------
def _normalize_schedule(raw, n_stages):
    """Collapse the two on-disk serializations into one canonical schedule.

    NEW (== schedule.schema.json): chunks carry 1-based start_stage/end_stage
    (+ 'hardware' iff GPU). OLD: chunks carry a 0-based 'stages' list and a
    sibling 'stage_assignments' dict (no start/end, no hardware). Both keep
    'time' (predicted ms) and a 'metrics' block. We deliberately DROP the
    predicted 'speedup_over_*' here -- they are bogus (units mismatch, e.g.
    7.58x where the measured value is 1.79x) and must never headline.
    """
    chunks_raw = raw.get("chunks", [])
    is_new = bool(chunks_raw) and "start_stage" in chunks_raw[0]
    chunks = []
    for c in chunks_raw:
        if is_new:
            start, end = c["start_stage"], c["end_stage"]
            hardware = c.get("hardware")
        else:
            stages = c["stages"]
            start, end = min(stages) + 1, max(stages) + 1  # 0-based -> 1-based
            hardware = None  # OLD format never recorded the GPU backend
        chunks.append(
            {
                "core_type": c["core_type"],
                "hardware": hardware,
                "start_stage": start,
                "end_stage": end,
                "time": c.get("time"),
            }
        )
    chunks.sort(key=lambda c: c["start_stage"])
    covers = chunks and chunks[0]["start_stage"] == 1 and chunks[-1]["end_stage"] == n_stages
    return {
        "uid": raw.get("uid", "?"),
        "solution_id": raw.get("solution_id"),
        "format": "new" if is_new else "old",
        "makespan": (raw.get("metrics") or {}).get("max_time"),
        "covers": bool(covers),
        "chunks": chunks,
    }


def _schedule_validator():
    from jsonschema import Draft202012Validator

    schema = json.loads(read_text(os.path.join(SCHEMA_DIR, "schedule.schema.json")))
    return Draft202012Validator(schema)


def _sched_root(table_type):
    """Resolve the schedule store root for a table_type. The current layout gives
    each table_type its own top-level dir (data/schedules_<tt>); older data used a
    single data/schedules root holding both. Prefer the per-table dir if present."""
    per_table = os.path.join(REPO_ROOT, "data", f"schedules_{table_type}")
    return per_table if os.path.isdir(per_table) else SCHED_ROOT


def build_schedules(app_stages):
    """Walk data/schedules; per (device, app, backend) collect the z3 candidate
    schedules for each (table_type, mode), normalised across both formats, with
    the whole-pipeline single-PU baseline the speedup is measured against.

    Returns (cells, stats). NEW-format files are schema-validated; a failure is
    flagged (validated=False) + warned rather than aborting the whole build --
    the dashboard should still render every other cell that is fine."""
    cells, stats = [], {"files": 0, "new": 0, "old": 0, "invalid": 0}
    roots = {tt: _sched_root(tt) for tt in TABLE_TYPES}
    existing = sorted({r for r in roots.values() if os.path.isdir(r)})
    if not existing:
        return cells, stats
    validator = _schedule_validator()
    devices = sorted(
        {d for r in existing for d in os.listdir(r) if os.path.isdir(os.path.join(r, d))}
    )
    for device in devices:
        for app in APPS:
            n_stages = app_stages[app]
            for backend in SCHED_BACKENDS:
                variants = []
                for tt in TABLE_TYPES:
                    for mode in MODES:
                        path = Case(device, app, backend).schedule_path(roots[tt], tt, mode)
                        if not os.path.isfile(path):
                            continue
                        stats["files"] += 1
                        raw = json.loads(read_text(path))
                        is_new = bool(raw) and "start_stage" in raw[0].get("chunks", [{}])[0]
                        stats["new" if is_new else "old"] += 1
                        validated = is_new
                        if is_new:
                            errs = sorted(validator.iter_errors(raw), key=lambda e: list(e.path))
                            if errs:
                                validated = False
                                stats["invalid"] += 1
                                print(
                                    f"Warning: {os.path.relpath(path, REPO_ROOT)} fails "
                                    f"schedule schema ({errs[0].message}) -- flagged, kept"
                                )
                        scheds = [_normalize_schedule(s, n_stages) for s in raw]
                        covering = [s for s in scheds if s["covers"] and s["makespan"] is not None]
                        best = min(covering, key=lambda s: s["makespan"]) if covering else None
                        variants.append(
                            {
                                "table_type": tt,
                                "mode": mode,
                                "format": "new" if is_new else "old",
                                "validated": validated,
                                "n_candidates": len(scheds),
                                "best_uid": best["uid"] if best else None,
                                "schedules": scheds,
                            }
                        )
                if not variants:
                    continue
                try:
                    baseline = get_baseline_for_config(device, app, backend, PROF_ROOT)
                except Exception:
                    baseline = None
                cells.append(
                    {
                        "device": device,
                        "app": app,
                        "backend": backend,
                        "n_stages": n_stages,
                        "baseline": baseline,
                        "variants": variants,
                    }
                )
    return cells, stats


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _first_num(text):
    m = _NUM_RE.search(text or "")
    return float(m.group()) if m else None


def _last_num(text):
    """The ms value in a summary cell is always the trailing number ('VK 5.67',
    'SCH-L1G6  7.25') -- a uid token like 'L1G6' would fool _first_num."""
    nums = _NUM_RE.findall(text or "")
    return float(nums[-1]) if nums else None


def _row_caveat(device_id, app, backend):
    """Match a measured row to the prose caveats in speedup-summary.md so the
    UI can flag it inline (keeps the matching in one place, not duplicated in JS)."""
    if device_id == "jetson" and backend == "cu":
        return "timing-only (Jetson managed-mem correctness bug)"
    if app == "tree" and ((device_id == "jetson" and backend == "vk") or device_id == "minipc"):
        return "z3 solved on BTPM/interference but the baseline ran isolated; tree is tiny so pure-GPU wins — not a framework defect"
    if device_id == "R5CY21Y3VEV" and app == "cifar-sparse":
        return "only the best-predicted schedule was run (CPU-only candidates too slow to sweep)"
    return None


def parse_speedup_summary():
    """Parse the hand-maintained data/sched_logs/speedup-summary.md into machine
    rows + prose blocks. Defensive: it is hand-edited Markdown (the '9 stages'
    prose is wrong for tree, device names are friendly, the speedup carries a
    'x'). Unparseable rows / unmapped devices are warned, not fatal."""
    if not os.path.isfile(SCHED_SUMMARY):
        return {"generated_note": "", "rows": [], "reading": [], "tree_losses": "", "caveats": []}
    lines = read_text(SCHED_SUMMARY).splitlines()

    rows, reading, caveats, tree_losses_parts = [], [], [], []
    generated_note = ""
    section = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            low = s.lower()
            section = (
                "reading"
                if "reading" in low
                else "losses"
                if "loss" in low
                else "caveats"
                if "caveat" in low
                else None
            )
            continue
        if not generated_note and s.startswith("Measured"):
            generated_note = s
        if s.startswith("|") and "|" in s[1:]:
            cells = [c.strip() for c in s.strip("|").split("|")]
            # skip the header row and the |---|---| separator row
            if len(cells) < 6 or cells[0].lower() == "device" or set(cells[0]) <= {"-", ":"}:
                continue
            device_label, app, backend, base_lbl, best_lbl, speedup_lbl = cells[:6]
            speedup = _first_num(speedup_lbl)
            device_id = SUMMARY_DEVICE_ALIAS.get(device_label.lower())
            if speedup is None:
                print(f"Warning: speedup-summary row unparseable, skipped: {s}")
                continue
            if device_id is None:
                print(
                    f"Warning: speedup-summary device '{device_label}' has no alias -> id; kept label only"
                )
            rows.append(
                {
                    "device_id": device_id,
                    "device_label": device_label,
                    "app": app,
                    "backend": backend,
                    "baseline_label": base_lbl,
                    "best_label": best_lbl,
                    "baseline_ms": _last_num(base_lbl),
                    "best_ms": _last_num(best_lbl),
                    "speedup": speedup,
                    "caveat": _row_caveat(device_id, app, backend),
                }
            )
        elif s.startswith("- ") and section in ("reading", "caveats"):
            (reading if section == "reading" else caveats).append(s[2:].strip())
        elif s and section == "losses":
            tree_losses_parts.append(s)
    return {
        "generated_note": generated_note,
        "rows": rows,
        "reading": reading,
        "tree_losses": " ".join(tree_losses_parts),
        "caveats": caveats,
    }


# --------------------------------------------------------------------------
# pipeline timelines (Section 4: the MEASURED per-chunk execution overlap)
# --------------------------------------------------------------------------
# Each data/sched_logs/<dir>/*.log (03_run_schedule.py output) records, per
# candidate schedule, the real execution interval of every pipeline chunk across
# ~100 pumped tasks. We turn those into a Gantt: the actual software-pipelining
# overlap (the CPU chunk of task N running concurrently with the GPU chunk of
# task N-1) that turns the z3 chunking into the measured speedup.
PIPE_WINDOW = 9  # Gantt window width, in task-periods of wall time (legible slice)
PIPE_WARMUP = 5  # skip the first few tasks (cold caches / ramp-up)

# friendly/abbreviated dir tokens -> canonical ids (devices/*.json + APPS)
PIPE_DEVICE_ALIAS = {
    "jetson": "jetson", "minipc": "minipc", "samsung": "R5CY21Y3VEV",
    "R5CY21Y3VEV": "R5CY21Y3VEV", "3A021JEHN02756": "3A021JEHN02756",
}
PIPE_APP_ALIAS = {
    "tree": "tree", "dense": "cifar-dense", "cifar-dense": "cifar-dense",
    "sparse": "cifar-sparse", "cifar-sparse": "cifar-sparse",
}
# chunk bracket label -> (core_type, hardware), matching schedule.schema.json's
# vocabulary so the JS can reuse chunkColor()/PU_COLOR verbatim.
PIPE_CHUNK_LABEL = {
    "CUDA": ("GPU", "gpu_cuda"),
    "Vulkan": ("GPU", "gpu_vulkan"),
    "OMP/Little": ("Little", None),
    "OMP/Medium": ("Medium", None),
    "OMP/Big": ("Big", None),
    "OMP/Super": ("Super", None),
}


def _resolve_pipe_name(name):
    """A log dir/file name -> (device_id, app, backend|None). Backend may be None
    when the name omits it (e.g. 'samsung_tree'); inferred from the log later."""
    toks = name.split("_")
    dev = PIPE_DEVICE_ALIAS.get(toks[0])
    app = next((PIPE_APP_ALIAS[t] for t in toks[1:] if t in PIPE_APP_ALIAS), None)
    if not dev or not app:
        return None
    be = next((t for t in toks[1:] if t in ("cu", "vk")), None)
    return dev, app, be


def _parse_pipe_headers(text):
    """UID -> ordered chunks [{core_type, hardware, start_stage, end_stage}]."""
    heads, cur = {}, None
    for ln in text.splitlines():
        m = re.match(r"\s*Schedule \d+ \[UID:\s*([^\]]+)\]", ln)
        if m:
            cur = m.group(1).strip()
            heads[cur] = []
            continue
        m = re.match(r"\s*Chunk \d+ \[([^\]]+)\]:\s*(.*)", ln)
        if m and cur is not None:
            ct, hw = PIPE_CHUNK_LABEL.get(m.group(1).strip(), (m.group(1).strip(), None))
            stages = [int(x) for x in re.findall(r"\d+", m.group(2))]
            heads[cur].append({
                "core_type": ct, "hardware": hw,
                "start_stage": min(stages) if stages else None,
                "end_stage": max(stages) if stages else None,
            })
    return heads


def _parse_pipe_timelines(text):
    """UID -> {freq, rows:[(task, chunk, start_tick, end_tick)]}. Each block is
    self-identifying (Schedule_UID=/Frequency= lines) so we never rely on order."""
    blocks, cur = {}, None
    for ln in text.splitlines():
        if ln.startswith("Schedule_UID="):
            cur = ln.split("=", 1)[1].strip()
            blocks.setdefault(cur, {"freq": None, "rows": []})
        elif ln.startswith("Frequency=") and cur:
            m = re.search(r"Frequency=(\d+)", ln)
            if m:
                blocks[cur]["freq"] = int(m.group(1))
        elif ln.startswith("Task=") and cur:
            m = re.match(r"Task=(\d+) Chunk=(\d+) Start=(\d+) End=(\d+)", ln)
            if m:
                blocks[cur]["rows"].append(tuple(int(g) for g in m.groups()))
    return blocks


def _coverage_time(intervals, min_cover):
    """Total wall-time during which >= min_cover of the intervals overlap."""
    events = []
    for a, b in intervals:
        events += [(a, 1), (b, -1)]
    events.sort()
    total, cov, prev = 0.0, 0, None
    for t, d in events:
        if prev is not None and cov >= min_cover:
            total += t - prev
        cov, prev = cov + d, t
    return total


def _build_pipe_schedule(uid, chunks, block):
    """One candidate's measured Gantt: a steady-state task window + metrics."""
    freq, rows = block["freq"], block["rows"]
    if not freq or not rows:
        return None
    t0 = min(r[2] for r in rows)  # global first start tick -> ms origin
    ms = lambda tick: (tick - t0) * 1000.0 / freq  # noqa: E731
    n_tasks = max(r[0] for r in rows) + 1

    # makespan/task = total wall / n_tasks (== speedup-summary's "ms = wall/100")
    span_all = ms(max(r[3] for r in rows)) - ms(min(r[2] for r in rows))
    makespan_per_task = span_all / n_tasks if n_tasks else None

    # visible window: a fixed TIME slice (~PIPE_WINDOW task-periods) of steady
    # state, holding every chunk interval that overlaps it. Windowing by time
    # (not by task index) keeps every lane back-to-back: a fast producer lane
    # would otherwise look like it stops once it races ahead of the bottleneck.
    lo = PIPE_WARMUP if n_tasks > 2 * PIPE_WARMUP else 0
    base = min(ms(r[2]) for r in rows if r[0] == lo)
    dur = (makespan_per_task or (span_all / n_tasks if n_tasks else 0)) * PIPE_WINDOW
    win = [r for r in rows if ms(r[3]) > base and ms(r[2]) < base + dur]
    window = [
        {"task": r[0], "chunk": r[1],
         "t0": round(ms(r[2]) - base, 3), "t1": round(ms(r[3]) - base, 3)}
        for r in win
    ]

    # steady-state concurrency: over the shown window, the share of *busy* time
    # during which >= 2 PU lanes run at once (intervals clipped to the window).
    # Measured over the window (not the whole run) so it matches the Gantt: the
    # unthrottled producer races ahead and idles during the queue-drain tail,
    # which would otherwise dilute a full-run number below the visible overlap.
    clip = [(max(ms(r[2]), base), min(ms(r[3]), base + dur)) for r in win]
    clip = [(a, b) for a, b in clip if b > a]
    active = _coverage_time(clip, 1)
    concurrency = 100.0 * _coverage_time(clip, 2) / active if active > 0 else 0.0
    return {
        "uid": uid,
        "chunks": chunks,
        "window": window,
        "window_ms": round(dur, 2),
        "metrics": {
            "makespan_per_task_ms": round(makespan_per_task, 3) if makespan_per_task else None,
            "concurrency_pct": round(concurrency, 1),
            "n_tasks": n_tasks,
        },
    }


def build_pipelines():
    """Walk data/sched_logs, parse each run log into measured per-chunk Gantts.

    Returns (cells, stats). Dir names use two conventions (old + new) and some
    cells have several dirs; we dedupe by (device, app, backend) keeping the
    newest log file. Unresolvable dirs are warned and skipped, never fatal."""
    cells, stats = [], {"dirs": 0, "skipped": 0}
    if not os.path.isdir(SCHED_LOGS_ROOT):
        return cells, stats

    # candidate logs: <dir>/*.log (name = dir) and top-level *.log (name = stem)
    candidates = {}  # (dev, app, be_hint) -> (mtime, path)
    logs = glob.glob(os.path.join(SCHED_LOGS_ROOT, "*", "*.log"))
    logs += glob.glob(os.path.join(SCHED_LOGS_ROOT, "*.log"))
    for path in logs:
        parent = os.path.dirname(path)
        name = (
            os.path.splitext(os.path.basename(path))[0]
            if os.path.samefile(parent, SCHED_LOGS_ROOT)
            else os.path.basename(parent)
        )
        resolved = _resolve_pipe_name(name)
        if not resolved:
            stats["skipped"] += 1
            continue
        mtime = os.path.getmtime(path)
        if resolved not in candidates or mtime > candidates[resolved][0]:
            candidates[resolved] = (mtime, path)

    # collapse (dev, app, be|None) onto a concrete backend, newest log wins
    chosen = {}  # (dev, app, be) -> (mtime, path, text)
    for (dev, app, be), (mtime, path) in candidates.items():
        text = read_text(path)
        if be is None:  # infer from the GPU chunk label in this log
            be = "cu" if "[CUDA" in text else "vk" if "[Vulkan" in text else None
        if be is None:
            stats["skipped"] += 1
            continue
        k = (dev, app, be)
        if k not in chosen or mtime > chosen[k][0]:
            chosen[k] = (mtime, path, text)

    for (dev, app, be), (_, path, text) in sorted(chosen.items()):
        stats["dirs"] += 1
        heads = _parse_pipe_headers(text)
        blocks = _parse_pipe_timelines(text)
        scheds = []
        for uid, block in blocks.items():
            if uid not in heads:
                continue
            s = _build_pipe_schedule(uid, heads[uid], block)
            if s:
                scheds.append(s)
        if not scheds:
            stats["skipped"] += 1
            continue
        best = min(
            (s for s in scheds if s["metrics"]["makespan_per_task_ms"] is not None),
            key=lambda s: s["metrics"]["makespan_per_task_ms"],
            default=None,
        )
        # best first, then ascending measured makespan (nulls last)
        scheds.sort(key=lambda s: (
            0 if best and s["uid"] == best["uid"] else 1,
            s["metrics"]["makespan_per_task_ms"]
            if s["metrics"]["makespan_per_task_ms"] is not None else float("inf"),
        ))
        cells.append({
            "device": dev,
            "app": app,
            "backend": be,
            "best_uid": best["uid"] if best else None,
            "source": os.path.relpath(path, REPO_ROOT),
            "schedules": scheds,
        })
    return cells, stats


# --------------------------------------------------------------------------
# assemble + write
# --------------------------------------------------------------------------
def main():
    vocab = load_vocab()

    # device -> native GPU backend (long), for the CPU-tier merge rule
    dev_gpu = {}
    for path in glob.glob(os.path.join(REPO_ROOT, "devices", "*.json")):
        d = json.loads(read_text(path))
        if d.get("gpu"):
            dev_gpu[d["id"]] = d["gpu"]["backend"]

    cells, coverage = build_profiling(dev_gpu)
    devices = build_devices(coverage)
    apps = build_apps(vocab)
    sched_cells, sched_stats = build_schedules(vocab["app_stages"])
    measured = parse_speedup_summary()
    pipe_cells, pipe_stats = build_pipelines()

    bundle = {
        "generated_by": "dashboard/generate.py",
        "git_sha": git_sha(),
        "vocab": {
            "processor_types": vocab["processor_types"],
            "backends": vocab["backends"],
            "app_stages": vocab["app_stages"],
        },
        "devices": devices,
        "apps": apps,
        "profiling": {
            "metrics": METRICS,
            "scenarios": SCENARIOS,
            "pus": PU_ORDER,
            "cells": cells,
            "coverage": coverage,
        },
        "schedules": {
            "table_types": TABLE_TYPES,
            "modes": MODES,
            "cells": sched_cells,
            "measured": measured,
            "pipelines": pipe_cells,
        },
    }

    # write site/ = static/ + bundle.js
    if os.path.isdir(SITE_DIR):
        shutil.rmtree(SITE_DIR)
    shutil.copytree(STATIC_DIR, SITE_DIR)
    payload = "window.BT_DATA = " + json.dumps(bundle, ensure_ascii=False) + ";\n"
    with open(os.path.join(SITE_DIR, "bundle.js"), "w", encoding="utf-8") as f:
        f.write(payload)

    # summary
    n_stage = sum(len(a["stages"]) for a in apps)
    n_with_data = sum(1 for d in devices if d["has_data"])
    leaves = sum(len(s) for dev in coverage.values() for app in dev.values() for s in app.values())
    print(
        f"dashboard: {len(devices)} devices ({n_with_data} with profiling), "
        f"{len(apps)} apps / {n_stage} stages, "
        f"{len(cells)} profiling cells, {leaves} collected (device,app,backend,scenario) leaves"
    )
    invalid_note = f", {sched_stats['invalid']} schema-invalid" if sched_stats["invalid"] else ""
    print(
        f"schedules: {len(sched_cells)} cells, {sched_stats['files']} files "
        f"({sched_stats['new']} new/validated, {sched_stats['old']} old/flagged{invalid_note}), "
        f"{len(measured['rows'])} measured speedup rows"
    )
    n_pipe_sched = sum(len(c["schedules"]) for c in pipe_cells)
    print(
        f"pipelines: {len(pipe_cells)} measured Gantt cells from {pipe_stats['dirs']} logs "
        f"({n_pipe_sched} candidate timelines, {pipe_stats['skipped']} skipped)"
    )
    print(f"bundle.js: {os.path.getsize(os.path.join(SITE_DIR, 'bundle.js')) / 1024:.0f} KiB")
    print(f"open: {os.path.join(SITE_DIR, 'index.html')}")


if __name__ == "__main__":
    main()
