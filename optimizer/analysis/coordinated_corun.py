#!/usr/bin/env python3
"""Coordinated co-run study: calibration, paced-alone and co-run tables, trade-off curves, and the
prediction grid with the best coordinated settings (design D4) from the raw phone outputs under
data/coordinated-corun/<serial>/ (openspec/changes/coordinated-corun).

Raw layout (written by data/coordinated-corun/drivers/):
  alone*/<model>_<cfg>_r<k>[_retry].{out,err,clk,meta}            lpa_pace alone, 3 rounds, 40 s
  calib/...                                                       the 30-s calibration runs (task 1.3)
  corun-<stage>/<model>_<A>_<B>_<params>_t<k>[_retry]/{A,B}.{out,err}, corun.{meta,clk}
lpa_pace prints one JSONL line per step (a prefill ubatch or a decode token: epoch start, end,
gap_end and tokens) and a summary line. Configuration names:
  gpuP_p<P>_u<U>[_c<C>]   GPU prefill of P tokens in ubatches of C (default min(512, P)) at duty U/100
  gpuD_d<d>_u<U>          GPU decode at KV depth d at duty U/100
  cpuD_d<d>_m<mask>       CPU decode at depth d on a hex core mask (one thread per core)
  cpuP_p<P>_m<mask>       CPU prefill
A co-run name joins the GPU process (A) and the CPU process (B), e.g. gpuP_cpuD_p512_u50_m3f0.

Every value is the median of its runs or trials with [min-max]; a _retry replaces the run it repeats.
In a co-run each side counts only its units (a whole prefill repetition, or a decode token) that lie
wholly inside the partner's measured interval (its first step's start to its last step's end). A
unit's wall time includes the gap after it, so a paced side's rate is its rate over wall time.

The prediction is llm-phase-affinity's design D5 with the co-run ratios of this study (co-run rate
over the harness's own alone rate at duty 1), applied to the harness's alone rates at prompt / KV
depth P; each pipeline takes the measured setting that maximizes it, per mix. The baseline is the
larger of the best coordinated data-parallel, the uncoordinated data-parallel of llm-phase-affinity
and the best single PU (of either study).

  uv run optimizer/analysis/coordinated_corun.py [--data data/coordinated-corun] [--json out.json]
"""

import argparse
import glob
import json
import os
import re
import statistics

from analysis import llm_phase_affinity as lpa

ROOT = lpa.ROOT
DATA = os.path.join(ROOT, "data", "coordinated-corun")
SERIAL = "R5CY21Y3VEV"
PROMPTS, OUTPUTS, GATE = lpa.PROMPTS, lpa.OUTPUTS, lpa.GATE
# The CPU prefill of every pipeline and of data-parallel runs on the widest mask (the fastest alone).
CPU_PREFILL_MASK = "3f0"
KEYS = {"p": "P", "d": "depth", "u": "u", "m": "mask", "c": "C"}
# Samsung CPU frequency domains: cores 4-6 (medium), 7-8 (big), 9 (prime).
CPU_POLICIES = ("policy4", "policy7", "policy9")


def parse_key(key):
    """'1b_gpuP_cpuD_p512_u50_m3f0' -> {model, phases: [gpuP, cpuD], P: 512, u: 0.5, mask: '3f0', C: 512}."""
    toks = key.split("_")
    out = {"model": toks[0], "phases": []}
    for t in toks[1:]:
        if re.fullmatch(r"(gpu|cpu)[PD]", t):
            out["phases"].append(t)
        else:
            out[KEYS[t[0]]] = t[1:] if t[0] == "m" else int(t[1:])
    out["u"] = out.get("u", 100) / 100
    if "P" in out:
        out.setdefault("C", min(512, out["P"]))
    return out


def pace_out(path):
    """(steps, summary) of one lpa_pace stdout: steps as dicts with rep, step, start, end, gap_end,
    tokens; summary None if the run did not finish."""
    steps, summary = [], None
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if line.startswith("{"):
                r = json.loads(line)
                if r.get("summary"):
                    summary = r
                else:
                    steps.append(r)
    return steps, summary


def clock_in(clk_path, lo=None, hi=None, field="gpu"):
    """Median clock (MHz; sysfs reads kHz) of a field of the .clk samples (gpu, policy4, ...) between
    lo and hi, over the positive samples."""
    vals = []
    if os.path.exists(clk_path):
        for line in open(clk_path):
            t = re.match(r"(\d+\.\d+) ", line)
            m = re.search(rf"\b{field}=(\d+)", line)
            if t and m and int(m.group(1)) > 0:
                if (lo is None or float(t.group(1)) >= lo) and (
                    hi is None or float(t.group(1)) <= hi
                ):
                    vals.append(int(m.group(1)) / 1000)
    return statistics.median(vals) if vals else None


def rel(path):
    return os.path.relpath(path, ROOT)


def load_alone(directory):
    """{key: {"wall", "compute", "duty", "clock_mhz": stat, "thermal": [..], "failed": [..], "cfg"}}."""
    out = {}
    for key, paths in lpa.run_files(directory, r"\w+").items():
        acc = {"wall": ([], []), "compute": ([], []), "duty": ([], []), "clock_mhz": ([], [])}
        failed, thermal = [], []
        for path in paths:
            base = path[: -len(".out")]
            _, s = pace_out(path)
            if not s:
                err = open(base + ".err", errors="replace").read().strip().splitlines()[-3:]
                failed.append({"file": rel(path), "error": err})
                continue
            for field, v in (
                ("wall", s["rate_wall"]),
                ("compute", s["rate_compute"]),
                ("duty", s["duty"]),
            ):
                acc[field][0].append(v)
                acc[field][1].append(rel(path))
            clk = clock_in(base + ".clk")
            if clk:
                acc["clock_mhz"][0].append(clk)
                acc["clock_mhz"][1].append(rel(path))
            thermal.append(lpa.thermal_after(base + ".meta"))
        out[key] = {
            **{k: lpa.stat(*v) for k, v in acc.items()},
            "thermal": thermal,
            "failed": failed,
            "cfg": parse_key(key),
        }
    return out


def units(steps, prefill):
    """Measured units of one side: a whole prefill repetition (all its ubatches) or a decode token, as
    (start, end_of_compute, gap_end, compute_s, tokens)."""
    if not prefill:
        return [
            (s["start"], s["end"], s["gap_end"], s["end"] - s["start"], s["tokens"]) for s in steps
        ]
    reps = {}
    for s in steps:
        reps.setdefault(s["rep"], []).append(s)
    n_steps = max((len(v) for v in reps.values()), default=0)
    return [
        (
            v[0]["start"],
            v[-1]["end"],
            v[-1]["gap_end"],
            sum(s["end"] - s["start"] for s in v),
            sum(s["tokens"] for s in v),
        )
        for v in reps.values()
        if len(v) == n_steps
    ]


def side_rates(mine, partner_interval):
    """Rates over my units wholly inside the partner's measured interval: tokens over their wall time
    (gap after each unit included) and over their compute time, the duty, the counted span in seconds,
    and (counted, total) units."""
    lo, hi = partner_interval
    kept = [u for u in mine if u[0] >= lo and u[2] <= hi]
    if not kept:
        return None
    tokens = sum(u[4] for u in kept)
    wall = sum(u[2] - u[0] for u in kept)
    comp = sum(u[3] for u in kept)
    return {
        "wall": tokens / wall,
        "compute": tokens / comp,
        "duty": comp / wall,
        "span_s": wall,
        "kept": (len(kept), len(mine)),
    }


def load_corun(directory):
    """{key: {"A": {...stats}, "B": {...}, "trials": [...], "cfg"}}: per trial each side's rates over
    its units inside the partner's measured interval; the stats are over trials."""
    out = {}
    for key, dirs in lpa.trial_dirs(directory).items():
        cfg = parse_key(key)
        sides = {"A": cfg["phases"][0], "B": cfg["phases"][1]}
        acc = {x: {f: ([], []) for f in ("wall", "compute", "duty", "span_s")} for x in sides}
        clocks, trials = ([], []), []
        cpu_clocks = {pol: [] for pol in CPU_POLICIES}
        for d in dirs:
            steps = {x: pace_out(os.path.join(d, f"{x}.out"))[0] for x in sides}
            if not all(steps.values()):
                trials.append({"trial": rel(d), "error": "a side has no steps"})
                continue
            interval = {x: (steps[x][0]["start"], steps[x][-1]["end"]) for x in sides}
            t = {"trial": rel(d), "thermal_after": lpa.thermal_after(os.path.join(d, "corun.meta"))}
            for x, o in (("A", "B"), ("B", "A")):
                r = side_rates(units(steps[x], sides[x].endswith("P")), interval[o])
                t[x] = r
                if r:
                    for f in acc[x]:
                        acc[x][f][0].append(r[f])
                        acc[x][f][1].append(rel(d))
            lo, hi = (
                max(interval["A"][0], interval["B"][0]),
                min(interval["A"][1], interval["B"][1]),
            )
            clk = clock_in(os.path.join(d, "corun.clk"), lo, hi)
            t["clock_mhz"] = clk
            if clk:
                clocks[0].append(clk)
                clocks[1].append(rel(d))
            for pol in CPU_POLICIES:
                c = clock_in(os.path.join(d, "corun.clk"), lo, hi, pol)
                if c:
                    cpu_clocks[pol].append(c)
            trials.append(t)
        out[key] = {
            **{x: {f: lpa.stat(*v) for f, v in a.items()} for x, a in acc.items()},
            "clock_mhz": lpa.stat(*clocks),
            "cpu_mhz": {pol: lpa.stat(v) for pol, v in cpu_clocks.items()},
            "trials": trials,
            "cfg": cfg,
        }
    return out


def load(serial=SERIAL):
    """(alone, corun) of one phone: every alone* and corun-* directory merged (names are unique)."""
    alone, corun = {}, {}
    for d in sorted(glob.glob(os.path.join(DATA, serial, "alone*"))):
        alone.update(load_alone(d))
    for d in sorted(glob.glob(os.path.join(DATA, serial, "corun-*"))):
        corun.update(load_corun(d))
    return alone, corun


# ----------------------------------------------------------------------------- prediction (design D4)


def spec_throughput(prompt, gen, prefill, decode, handoff_s):
    """Specialized pipeline (llm-phase-affinity D5): one request per period max(P/p', D/d') + h."""
    return 1 / (max(prompt / prefill, gen / decode) + handoff_s)


def dp_throughput(prompt, gen, pus):
    """Data-parallel: each PU serves whole requests; pus = [(prefill rate, decode rate)]."""
    return sum(1 / (prompt / p + gen / d) for p, d in pus)


def near(prompt):
    """The measured prompt length whose co-run ratios a mix uses (128 borrows 512, as in D5)."""
    return prompt if prompt in (512, 2048) else 512


def med(s):
    return s["median"] if s else None


class Refs:
    """The harness's alone rates (duty 1) of one model: prefill at P, decode at KV depth d."""

    def __init__(self, alone, model):
        self.alone, self.model = alone, model

    def get(self, name):
        e = self.alone.get(f"{self.model}_{name}")
        if not e or not e["wall"]:
            raise KeyError(f"{self.model}_{name}")
        return e["wall"]

    def gpu_prefill(self, prompt):
        return self.get(f"gpuP_p{prompt}_u100")

    def gpu_decode(self, depth):
        return self.get(f"gpuD_d{depth}_u100")

    def cpu_prefill(self, prompt, mask=CPU_PREFILL_MASK):
        return self.get(f"cpuP_p{prompt}_m{mask}")

    def cpu_decode(self, depth, mask):
        return self.get(f"cpuD_d{depth}_m{mask}")


def settings(corun, refs):
    """Co-run ratios per measured setting, from the co-runs of refs.model with 3 trials on both sides.
    Ratios are co-run wall rates over the duty-1 alone rate of the same phase (so pacing's own cost is
    in them): prefill against the alone prefill at the co-run's P and ubatch 512 (or P if smaller),
    decode (depth-0 loops) against the alone decode at depth 0 on the same mask."""
    out = {"gpu->cpu": [], "cpu->gpu": [], "dp": []}
    for key, c in sorted(corun.items()):
        cfg = c["cfg"]
        if cfg["model"] != refs.model:
            continue
        a, b = c["A"]["wall"], c["B"]["wall"]
        if not a or not b or a["n"] < 3 or b["n"] < 3:
            continue
        s = {"key": key, "u": cfg["u"], "mask": cfg["mask"], "P": cfg.get("P"), "C": cfg.get("C")}
        try:
            if cfg["phases"] == ["gpuP", "cpuD"]:
                pref, dref = refs.gpu_prefill(cfg["P"]), refs.cpu_decode(0, cfg["mask"])
                fam = "gpu->cpu"
            elif cfg["phases"] == ["gpuD", "cpuP"]:
                # A = GPU decode, B = CPU prefill: the prefill side is B.
                a, b = b, a
                pref, dref = refs.cpu_prefill(cfg["P"], cfg["mask"]), refs.gpu_decode(0)
                fam = "cpu->gpu"
            elif cfg["phases"] == ["gpuD", "cpuD"]:
                pref, dref = refs.gpu_decode(0), refs.cpu_decode(0, cfg["mask"])
                fam = "dp"
            else:
                continue
        except KeyError:
            continue
        # spec: (prefill ratio, decode ratio); dp: (GPU decode ratio, CPU decode ratio).
        s["ratios"] = (a["median"] / pref["median"], b["median"] / dref["median"])
        # The trial spread: each co-run rate at its slowest and fastest trial.
        s["ratios_lo"] = (a["min"] / pref["median"], b["min"] / dref["median"])
        s["ratios_hi"] = (a["max"] / pref["median"], b["max"] / dref["median"])
        s["files"] = lpa.files_of(a, b, pref, dref)
        out[fam].append(s)
    return out


def predict(prompt, gen, refs, sets, handoff_s, uncoord):
    """One mix: the best coordinated specialized pipeline and data-parallel with their settings, the
    baseline and the gain. uncoord: llm-phase-affinity's grid row for the mix (or None)."""
    al = {
        "gpu": (med(refs.gpu_prefill(prompt)), med(refs.gpu_decode(prompt))),
        "cpu": (med(refs.cpu_prefill(prompt)), None),
    }
    m = near(prompt)

    def spec_of(fam, s, key="ratios"):
        rp, rd = s[key]
        p = al["gpu" if fam == "gpu->cpu" else "cpu"][0] * rp
        d = med(refs.cpu_decode(prompt, s["mask"])) * rd if fam == "gpu->cpu" else al["gpu"][1] * rd
        return spec_throughput(prompt, gen, p, d, handoff_s)

    def dp_of(s, key="ratios"):
        rg, rc = s[key]
        pus = [
            (al["gpu"][0], al["gpu"][1] * rg),
            (al["cpu"][0], med(refs.cpu_decode(prompt, s["mask"])) * rc),
        ]
        return dp_throughput(prompt, gen, pus)

    spec = [
        (spec_of(fam, s), fam, s)
        for fam in ("gpu->cpu", "cpu->gpu")
        for s in sets[fam]
        if s["P"] == m
    ]
    dp = [(dp_of(s), s) for s in sets["dp"]]
    # Best single PU of this study's alone rates (GPU; CPU prefill on the widest mask, decode on any).
    masks = sorted({s["mask"] for f in sets.values() for s in f} | {CPU_PREFILL_MASK})
    singles = [("GPU", dp_throughput(prompt, gen, [al["gpu"]]))]
    for mk in masks:
        try:
            singles.append(
                (
                    f"CPU {mk}",
                    dp_throughput(prompt, gen, [(al["cpu"][0], med(refs.cpu_decode(prompt, mk)))]),
                )
            )
        except KeyError:
            pass
    single = max(singles, key=lambda x: x[1])
    best_spec = max(spec, key=lambda x: x[0]) if spec else None
    best_dp = max(dp, key=lambda x: x[0]) if dp else None
    cands = [
        ("coordinated data-parallel", best_dp[0] if best_dp else 0.0),
        (f"single PU ({single[0]})", single[1]),
    ]
    if uncoord:
        cands += [
            ("uncoordinated data-parallel (llm-phase-affinity)", uncoord["dp_corun"]),
            (f"single PU ({uncoord['single_pu'].upper()}, llm-phase-affinity)", uncoord["single"]),
        ]
    base = max(cands, key=lambda x: x[1])
    own = max(cands[:2], key=lambda x: x[1])  # this study's sustained rates only
    row = {
        "prompt": prompt,
        "gen": gen,
        "handoff_s": handoff_s,
        "spec": best_spec[0] if best_spec else None,
        "spec_family": best_spec[1] if best_spec else None,
        "spec_setting": best_spec[2]["key"] if best_spec else None,
        "dp": best_dp[0] if best_dp else None,
        "dp_setting": best_dp[1]["key"] if best_dp else None,
        "single": single[1],
        "single_pu": single[0],
        "baseline": base[1],
        "baseline_from": base[0],
        "gain": best_spec[0] / base[1] if best_spec else None,
        "gain_vs_coordinated_dp": best_spec[0] / best_dp[0] if best_spec and best_dp else None,
        "baseline_own": own[1],
        "gain_own": best_spec[0] / own[1] if best_spec else None,
        "uncoord": uncoord,
        "sources": sorted(
            set(
                lpa.files_of(
                    refs.gpu_prefill(prompt), refs.gpu_decode(prompt), refs.cpu_prefill(prompt)
                )
                + (best_spec[2]["files"] if best_spec else [])
                + (best_dp[1]["files"] if best_dp else [])
            )
        ),
    }
    # The trial spread of the gain: every pipeline's co-run rates at their slowest (fastest) trial
    # against every data-parallel setting's at their fastest (slowest), the other baselines unchanged.
    if spec:
        rest = max(x[1] for x in cands[1:])
        for tag, sk, dk in (("lo", "ratios_lo", "ratios_hi"), ("hi", "ratios_hi", "ratios_lo")):
            sp = max(spec_of(f, s_, sk) for _, f, s_ in spec)
            row[f"gain_{tag}"] = sp / max([rest] + [dp_of(s_, dk) for _, s_ in dp])
    # What coordination buys each side: the duty-1, widest-mask, ubatch-512 settings alone.
    plain = [
        x for x in spec if x[2]["u"] == 1 and x[2]["mask"] == "3f0" and x[2]["C"] == min(512, m)
    ]
    plain_dp = [x for x in dp if x[1]["u"] == 1 and x[1]["mask"] == "3f0"]
    row["spec_u1"] = max(x[0] for x in plain) if plain else None
    row["dp_u1"] = plain_dp[0][0] if plain_dp else None
    # Diagnostic (not the gate): data-parallel with each PU's prefill at its co-run rate next to the
    # other's decode (the duty-1 widest-mask specialized pairs), decode as in the best coordinated setting.
    pr = {f: [x[2] for x in plain if x[1] == f] for f in ("gpu->cpu", "cpu->gpu")}
    if best_dp and pr["gpu->cpu"] and pr["cpu->gpu"]:
        rg, rc = best_dp[1]["ratios"]
        row["dp_prefill_corun"] = dp_throughput(
            prompt,
            gen,
            [
                (al["gpu"][0] * pr["gpu->cpu"][0]["ratios"][0], al["gpu"][1] * rg),
                (
                    al["cpu"][0] * pr["cpu->gpu"][0]["ratios"][0],
                    med(refs.cpu_decode(prompt, best_dp[1]["mask"])) * rc,
                ),
            ],
        )
    return row


def uncoordinated_grid(model):
    """llm-phase-affinity's Samsung llama.cpp grid rows for a model, keyed by (P, D), or {}."""
    alone = lpa.load_alone(SERIAL, "llamacpp")
    coruns = lpa.load_coruns(SERIAL, "llamacpp")
    bw = lpa.load_bandwidth(os.path.join(lpa.DATA, SERIAL, "bandwidth"))
    handoff_bw = {"cpu": bw["cpu"]["cpu"], "gpu": bw["gpu"]["gpu"]}
    rows, _ = lpa.grid(SERIAL, "llamacpp", model, alone, coruns, handoff_bw)
    return {(r["prompt"], r["gen"]): r for r in rows or []}, handoff_bw


def rank_settings(alone, corun, model):
    """Each specialized setting on its own: its best gain over the mixes that use its P (with every
    measured data-parallel setting in the baseline). Used to pick the settings of stages B, C and F."""
    refs = Refs(alone, model)
    sets = settings(corun, refs)
    uncoord, handoff_bw = uncoordinated_grid(model)
    gbps = min(handoff_bw[c]["median"] for c in ("cpu", "gpu"))
    out = []
    for fam in ("gpu->cpu", "cpu->gpu"):
        for s in sets[fam]:
            only = {"gpu->cpu": [], "cpu->gpu": [], "dp": sets["dp"], fam: [s]}
            best = None
            for prompt in PROMPTS:
                if near(prompt) != s["P"]:
                    continue
                h = lpa.handoff_seconds(model, prompt, gbps, lpa.KV_RW_BYTES["llamacpp"])
                for gen in OUTPUTS:
                    try:
                        r = predict(prompt, gen, refs, only, h, uncoord.get((prompt, gen)))
                    except (KeyError, TypeError):
                        continue
                    if r["gain"] and (best is None or r["gain"] > best[0]):
                        best = (r["gain"], prompt, gen)
            if best:
                out.append({"key": s["key"], "family": fam, "gain": best[0], "mix": best[1:], **s})
    return sorted(out, key=lambda x: -x["gain"])


def grid(alone, corun, model):
    """Prediction rows for every mix, or (None, reason) when an input is missing."""
    refs = Refs(alone, model)
    sets = settings(corun, refs)
    uncoord, handoff_bw = uncoordinated_grid(model)
    gbps = min(handoff_bw[c]["median"] for c in ("cpu", "gpu"))
    rows = []
    for prompt in PROMPTS:
        h = lpa.handoff_seconds(model, prompt, gbps, lpa.KV_RW_BYTES["llamacpp"])
        for gen in OUTPUTS:
            try:
                rows.append(predict(prompt, gen, refs, sets, h, uncoord.get((prompt, gen))))
            except (KeyError, TypeError) as e:
                return None, f"missing input {e}"
    return rows, None


# ----------------------------------------------------------------------------- tables


def f(s, digits=1):
    return lpa.fmt_stat(s, digits)


def lpa_reference(lpa_alone, model, cfg):
    """The llm-phase-affinity median a duty-1 harness alone run is calibrated against (or None): GPU
    prefill against flash attention auto at P <= 512 and off at 2048 (the configurations the harness
    runs), decode against tg128 at the same depth, CPU on the same mask."""
    if cfg["u"] != 1 or cfg.get("C", 512) != min(512, cfg.get("P", 512)):
        return None
    if cfg["phases"][0] == "gpuP":
        name, kind, x = "gpufa0" if cfg["P"] == 2048 else "gpu", "pp", cfg["P"]
    elif cfg["phases"][0] == "gpuD":
        name, kind, x = "gpu", "tg", cfg["depth"]
    elif cfg["phases"][0] == "cpuD":
        name, kind, x = f"cpu{cfg['mask']}", "tg", cfg["depth"]
    else:
        name, kind, x = f"cpu{cfg['mask']}", "pp", cfg["P"]
    e = lpa_alone.get((model, name))
    return e[kind].get(x) if e else None


def alone_table(alone, lpa_alone):
    lines = [
        "| config | target duty | rate over wall (tok/s) | rate over compute | achieved duty | wall / duty-1 rate "
        "| llm-phase-affinity median | harness / that | GPU MHz | thermal after |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(alone, key=lambda k: (parse_key(k)["model"], parse_key(k)["phases"], k)):
        e = alone[key]
        cfg = e["cfg"]
        if not e["wall"]:
            lines.append(f"| {key} | FAILED: {e['failed'][:1]} |||||||||")
            continue
        base = re.sub(r"_u\d+", "_u100", key)
        one = alone.get(base, {}).get("wall")
        ref = lpa_reference(lpa_alone, cfg["model"], cfg)
        lines.append(
            f"| {key} | {cfg['u']:.2f} | {f(e['wall'])} | {f(e['compute'])} | {f(e['duty'], 3)} | "
            + (f"{e['wall']['median'] / one['median']:.3f}" if one else "—")
            + f" | {f(ref)} | "
            + (f"{e['wall']['median'] / ref['median']:.3f}" if ref else "—")
            + " | "
            + (
                f"{e['clock_mhz']['median']:.0f}"
                if e["clock_mhz"] and cfg["phases"][0].startswith("gpu")
                else "—"
            )
            + f" | {','.join(str(t) for t in e['thermal'])} |"
        )
    return "\n".join(lines)


def calib_table(calib, alone, lpa_alone):
    """Calibration (spec "The paced harness matches the reference rates"): the 30-s calibration runs
    against the llm-phase-affinity medians (llama-bench, 2-5 s per repetition) and against the 40-s
    alone runs that serve as the reference of every ratio."""
    lines = [
        "| config | target duty | calibration, 30 s (tok/s) | achieved duty | llm-phase-affinity median "
        "| calibration / that | 40-s alone reference | calibration / that |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(calib):
        e, cfg = calib[key], calib[key]["cfg"]
        ref = lpa_reference(lpa_alone, cfg["model"], cfg)
        ours = alone.get(key, {}).get("wall")
        lines.append(
            f"| {key} | {cfg['u']:.2f} | {f(e['wall'])} | {f(e['duty'], 3)} | {f(ref)} | "
            + (f"{e['wall']['median'] / ref['median']:.3f}" if ref and e["wall"] else "—")
            + f" | {f(ours)} | "
            + (f"{e['wall']['median'] / ours['median']:.3f}" if ours and e["wall"] else "—")
            + " |"
        )
    return "\n".join(lines)


def corun_table(corun, alone):
    """Per co-run: both sides' wall rates, ratios to duty-1 alone and to the same pacing alone, the
    achieved duty, counted units, the GPU clock over the overlap and the thermal status after."""
    lines = [
        "| co-run | GPU side (A) rate | / alone duty 1 | / paced alone | A duty | CPU side (B) rate | / alone "
        "| GPU MHz | CPU MHz, cores 4-6 / 7-8 / 9 | counted A, B per trial | counted s A, B (median) | thermal after |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(corun):
        c, cfg = corun[key], corun[key]["cfg"]
        m = cfg["model"]
        pa, pb = cfg["phases"]

        def alone_of(phase, paced):
            if phase == "gpuP":
                name = f"gpuP_p{cfg['P']}_u{round(cfg['u'] * 100) if paced else 100}"
                name += f"_c{cfg['C']}" if paced and cfg["C"] != min(512, cfg["P"]) else ""
            elif phase == "gpuD":
                name = f"gpuD_d0_u{round(cfg['u'] * 100) if paced else 100}"
            elif phase == "cpuD":
                name = f"cpuD_d0_m{cfg['mask']}"
            else:
                name = f"cpuP_p{cfg['P']}_m{cfg['mask']}"
            e = alone.get(f"{m}_{name}")
            return e["wall"] if e else None

        def r(x, y):
            return f"{x['median'] / y['median']:.2f}" if x and y else "—"

        a, b = c["A"]["wall"], c["B"]["wall"]
        kept = ", ".join(
            f"{t['A']['kept'][0]}/{t['A']['kept'][1]} · {t['B']['kept'][0]}/{t['B']['kept'][1]}"
            if t.get("A") and t.get("B")
            else "—"
            for t in c["trials"]
        )
        spans = (
            f"{c['A']['span_s']['median']:.0f}, {c['B']['span_s']['median']:.0f}"
            if a and b
            else "—"
        )
        lines.append(
            f"| {key} | {f(a)} | {r(a, alone_of(pa, False))} | {r(a, alone_of(pa, True))} | {f(c['A']['duty'], 2)} | "
            f"{f(b)} | {r(b, alone_of(pb, False))} | "
            + (f"{c['clock_mhz']['median']:.0f}" if c["clock_mhz"] else "—")
            + " | "
            + " / ".join(
                f"{c['cpu_mhz'][p]['median']:.0f}" if c["cpu_mhz"][p] else "—" for p in CPU_POLICIES
            )
            + f" | {kept} | {spans} | {','.join(str(t.get('thermal_after')) for t in c['trials'])} |"
        )
    return "\n".join(lines)


def tradeoff_table(sets, family):
    """The trade-off curve of one pairing: per setting, the two co-run ratios (to duty-1 alone)."""
    head = {
        "gpu->cpu": ("GPU prefill ratio", "CPU decode ratio"),
        "cpu->gpu": ("CPU prefill ratio", "GPU decode ratio"),
        "dp": ("GPU decode ratio", "CPU decode ratio"),
    }[family]
    lines = [
        f"| setting | duty | mask | P | C | {head[0]} | {head[1]} |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(sets[family], key=lambda s: (s["P"] or 0, s["mask"], s["C"] or 0, -s["u"])):
        lines.append(
            f"| {s['key']} | {s['u']:.2f} | {s['mask']} | {s['P'] or '—'} | {s['C'] or '—'} | "
            f"{s['ratios'][0]:.3f} | {s['ratios'][1]:.3f} |"
        )
    return "\n".join(lines)


def grid_table(rows):
    lines = [
        "| P | D | best coordinated specialized (setting) | best coordinated data-parallel (setting) | best single PU "
        "| uncoordinated: specialized, data-parallel, gain (llm-phase-affinity) | duty-1 3f0: specialized, data-parallel "
        "| baseline (from) | **gain** | gain, trial spread | gain vs coordinated data-parallel | gain vs this study's baseline only "
        "| diag.: data-parallel with co-run prefill |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        u = r["uncoord"]
        unc = (
            f"{max(u['specialized'].values()):.4f}, {u['dp_corun']:.4f}, {u['gain']:.2f}"
            if u
            else "—"
        )

        def g(x):
            return f"{x:.4f}" if x is not None else "—"

        def g2(x):
            return f"{x:.2f}" if x is not None else "—"

        lines.append(
            f"| {r['prompt']} | {r['gen']} | {g(r['spec'])} ({r['spec_setting']}) | {g(r['dp'])} ({r['dp_setting']}) | "
            f"{r['single']:.4f} ({r['single_pu']}) | {unc} | {g(r['spec_u1'])}, {g(r['dp_u1'])} | "
            f"{r['baseline']:.4f} ({r['baseline_from']}) | **{g2(r['gain'])}** | "
            f"{g2(r.get('gain_lo'))}-{g2(r.get('gain_hi'))} | "
            + (f"{r['gain_vs_coordinated_dp']:.2f}" if r["gain_vs_coordinated_dp"] else "—")
            + f" | {g2(r['gain_own'])} | {g(r.get('dp_prefill_corun'))} |"
        )
    return "\n".join(lines)


PIXEL = "3A021JEHN02756"
PIXEL_MASKS = ("f0", "c0", "30")


def pixel_ratios(corun, mask):
    """ExecuTorch co-run ratios on one CPU decode mask (design D5), each against its solo loop: GPU
    prefill next to a CPU decode, CPU decode next to a GPU prefill (P = 512), and both decoding."""
    used = []

    def r(key, side, solo):
        a, b = lpa.et_rate(corun, key, side), lpa.et_rate(corun, solo)
        used.extend((a, b))
        return a["median"] / b["median"]

    return {
        "prefill": r(f"gpuP_cpuD_p512_m{mask}", "A", "solo_gpuP_p512"),
        "decode": r(f"gpuP_cpuDs_p512_m{mask}", "B", f"solo_cpuDs_m{mask}"),
        "both_cpu": r(f"cpuDs_gpuD_m{mask}", "A", f"solo_cpuDs_m{mask}"),
        "both_gpu": r(f"gpuDs_cpuD_m{mask}", "A", "solo_gpuDs"),
        "files": lpa.files_of(*used),
    }


def pixel_grid():
    """The Pixel ExecuTorch 1B grid at P = 128 and 512 (co-runs at 512 only, design D5), or (None,
    reason): the best GPU-prefill / CPU-decode pipeline and the best data-parallel pair over the CPU
    decode masks, against the larger of that, llm-phase-affinity's data-parallel and the best single PU.
    The CPU prefill of data-parallel and of the single PU runs on f0 (the fastest alone)."""
    d = os.path.join(DATA, PIXEL)
    if not os.path.isdir(os.path.join(d, "corun-et")):
        return None, "no co-runs", {}
    alone = lpa.load_et_alone(os.path.join(d, "alone-et"))
    corun = lpa.load_et_corun(os.path.join(d, "corun-et"))
    gpu = lpa.pu_rates(alone, "1b", ["vk4w", "vk8da4w"])
    cpu = {m: lpa.pu_rates(alone, "1b", [f"xnn{m}"]) for m in PIXEL_MASKS}
    try:
        rt = {m: pixel_ratios(corun, m) for m in PIXEL_MASKS}
    except KeyError as e:
        return None, f"missing input {e}", {}
    u_alone = lpa.load_alone(PIXEL, "et")
    u_rows, _ = lpa.grid(
        PIXEL,
        "et",
        "1b",
        u_alone,
        lpa.load_coruns(PIXEL, "et"),
        (lambda bw: {"cpu": bw["cpu"]["cpu"], "gpu": bw["gpu"]["gpu"]})(
            lpa.load_bandwidth(os.path.join(lpa.DATA, PIXEL, "bandwidth"))
        ),
    )
    uncoord = {(r["prompt"], r["gen"]): r for r in u_rows or []}
    rows = []
    for prompt in (128, 512):
        gp, gd = gpu["pp"][prompt]["median"], gpu["tg"][prompt]["median"]
        cp = cpu["f0"]["pp"][prompt]["median"]
        for gen in OUTPUTS:
            unc = uncoord.get((prompt, gen))
            h = unc["handoff_s"] if unc else 0.0
            spec = {
                m: spec_throughput(
                    prompt,
                    gen,
                    gp * rt[m]["prefill"],
                    cpu[m]["tg"][prompt]["median"] * rt[m]["decode"],
                    h,
                )
                for m in PIXEL_MASKS
            }
            dp = {
                m: dp_throughput(
                    prompt,
                    gen,
                    [
                        (gp, gd * rt[m]["both_gpu"]),
                        (cp, cpu[m]["tg"][prompt]["median"] * rt[m]["both_cpu"]),
                    ],
                )
                for m in PIXEL_MASKS
            }
            singles = {"GPU": dp_throughput(prompt, gen, [(gp, gd)])}
            singles.update(
                {
                    f"CPU {m}": dp_throughput(prompt, gen, [(cp, cpu[m]["tg"][prompt]["median"])])
                    for m in PIXEL_MASKS
                }
            )
            sm, dm = max(spec, key=spec.get), max(dp, key=dp.get)
            su = max(singles, key=singles.get)
            cands = [("coordinated data-parallel", dp[dm]), (f"single PU ({su})", singles[su])]
            if unc:
                cands += [
                    ("uncoordinated data-parallel (llm-phase-affinity)", unc["dp_corun"]),
                    (f"single PU ({unc['single_pu'].upper()}, llm-phase-affinity)", unc["single"]),
                ]
            base = max(cands, key=lambda x: x[1])
            rows.append(
                {
                    "prompt": prompt,
                    "gen": gen,
                    "handoff_s": h,
                    "spec": spec[sm],
                    "spec_setting": f"GPU prefill -> CPU decode on {sm}",
                    "spec_by_mask": spec,
                    "dp": dp[dm],
                    "dp_setting": f"CPU decode on {dm}",
                    "dp_by_mask": dp,
                    "single": singles[su],
                    "single_pu": su,
                    "baseline": base[1],
                    "baseline_from": base[0],
                    "gain": spec[sm] / base[1],
                    "gain_vs_coordinated_dp": spec[sm] / dp[dm],
                    "uncoord": unc,
                    "sources": sorted(
                        set(
                            lpa.files_of(
                                gpu["pp"][prompt],
                                gpu["tg"][prompt],
                                cpu["f0"]["pp"][prompt],
                                *(cpu[m]["tg"][prompt] for m in PIXEL_MASKS),
                            )
                            + [f for m in PIXEL_MASKS for f in rt[m]["files"]]
                        )
                    ),
                }
            )
    return rows, None, {"ratios": rt, "alone": alone, "corun": corun}


def pixel_tables(rows, extra):
    rt = extra["ratios"]
    out = [
        "| CPU decode mask | GPU prefill next to CPU decode | CPU decode next to GPU prefill | both decoding: CPU, GPU |",
        "|---|---|---|---|",
    ]
    for m in PIXEL_MASKS:
        r = rt[m]
        out.append(
            f"| {m} | {r['prefill']:.3f} | {r['decode']:.3f} | {r['both_cpu']:.3f}, {r['both_gpu']:.3f} |"
        )
    out += [
        "",
        "| P | D | specialized by CPU mask (f0, c0, 30) | data-parallel by mask (f0, c0, 30) | best single PU "
        "| uncoordinated: specialized, data-parallel, gain | baseline (from) | **gain** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        u = r["uncoord"]
        unc = (
            f"{max(u['specialized'].values()):.4f}, {u['dp_corun']:.4f}, {u['gain']:.2f}"
            if u
            else "—"
        )
        sp = ", ".join(f"{r['spec_by_mask'][m]:.4f}" for m in PIXEL_MASKS)
        dp = ", ".join(f"{r['dp_by_mask'][m]:.4f}" for m in PIXEL_MASKS)
        out.append(
            f"| {r['prompt']} | {r['gen']} | {sp} | {dp} | {r['single']:.4f} ({r['single_pu']}) | {unc} | "
            f"{r['baseline']:.4f} ({r['baseline_from']}) | **{r['gain']:.2f}** |"
        )
    return "\n".join(out)


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--json", help="write every grid row and setting here")
    args = ap.parse_args()
    DATA = args.data
    alone, corun = load()
    lpa_alone = lpa.load_alone(SERIAL, "llamacpp")
    print(f"## Samsung S24+ ({SERIAL})\n")
    cdir = os.path.join(DATA, SERIAL, "calib")
    if os.path.isdir(cdir):
        print(f"### Calibration\n\n{calib_table(load_alone(cdir), alone, lpa_alone)}\n")
    print(
        f"### Alone rates, lpa_pace (median [min-max] of 3 runs)\n\n{alone_table(alone, lpa_alone)}\n"
    )
    if corun:
        print(
            f"### Co-runs (wall rates, tok/s; median [min-max] of 3 trials)\n\n{corun_table(corun, alone)}\n"
        )
    dump, best = {}, None
    for model in sorted({e["cfg"]["model"] for e in alone.values()}):
        sets = settings(corun, Refs(alone, model))
        for fam in sets:
            if sets[fam]:
                print(f"### {model}: trade-off curve, {fam}\n\n{tradeoff_table(sets, fam)}\n")
        ranked = rank_settings(alone, corun, model)
        if ranked:
            print(f"### {model}: each specialized setting alone, best gain over its mixes\n")
            print("| setting | family | best gain | at P, D |\n|---|---|---|---|")
            for x in ranked:
                print(
                    f"| {x['key']} | {x['family']} | {x['gain']:.3f} | {x['mix'][0]}, {x['mix'][1]} |"
                )
            print()
        rows, why = grid(alone, corun, model)
        if rows is None:
            print(f"### {model}: no prediction grid ({why})\n")
            continue
        print(f"### {model}: predicted request throughput (requests/s)\n\n{grid_table(rows)}\n")
        for r in rows:
            if r["gain"] is not None:
                cand = (
                    r["gain"],
                    model,
                    r["prompt"],
                    r["gen"],
                    r["spec_setting"],
                    r["baseline_from"],
                )
                best = max(best, cand) if best else cand
        dump[model] = {"rows": rows, "settings": sets}
    prows, why, extra = pixel_grid()
    print(f"## Pixel 7a ({PIXEL}), ExecuTorch 1B, CPU decode width (design D5)\n")
    if prows is None:
        print(f"No grid ({why}).\n")
    else:
        print(f"{pixel_tables(prows, extra)}\n")
        dump["pixel_et_1b"] = {"rows": prows, "ratios": extra["ratios"]}
    if best:
        gain, model, prompt, gen, setting, base = best
        print(
            f"\n## Gate\n\n**{'GO' if gain >= GATE else 'NO-GO'}**: best gain {gain:.2f} (gate {GATE}) with {model}, "
            f"P={prompt}, D={gen}, setting {setting}, against {base}."
        )
    if prows:
        pb = max(prows, key=lambda r: r["gain"])
        print(
            f"Pixel ExecuTorch 1B: best gain {pb['gain']:.2f} at P={pb['prompt']}, D={pb['gen']} "
            f"({pb['spec_setting']}), against {pb['baseline_from']}."
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(lpa.jsonable(dump), fh, indent=1, default=str)


if __name__ == "__main__":
    main()
