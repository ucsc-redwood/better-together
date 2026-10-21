#!/usr/bin/env python3
"""LLM phase-affinity study: tables, co-run ratios and the prediction grid (design D5) from the
raw phone outputs under data/llm-phase-affinity/<serial>/
(openspec/changes/llm-phase-affinity, decision record docs/reports-for-human/2026-10-21-llm-phase-affinity.md).

Raw layout (written by data/llm-phase-affinity/drivers/):
  alone-llamacpp/<model>_<cfg>_{pp,tg}_r<k>[_retry].{out,err,clk,meta}   llama-bench -o jsonl
  alone-et/<model>_<cfg>_p<P>_r<k>[_retry].{out,err,clk,meta}            llama_main stdout
  corun-llamacpp/<pair>_p<P>_t<k>/{A,B}.{out,err} + corun.{meta,clk}     two pinned processes
  corun-et/<pair>_p<P>_t<k>/...                                          same, llama_main loops
  bandwidth/<case>_t<k>/...                                              cpu_readbw / roofline

Every value is the median of its runs with [min-max]; a run whose thermal status ended >= 1 is
replaced by its _retry. A co-run rate keeps only the repetitions that lie wholly inside the
interval where both processes were running repetitions.

  uv run optimizer/analysis/llm_phase_affinity.py [--data data/llm-phase-affinity] [--json out.json]
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "llm-phase-affinity")
PHONES = {"R5CY21Y3VEV": "Samsung S24+", "3A021JEHN02756": "Pixel 7a"}
PROMPTS = (128, 512, 2048)
OUTPUTS = (32, 128, 512)
GATE = 1.2
# KV cache shape per model (Llama 3.2 1B/3B params.json).
KV = {
    "1b": {"layers": 16, "kv_heads": 8, "head_dim": 64},
    "3b": {"layers": 28, "kv_heads": 8, "head_dim": 128},
}
# Bytes read + written per KV element in a hand-off: llama.cpp keeps fp16 on both PUs; ExecuTorch's
# Vulkan export keeps fp16 (--vulkan-force-fp16) and its XNNPACK export fp32 (-d fp32).
KV_RW_BYTES = {"llamacpp": 2 + 2, "et": 2 + 4}


def stat(values, files=()):
    """Median, min, max and count of a list of numbers, with the raw files it came from."""
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "files": sorted(files),
    }


def fmt_stat(s, digits=1):
    if s is None:
        return "—"
    return f"{s['median']:.{digits}f} [{s['min']:.{digits}f}-{s['max']:.{digits}f}]"


def run_files(directory, stem_re):
    """Map run key (without the _r<k>[_retry] suffix) -> list of .out paths, one per round, the
    _retry of a round replacing the throttled original."""
    rounds = {}
    for path in glob.glob(os.path.join(directory, "*.out")):
        name = os.path.basename(path)[: -len(".out")]
        m = re.fullmatch(f"(?P<key>{stem_re})" + r"_r(?P<k>\d+)(?P<retry>_retry)?", name)
        if not m:
            continue
        key, k, retry = m["key"], int(m["k"]), bool(m["retry"])
        slot = rounds.setdefault(key, {})
        if retry or k not in slot:
            slot[k] = path
    return {key: [slot[k] for k in sorted(slot)] for key, slot in rounds.items()}


def gpu_clock_mhz(clk_path):
    """Median of the positive GPU clock samples (sysfs reads kHz on both phones) during a run."""
    vals = []
    if os.path.exists(clk_path):
        for line in open(clk_path):
            m = re.search(r"gpu=(\d+)", line)
            if m and int(m.group(1)) > 0:
                vals.append(int(m.group(1)) / 1000)
    return statistics.median(vals) if vals else None


def meta_field(meta_path, field):
    if not os.path.exists(meta_path):
        return None
    for line in open(meta_path):
        if line.startswith(field + "="):
            return line.strip().split("=", 1)[1]
    return None


def thermal_after(meta_path):
    """Thermal status the phone reported when the run ended (None if unknown)."""
    if not os.path.exists(meta_path):
        return None
    after = False
    for line in open(meta_path):
        if line.startswith("thermal_after"):
            after = True
        m = re.match(r"Thermal Status: (\d+)", line)
        if after and m:
            return int(m.group(1))
    return None


def read_bench_jsonl(path):
    """llama-bench -o jsonl records of one run."""
    recs = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            recs.append(json.loads(line))
    return recs


def load_llamacpp_alone(directory):
    """{(model, cfg): {"pp": {P: stat}, "tg": {depth: stat}, "mask", "clock_mhz", "failed"}}."""
    out = {}
    for key, paths in run_files(directory, r"\w+?_\w+_(?:pp|tg)").items():
        model, cfg, kind = re.fullmatch(r"(\w+?)_(\w+)_(pp|tg)", key).groups()
        entry = out.setdefault((model, cfg), {"pp": {}, "tg": {}, "clock_mhz": [], "failed": []})
        series = {}
        for path in paths:
            base = path[: -len(".out")]
            entry["mask"] = meta_field(base + ".meta", "mask")
            recs = read_bench_jsonl(path)
            if not recs:
                rc = meta_field(base + ".meta", "exit_code")
                err = (
                    open(base + ".err").read().strip().splitlines()[-3:]
                    if os.path.exists(base + ".err")
                    else []
                )
                entry["failed"].append(
                    {"file": os.path.relpath(path, ROOT), "exit_code": rc, "error": err}
                )
                continue
            clk = gpu_clock_mhz(base + ".clk")
            if clk:
                entry["clock_mhz"].append(clk)
            for r in recs:
                entry["model_size"] = r["model_size"]
                x = r["n_prompt"] if kind == "pp" else r["n_depth"]
                series.setdefault(x, ([], []))
                series[x][0].append(r["avg_ts"])
                series[x][1].append(os.path.relpath(path, ROOT))
        for x, (vals, files) in series.items():
            entry[kind][x] = stat(vals, files)
    for entry in out.values():
        entry["clock_mhz"] = stat(entry["clock_mhz"])
    return out


def et_observer(path):
    """PyTorchObserver records printed by llama_main (one per generate(); with --warmup the last one
    is the measured run). Each gets prefill/decode tok/s as the runner computes them and its epoch
    intervals: prefill [inference_start, prompt_eval_end], decode [prompt_eval_end, inference_end]."""
    recs = []
    for line in open(path, errors="replace"):
        i = line.find("PyTorchObserver ")
        if i < 0:
            continue
        r = json.loads(line[i + len("PyTorchObserver ") :])
        pre_s = (r["prompt_eval_end_ms"] - r["inference_start_ms"]) / 1000
        dec_s = (r["inference_end_ms"] - r["prompt_eval_end_ms"]) / 1000
        r["prefill_tps"] = r["prompt_tokens"] / pre_s if pre_s > 0 else None
        r["decode_tps"] = (
            r["generated_tokens"] / dec_s if dec_s > 0 and r["generated_tokens"] else None
        )
        r["prefill_interval"] = (r["inference_start_ms"] / 1000, r["prompt_eval_end_ms"] / 1000)
        # With --warmup the warm-up generate() runs between load and inference_start (empty otherwise).
        r["warmup_interval"] = (r["model_load_end_ms"] / 1000, r["inference_start_ms"] / 1000)
        r["decode_interval"] = (r["prompt_eval_end_ms"] / 1000, r["inference_end_ms"] / 1000)
        recs.append(r)
    return recs


def load_et_alone(directory):
    """{(model, cfg): {"pp": {P: stat}, "tg": {P: stat}, "mask", "clock_mhz", "failed"}}; the decode
    rate at prompt P is the decode that follows that prompt (KV depth P)."""
    out = {}
    for key, paths in run_files(directory, r"\w+?_\w+_p\d+").items():
        model, cfg, prompt = re.fullmatch(r"(\w+?)_(\w+)_p(\d+)", key).groups()
        prompt = int(prompt)
        entry = out.setdefault(
            (model, cfg), {"pp": {}, "tg": {}, "clock_mhz": [], "failed": [], "prompt_tokens": {}}
        )
        pre, dec, files = [], [], []
        for path in paths:
            base = path[: -len(".out")]
            entry["mask"] = meta_field(base + ".meta", "mask")
            recs = et_observer(path)
            if not recs:
                err = open(base + ".err", errors="replace").read().strip().splitlines()[-3:]
                rc = meta_field(base + ".meta", "exit_code")
                entry["failed"].append(
                    {"file": os.path.relpath(path, ROOT), "exit_code": rc, "error": err}
                )
                continue
            r = recs[-1]
            entry["prompt_tokens"][prompt] = r["prompt_tokens"]
            pre.append(r["prefill_tps"])
            dec.append(r["decode_tps"])
            files.append(os.path.relpath(path, ROOT))
            clk = gpu_clock_mhz(base + ".clk")
            if clk:
                entry["clock_mhz"].append(clk)
        entry["pp"][prompt] = stat([v for v in pre if v], files)
        entry["tg"][prompt] = stat([v for v in dec if v], files)
    for entry in out.values():
        entry["clock_mhz"] = stat(entry["clock_mhz"])
    return out


def merge(intervals, gap=0.05):
    """Union of (start, end) intervals; gaps shorter than `gap` seconds are closed."""
    out = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def covered(interval, partner):
    """True if (start, end) lies wholly inside one interval of the merged partner list."""
    s, e = interval
    return any(ps <= s and e <= pe for ps, pe in partner)


def bench_reps(out_path, err_path):
    """Repetitions of a llama-bench --progress run: [(start, end, tokens)] from the epoch-stamped
    stderr lines '... prompt run i/R' / '... generation run i/R' and samples_ns of the record."""
    recs = read_bench_jsonl(out_path) if os.path.exists(out_path) else []
    if len(recs) != 1:
        return []
    rec = recs[0]
    tokens = rec["n_prompt"] if rec["n_prompt"] > 0 else rec["n_gen"]
    starts = {}
    for line in open(err_path):
        m = re.match(
            r"(\d+\.\d+) llama-bench: benchmark \d+/\d+: (?:prompt|generation) run (\d+)/\d+", line
        )
        if m:
            starts[int(m.group(2))] = float(m.group(1))
    return [
        (starts[i + 1], starts[i + 1] + ns / 1e9, tokens)
        for i, ns in enumerate(rec["samples_ns"])
        if i + 1 in starts
    ]


def overlapped_rate(mine, partner):
    """Tokens/s over my repetitions that the partner's repetitions wholly cover; (rate, kept, total)."""
    cover = merge([(s, e) for s, e, _ in partner])
    kept = [(s, e, t) for s, e, t in mine if covered((s, e), cover)]
    if not kept:
        return None, 0, len(mine)
    return sum(t for _, _, t in kept) / sum(e - s for s, e, _ in kept), len(kept), len(mine)


def trial_dirs(directory):
    """Map trial key (without _t<k>[_retry]) -> trial directories, the _retry replacing its original."""
    slots = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_t*"))):
        m = re.fullmatch(r"(?P<key>.+)_t(?P<k>\d+)(?P<retry>_retry)?", os.path.basename(path))
        if not m or not os.path.isdir(path):
            continue
        slot = slots.setdefault(m["key"], {})
        if m["retry"] or int(m["k"]) not in slot:
            slot[int(m["k"])] = path
    return {key: [slot[k] for k in sorted(slot)] for key, slot in slots.items()}


def load_llamacpp_corun(directory):
    """{pair_key: {"A": stat, "B": stat, "kept": [(kept, total) per trial for A and B], "clock_mhz"}}.
    Per trial each process's rate is tokens / seconds over its repetitions that the other's
    repetitions wholly cover; the stat is over trials."""
    out = {}
    for key, dirs in trial_dirs(directory).items():
        rates = {"A": ([], []), "B": ([], [])}
        kept, clocks = [], []
        for d in dirs:
            reps = {
                x: bench_reps(os.path.join(d, f"{x}.out"), os.path.join(d, f"{x}.err"))
                for x in "AB"
            }
            ka = overlapped_rate(reps["A"], reps["B"])
            kb = overlapped_rate(reps["B"], reps["A"])
            kept.append({"trial": os.path.relpath(d, ROOT), "A": ka[1:], "B": kb[1:]})
            for x, (rate, n, _) in (("A", ka), ("B", kb)):
                if rate is not None and n > 0:
                    rates[x][0].append(rate)
                    rates[x][1].append(os.path.relpath(d, ROOT))
            clk = gpu_clock_mhz(os.path.join(d, "corun.clk"))
            if clk:
                clocks.append(clk)
        out[key] = {
            "A": stat(*rates["A"]),
            "B": stat(*rates["B"]),
            "kept": kept,
            "clock_mhz": stat(clocks),
        }
    return out


def et_phase_intervals(path, phase, busy=False):
    """[(start, end, tokens)] of one phase ("prefill" or "decode") over every llama_main run in a
    co-run log. busy=True adds each run's warm-up generate() (tokens 0): the process was computing then,
    which matters when it is the partner whose activity a decode window's duty counts."""
    out = []
    for r in et_observer(path) if os.path.exists(path) else []:
        s, e = r[f"{phase}_interval"]
        tokens = r["prompt_tokens"] if phase == "prefill" else r["generated_tokens"]
        if e > s and tokens:
            out.append((s, e, tokens))
        ws, we = r["warmup_interval"]
        if busy and we > ws:
            out.append((ws, we, 0))
    return out


def duty(interval, partner):
    """Fraction of (start, end) during which one of the partner intervals was running."""
    s, e = interval
    busy = sum(max(0.0, min(e, pe) - max(s, ps)) for ps, pe in merge(partner))
    return busy / (e - s)


def load_et_corun(directory):
    """ExecuTorch co-runs (stock llama_main in loops, no per-token timing). A key is
    <A-tag>_<B-tag>[_p<P>] with tags gpuP/cpuP (prefill loop), cpuD/gpuD (512-token decode loop) and
    cpuDs/gpuDs (8-token decode loop), or solo_<tag>[_p<P>] for the same loop alone.

    Full overlap (spec "Co-run interference is measured"): in a co-run a prefill or a decode window
    counts only if it lies wholly inside the partner's compute, its prefills (warm-up generate()
    included) or its decode windows; model loads are never compute. Per side: "rate" over the counted
    intervals (all of them in a solo loop) with "kept" = (counted, total) per trial. For the previous
    estimate, kept as a diagnostic: "rate_all" over every interval and "duty", the length-weighted
    fraction of each interval during which the partner computed."""
    out = {}
    for key, dirs in trial_dirs(directory).items():
        solo = key.startswith("solo_")
        tags = key.split("_")
        sides = {"A": tags[1]} if solo else {"A": tags[0], "B": tags[1]}
        acc = {x: {"rate": ([], []), "rate_all": ([], []), "duty": ([], [])} for x in sides}
        kept = {x: [] for x in sides}
        for d in dirs:
            rel = os.path.relpath(d, ROOT)
            for x, tag in sides.items():
                phase = "prefill" if tag[3] == "P" else "decode"
                mine = et_phase_intervals(os.path.join(d, f"{x}.out"), phase)
                if not mine:
                    continue
                span = sum(e - s for s, e, _ in mine)
                if solo:
                    rate, n, total = sum(t for _, _, t in mine) / span, len(mine), len(mine)
                    dt = [0.0] * len(mine)
                else:
                    o = "B" if x == "A" else "A"
                    o_phase = "prefill" if sides[o][3] == "P" else "decode"
                    busy = et_phase_intervals(os.path.join(d, f"{o}.out"), o_phase, busy=True)
                    rate, n, total = overlapped_rate(mine, busy)
                    dt = [duty(m[:2], [(s, e) for s, e, _ in busy]) for m in mine]
                kept[x].append((n, total))
                if rate is not None:
                    acc[x]["rate"][0].append(rate)
                    acc[x]["rate"][1].append(rel)
                acc[x]["rate_all"][0].append(sum(t for _, _, t in mine) / span)
                acc[x]["rate_all"][1].append(rel)
                acc[x]["duty"][0].append(sum(u * (e - s) for u, (s, e, _) in zip(dt, mine)) / span)
                acc[x]["duty"][1].append(rel)
        out[key] = {
            x: {**{k: stat(*v) for k, v in a.items()}, "kept": kept[x]} for x, a in acc.items()
        }
    return out


def load_bandwidth(directory):
    """{case: {"cpu": stat, "gpu": stat}} in GB/s, case in cpu / gpu / both. CPU intervals are 0.5 s
    windows ending at the printed epoch; GPU samples end at the epoch their line was read and last
    the GPU-timed seconds of n x width x 4 x loops bytes. In "both", only samples wholly covered by
    the other side's samples count."""
    out = {}
    for key, dirs in trial_dirs(directory).items():
        per = {"cpu": ([], []), "gpu": ([], [])}
        for d in dirs:
            cpu, gpu = [], []
            cpu_path, gpu_path = os.path.join(d, "cpu.out"), os.path.join(d, "gpu.out")
            if os.path.exists(cpu_path):
                for line in open(cpu_path):
                    m = re.fullmatch(r"(\d+\.\d+) (\d+\.\d+)\s*", line)
                    if m:
                        t, g = float(m.group(1)), float(m.group(2))
                        cpu.append((t - 0.5, t, g))
            if os.path.exists(gpu_path):
                cfg = json.load(open(os.path.join(d, "gpu.config.json")))
                nbytes = cfg["n"] * cfg["width"] * 4 * cfg["loops"]
                for line in open(gpu_path):
                    t, _, rest = line.partition(" ")
                    if '"event":"sample"' in rest.replace(" ", ""):
                        r = json.loads(rest)
                        gpu.append((float(t) - r["seconds"], float(t), nbytes / r["seconds"] / 1e9))
            if cpu and gpu:
                cpu_cov, gpu_cov = (
                    merge([(a, b) for a, b, _ in gpu], 0.6),
                    merge([(a, b) for a, b, _ in cpu], 0.6),
                )
                cpu = [x for x in cpu if covered(x[:2], cpu_cov)]
                gpu = [x for x in gpu if covered(x[:2], gpu_cov)]
            for side, samples in (("cpu", cpu), ("gpu", gpu)):
                if samples:
                    per[side][0].append(statistics.median(g for _, _, g in samples))
                    per[side][1].append(os.path.relpath(d, ROOT))
        out[key] = {side: stat(*vals) for side, vals in per.items()}
    return out


def handoff_seconds(model, prompt, gbps, rw_bytes=KV_RW_BYTES["llamacpp"]):
    """KV-cache hand-off upper bound: every K and V element of the prompt read once and written once
    at the given bandwidth (design D5; on unified memory it may be a layout change or nothing)."""
    kv = KV[model]
    elements = 2 * kv["layers"] * kv["kv_heads"] * kv["head_dim"] * prompt
    return elements * rw_bytes / (gbps * 1e9)


def partial_overlap_period(prompt, gen, alone_rates, corun_rates):
    """Specialized period if the two phases slow each other only while both run (design D5 assumes
    they always do): the shorter phase runs wholly at its co-run rate, the longer one at its co-run
    rate for that long and at its alone rate for the rest. Rates are (prefill on A, decode on B)."""
    (p, d), (pc, dc) = alone_rates, corun_rates
    tp, td = prompt / pc, gen / dc
    if tp <= td:
        return tp + (gen - dc * tp) / d
    return td + (prompt - pc * td) / p


def predict_mix(prompt, gen, alone, both_decode, specialized, handoff_s):
    """Design D5 for one request (prompt P, output D tokens), rates in tokens/s.

    alone:        {pu: (prefill, decode)} measured alone
    both_decode:  {pu: decode} while both PUs decode (data-parallel co-run)
    specialized:  {(a, b): (prefill on a, decode on b)} while a prefills and b decodes
    handoff_s:    KV hand-off time added to every specialized period
    Returns request throughputs (requests/s) and the gain of the best specialized pipeline over
    the co-run-corrected data-parallel bound."""
    t_alone = {pu: prompt / p + gen / d for pu, (p, d) in alone.items()}
    single_pu = min(t_alone, key=t_alone.get)
    dp_alone = sum(1 / t for t in t_alone.values())
    # Co-run correction applies to decode only; prefill keeps its alone rate (conservative: it
    # makes the data-parallel bound larger).
    dp_corun = sum(1 / (prompt / alone[pu][0] + gen / both_decode[pu]) for pu in alone)
    spec = {
        pair: 1 / (max(prompt / p, gen / d) + handoff_s) for pair, (p, d) in specialized.items()
    }
    best = max(spec, key=spec.get)
    return {
        "single": 1 / t_alone[single_pu],
        "single_pu": single_pu,
        "dp_alone": dp_alone,
        "dp_corun": dp_corun,
        "specialized": spec,
        "best_pair": best,
        "gain": spec[best] / dp_corun,
    }


# The CPU and GPU processing units of each (phone, runtime): the configuration names of the alone
# runs. The CPU unit is the configuration the co-runs use. A GPU unit takes per point the faster of
# its configurations: llama.cpp flash attention auto (gpu) or off (gpufa0), as the co-runs do
# (design.md "Deviations"); ExecuTorch Vulkan the 4w or 8da4w file (design D2).
PU = {
    ("R5CY21Y3VEV", "llamacpp"): {"cpu": ["cpu3f0"], "gpu": ["gpu", "gpufa0"]},
    ("3A021JEHN02756", "llamacpp"): {"cpu": ["cpuf0"], "gpu": ["gpu", "gpufa0"]},
    ("R5CY21Y3VEV", "et"): {"cpu": ["xnn3f0"], "gpu": ["vk4w", "vk8da4w"]},
    ("3A021JEHN02756", "et"): {"cpu": ["xnnf0"], "gpu": ["vk4w", "vk8da4w"]},
}
# Co-run pair keys (directory names without _p<P>_t<k>): A is the prefill process, B the decode
# process; in the both-decoding pair A is the CPU and B the GPU.
PAIRS = {("gpu", "cpu"): "gpuP_cpuD", ("cpu", "gpu"): "cpuP_gpuD"}
BOTH_DECODE = "cpuD_gpuD"
CORUN_MODEL = "1b"
# Measured single-backend as a reference only (design Non-Goals): no grid, not in the gate.
REFERENCE_ONLY = {"8b"}


def pu_rates(alone, model, cfgs):
    """{"pp": {P: stat}, "tg": {depth: stat}, "cfg": {...}} of a PU, taking per point the best of
    its configurations (one configuration for every PU except ExecuTorch Vulkan)."""
    out = {"pp": {}, "tg": {}, "cfg": {"pp": {}, "tg": {}}}
    for cfg in cfgs:
        entry = alone.get((model, cfg))
        if not entry:
            continue
        for kind in ("pp", "tg"):
            for x, s in entry[kind].items():
                if s and (x not in out[kind] or s["median"] > out[kind][x]["median"]):
                    out[kind][x] = s
                    out["cfg"][kind][x] = cfg
    return out


def ratio(corun, alone):
    return corun["median"] / alone["median"] if corun and alone else None


def files_of(*stats):
    """Raw files behind some stats (for tracing a prediction to its inputs)."""
    return sorted({f for s in stats if s for f in s["files"]})


def ratios_llamacpp(corun, pus, near):
    """Co-run / alone ratios from llama-bench co-runs: prefill against the alone pp at the co-run's
    prompt, decode (depth 0 loops) against the alone tg at depth 0."""
    both = corun[BOTH_DECODE]
    out = {
        "both": {
            "cpu": ratio(both["A"], pus["cpu"]["tg"][0]),
            "gpu": ratio(both["B"], pus["gpu"]["tg"][0]),
        },
        "spec": {},
    }
    used = [both["A"], both["B"], pus["cpu"]["tg"][0], pus["gpu"]["tg"][0]]
    for (a, b), key in PAIRS.items():
        c = corun[f"{key}_p{near}"]
        out["spec"][(a, b)] = (ratio(c["A"], pus[a]["pp"][near]), ratio(c["B"], pus[b]["tg"][0]))
        used += [c["A"], c["B"], pus[a]["pp"][near]]
    out["files"] = files_of(*used)
    return out


def et_rate(corun, key, side="A"):
    """A co-run (or solo) rate that counts for the gate: 3 trials, each with counted intervals."""
    st = corun[key][side]["rate"]
    if st is None or st["n"] < 3:
        raise KeyError(f"{key} {side}: fewer than 3 trials with fully overlapped intervals")
    return st


def ratios_et(corun, near):
    """Co-run / solo ratios from ExecuTorch loops, every co-run interval fully overlapped by the
    partner's compute. Prefill: the prefills of <a>P_<b>D_p<P> inside a partner decode window, against
    solo_<a>P_p<P>. Decode: the 8-token windows of <a>P_<b>Ds_p<P> inside a partner prefill, against
    solo_<b>Ds. Both decoding: the 8-token windows of cpuDs_gpuD / gpuDs_cpuD inside the other PU's
    decode windows, against solo_cpuDs / solo_gpuDs."""
    used = []

    def r(key, side, solo):
        a, b = et_rate(corun, key, side), et_rate(corun, solo)
        used.extend((a, b))
        return ratio(a, b)

    both = {"cpu": r("cpuDs_gpuD", "A", "solo_cpuDs"), "gpu": r("gpuDs_cpuD", "A", "solo_gpuDs")}
    spec = {}
    for (a, b), key in PAIRS.items():
        spec[(a, b)] = (
            r(f"{key}_p{near}", "A", f"solo_{a}P_p{near}"),
            r(f"{a}P_{b}Ds_p{near}", "B", f"solo_{b}Ds"),
        )
    return {"both": both, "spec": spec, "files": files_of(*used)}


def ratios_et_extrapolated(corun, near):
    """The previous estimate, a diagnostic only (it does not meet the spec's full overlap): the decode
    side of <a>P_<b>D_p<P> over all its 512-token windows, whose partner computed only a fraction u of
    the time, extrapolated as 1 - (1 - r) / u; both decoding over all windows of cpuD_gpuD, as
    measured. The prefill ratio is the full-overlap one."""

    def r(key, side, solo, field):
        return ratio(corun[key][side][field], corun[solo]["A"][field])

    both = {
        "cpu": r(BOTH_DECODE, "A", "solo_cpuD", "rate_all"),
        "gpu": r(BOTH_DECODE, "B", "solo_gpuD", "rate_all"),
    }
    spec = {}
    for (a, b), key in PAIRS.items():
        k = f"{key}_p{near}"
        rd = r(k, "B", f"solo_{b}D", "rate_all")
        u = corun[k]["B"]["duty"]["median"]
        spec[(a, b)] = (
            r(k, "A", f"solo_{a}P_p{near}", "rate"),
            max(0.05, 1 - (1 - rd) / u) if u > 0 else None,
        )
    return {"both": both, "spec": spec}


def grid(serial, runtime, model, alone, corun_by_model, bw):
    """Prediction rows for every mix, or (None, reason) when an input is missing. Decode rates at
    prompt P are the alone decode at KV depth P scaled by the co-run ratios; a prompt of 128 uses the
    ratios measured at 512."""
    pus = {pu: pu_rates(alone, model, cfgs) for pu, cfgs in PU[(serial, runtime)].items()}
    # A model without co-runs of its own borrows the ratios of CORUN_MODEL (co-run / alone of that
    # model) applied to its own alone rates: a transfer, flagged in the rows and not gate-eligible.
    corun_model = model if model in corun_by_model else CORUN_MODEL
    corun = corun_by_model.get(corun_model, {})
    pus_corun = {
        pu: pu_rates(alone, corun_model, cfgs) for pu, cfgs in PU[(serial, runtime)].items()
    }
    gbps = min(bw[c]["median"] for c in ("cpu", "gpu") if bw.get(c)) if bw else None
    rows = []
    for prompt in PROMPTS:
        near = prompt if prompt in (512, 2048) else 512
        try:
            al = {
                pu: (r["pp"][prompt]["median"], r["tg"][prompt]["median"]) for pu, r in pus.items()
            }
            rt = (
                ratios_llamacpp(corun, pus_corun, near)
                if runtime == "llamacpp"
                else ratios_et(corun, near)
            )
            bd = {pu: al[pu][1] * rt["both"][pu] for pu in al}
            spec = {
                (a, b): (al[a][0] * rp, al[b][1] * rd) for (a, b), (rp, rd) in rt["spec"].items()
            }
        except (KeyError, TypeError) as e:
            return None, f"missing input {e}"
        h = handoff_seconds(model, prompt, gbps, KV_RW_BYTES[runtime]) if gbps else 0.0
        prev = None
        if runtime == "et":
            try:
                old = ratios_et_extrapolated(corun, near)
                prev = (
                    {pu: al[pu][1] * old["both"][pu] for pu in al},
                    {
                        (a, b): (al[a][0] * rp, al[b][1] * rd)
                        for (a, b), (rp, rd) in old["spec"].items()
                    },
                )
            except (KeyError, TypeError):
                prev = None
        part = pu_rates(alone, model, ["partial", "partialnh"])
        free = {(a, b): (al[a][0], al[b][1]) for a, b in spec}
        for gen in OUTPUTS:
            r = predict_mix(prompt, gen, al, bd, spec, h)
            # Diagnostics for the limiting effect, not the gate: the gain if the specialized pair ran
            # at its alone rates, and if the KV hand-off were free.
            r["gain_no_interference"] = predict_mix(prompt, gen, al, bd, free, h)["gain"]
            r["gain_no_handoff"] = predict_mix(prompt, gen, al, bd, spec, 0.0)["gain"]
            # ExecuTorch only: the gain with the previous, duty-extrapolated co-run estimate.
            r["gain_extrapolated"] = (
                predict_mix(prompt, gen, al, *prev, h)["gain"] if prev else None
            )
            # Against the better of data-parallel and the best single PU: on the Samsung the co-run
            # data-parallel bound can fall below one PU alone (a GPU decode starves the CPU decode).
            r["gain_partial_overlap"] = max(
                1 / (partial_overlap_period(prompt, gen, free[pair], spec[pair]) + h)
                for pair in spec
            ) / max(r["single"], r["dp_corun"])
            # The decode co-run ratio the best pair would need for the gate, everything else as is
            # (None: its prefill alone already exceeds the period the gate allows).
            # Also not the gate: the data-parallel bound with each PU's prefill at its co-run rate next
            # to the other's decode as well (design D5 keeps prefill at its alone rate, conservatively).
            dp_full = sum(
                1 / (prompt / spec[(pu, other)][0] + gen / bd[pu])
                for pu, other in (("gpu", "cpu"), ("cpu", "gpu"))
            )
            r["gain_vs_dp_prefill_corun"] = max(r["specialized"].values()) / dp_full
            a, b = r["best_pair"]
            budget = 1 / (GATE * r["dp_corun"]) - h
            r["decode_ratio"] = rt["spec"][(a, b)][1]
            r["decode_ratio_needed"] = (
                gen / (al[b][1] * budget) if budget >= prompt / spec[(a, b)][0] else None
            )
            r.update(
                prompt=prompt,
                gen=gen,
                handoff_s=h,
                alone=al,
                both_decode=bd,
                spec_rates=spec,
                ratios=rt,
                sources=files_of(
                    *(pus[pu][kind][prompt] for pu in pus for kind in ("pp", "tg")),
                    *(bw or {}).values(),
                )
                + rt["files"],
            )
            r["ratios_from"] = corun_model
            r["gate_eligible"] = corun_model == model
            if part["pp"].get(prompt) and part["tg"].get(prompt):
                r["layer_split"] = 1 / (
                    prompt / part["pp"][prompt]["median"] + gen / part["tg"][prompt]["median"]
                )
            rows.append(r)
    return rows, None


def load_alone(serial, runtime):
    """Alone runs of a phone and runtime, the 3-round additions (alone-*-extra) and the 8B reference
    runs (alone-*-8b) merged in."""
    sub, loader = (
        ("alone-llamacpp", load_llamacpp_alone)
        if runtime == "llamacpp"
        else ("alone-et", load_et_alone)
    )
    out = {}
    for d in (sub, sub + "-extra", sub + "-8b"):
        path = os.path.join(DATA, serial, d)
        if os.path.isdir(path):
            out.update(loader(path))
    return out


def alone_table(alone):
    lines = [
        "| model | config | mask | pp128 | pp512 | pp2048 | tg@0 | tg@128 | tg@512 | tg@2048 | GPU MHz |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, cfg), e in sorted(alone.items()):
        if e["failed"] and not e["pp"] and not e["tg"]:
            err = "; ".join(" ".join(f["error"])[-160:] for f in e["failed"][:1])
            lines.append(
                f"| {model} | {cfg} | {e.get('mask')} | FAILED ({len(e['failed'])} runs): {err} |||||||||"
            )
            continue
        pp = [fmt_stat(e["pp"].get(p)) for p in PROMPTS]
        tg = [fmt_stat(e["tg"].get(d)) for d in (0,) + PROMPTS]
        gpu_used = not cfg.startswith(("cpu", "xnn"))
        clk = f"{e['clock_mhz']['median']:.0f}" if e["clock_mhz"] and gpu_used else "—"
        lines.append(
            f"| {model} | {cfg} | {e.get('mask')} | " + " | ".join(pp + tg) + f" | {clk} |"
        )
    return "\n".join(lines)


def corun_table(serial, alone, corun, model="1b", runtime="llamacpp"):
    pus = {pu: pu_rates(alone, model, cfgs) for pu, cfgs in PU[(serial, runtime)].items()}
    lines = [
        "| pair | P | A: phase on PU | A co-run | A alone | A ratio | B: phase on PU | B co-run | B alone | B ratio | kept reps A / B per trial |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(corun):
        c = corun[key]
        m = re.fullmatch(r"(\w+?)_(\w+?)(?:_p(\d+))?", key)
        a_tag, b_tag, prompt = m.group(1), m.group(2), int(m.group(3) or 0)

        def ref(tag):
            pu, phase = tag[:3], tag[3]
            return (
                f"{'prefill' if phase == 'P' else 'decode'} on {pu.upper()}",
                pus[pu]["pp"].get(prompt) if phase == "P" else pus[pu]["tg"].get(0),
            )

        (a_name, a_ref), (b_name, b_ref) = ref(a_tag), ref(b_tag)
        ra, rb = ratio(c["A"], a_ref), ratio(c["B"], b_ref)
        kept = ", ".join(f"{k['A'][0]}/{k['A'][1]} · {k['B'][0]}/{k['B'][1]}" for k in c["kept"])
        lines.append(
            f"| {key} | {prompt or '—'} | {a_name} | {fmt_stat(c['A'])} | {fmt_stat(a_ref)} | "
            f"{ra:.2f} | {b_name} | {fmt_stat(c['B'])} | {fmt_stat(b_ref)} | {rb:.2f} | {kept} |"
            if ra and rb
            else f"| {key} | {prompt or '—'} | {a_name} | {fmt_stat(c['A'])} | — | — | {b_name} | {fmt_stat(c['B'])} | — | — | {kept} |"
        )
    return "\n".join(lines)


def grid_table(rows):
    lines = [
        "| P | D | best single PU | layer split | data-parallel, alone | data-parallel, co-run "
        "| GPU prefill → CPU decode | CPU prefill → GPU decode | gain | gain, no interference "
        "| gain, no hand-off | gain, overlap only while both run (vs the better of data-parallel and one PU) "
        "| gain vs data-parallel with co-run prefill | gain, previous duty-extrapolated estimate "
        "| best pair: decode ratio used / needed for the gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sp = r["specialized"]
        split = f"{r['layer_split']:.4f}" if "layer_split" in r else "—"
        lines.append(
            f"| {r['prompt']} | {r['gen']} | {r['single']:.4f} ({r['single_pu'].upper()}) | {split} | "
            f"{r['dp_alone']:.4f} | {r['dp_corun']:.4f} | {sp[('gpu', 'cpu')]:.4f} | {sp[('cpu', 'gpu')]:.4f} | "
            f"**{r['gain']:.2f}** | {r['gain_no_interference']:.2f} | {r['gain_no_handoff']:.2f} | "
            f"{r['gain_partial_overlap']:.2f} | "
            f"{r['gain_vs_dp_prefill_corun']:.2f} | "
            + (f"{r['gain_extrapolated']:.2f}" if r["gain_extrapolated"] else "—")
            + " | "
            f"{r['decode_ratio']:.2f} / "
            + (f"{r['decode_ratio_needed']:.2f}" if r["decode_ratio_needed"] else "unreachable")
            + " |"
        )
    return "\n".join(lines)


def affinity_table(serial, runtime, alone):
    """Per model and prompt length, each phase on the CPU and GPU units (decode at KV depth P) and
    the GPU / CPU ratio: the phase affinity the pipeline would exploit."""
    lines = [
        "| model | P | prefill CPU | prefill GPU | GPU / CPU | decode CPU | decode GPU | GPU / CPU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in sorted({m for m, _ in alone} - REFERENCE_ONLY):
        pus = {pu: pu_rates(alone, model, cfgs) for pu, cfgs in PU[(serial, runtime)].items()}
        for prompt in PROMPTS:
            cells = []
            for kind in ("pp", "tg"):
                c, g = pus["cpu"][kind].get(prompt), pus["gpu"][kind].get(prompt)
                cells += [fmt_stat(c), fmt_stat(g), f"{ratio(g, c):.2f}" if c and g else "—"]
            lines.append(f"| {model} | {prompt} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kept_str(side):
    return ", ".join(f"{n}/{t}" for n, t in side["kept"])


def et_ratio_table(corun):
    """ExecuTorch co-run ratios against the solo loops, every interval fully overlapped (what the grid
    uses), with the counted / total intervals per trial; and the previous estimate (diagnostic)."""
    lines = [
        "| pair | P | prefill ratio | prefills counted per trial | decode ratio | decode windows counted per trial "
        "| previous estimate: decode ratio (duty) |",
        "|---|---|---|---|---|---|---|",
    ]

    def prev(near):
        try:
            return ratios_et_extrapolated(corun, near)
        except (KeyError, TypeError):
            return None

    for near in (512, 2048):
        rt, old = ratios_et(corun, near), prev(near)
        for (a, b), key in PAIRS.items():
            k, kd = f"{key}_p{near}", f"{a}P_{b}Ds_p{near}"
            o = (
                f"{old['spec'][(a, b)][1]:.2f} ({corun[k]['B']['duty']['median']:.2f})"
                if old
                else "—"
            )
            lines.append(
                f"| {a.upper()} prefill → {b.upper()} decode | {near} | {rt['spec'][(a, b)][0]:.2f} | "
                f"{kept_str(corun[k]['A'])} | {rt['spec'][(a, b)][1]:.2f} | {kept_str(corun[kd]['B'])} | {o} |"
            )
    rt, old = ratios_et(corun, 512), prev(512)
    o = f"{old['both']['cpu']:.2f}, {old['both']['gpu']:.2f} (as measured)" if old else "—"
    lines.append(
        f"| both decoding (CPU, GPU) | — | — | — | {rt['both']['cpu']:.2f}, {rt['both']['gpu']:.2f} | "
        f"{kept_str(corun['cpuDs_gpuD']['A'])}; {kept_str(corun['gpuDs_cpuD']['A'])} | {o} |"
    )
    return "\n".join(lines)


def et_corun_table(corun):
    lines = [
        "| loop | A rate, counted | A counted / total per trial | B rate, counted | B counted / total per trial "
        "| A, B rate over all intervals (duty) |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(corun):
        c = corun[key]
        cells, prev = [], []
        for x in ("A", "B"):
            if x in c:
                cells += [fmt_stat(c[x]["rate"]), kept_str(c[x])]
                prev.append(f"{fmt_stat(c[x]['rate_all'])} ({fmt_stat(c[x]['duty'], 2)})")
            else:
                cells += ["—", "—"]
        lines.append(f"| {key} | " + " | ".join(cells) + " | " + "; ".join(prev) + " |")
    return "\n".join(lines)


def bandwidth_table(serial, alone, bw, corun):
    """Decode at depth 0 against the bytes-per-token bound of the measured read bandwidth: every
    weight is read once per token (Llama 3.2 ties the output matrix to the embedding, so llama-bench's
    model_size is the bytes per token; the KV cache at depth 0 adds nothing). Alone rates against the
    alone bandwidth; both-decoding co-run rates against the bandwidth each PU kept while both streamed."""
    lines = [
        "| model | PU | bandwidth GB/s | bound tok/s | decode tok/s | fraction |",
        "|---|---|---|---|---|---|",
    ]
    for model in sorted({m for m, _ in alone}):
        pus = {pu: pu_rates(alone, model, cfgs) for pu, cfgs in PU[(serial, "llamacpp")].items()}
        size = next(
            e["model_size"] for (m, _), e in alone.items() if m == model and "model_size" in e
        )
        rows = [(pu, "alone", bw[pu][pu], pus[pu]["tg"].get(0)) for pu in ("cpu", "gpu")]
        if model == "1b" and BOTH_DECODE in corun and "both" in bw:
            rows += [
                ("cpu", "both decoding", bw["both"]["cpu"], corun[BOTH_DECODE]["A"]),
                ("gpu", "both decoding", bw["both"]["gpu"], corun[BOTH_DECODE]["B"]),
            ]
        for pu, case, b, d in rows:
            if not b or not d:
                continue
            bound = b["median"] * 1e9 / size
            lines.append(
                f"| {model} | {pu.upper()} {case} | {fmt_stat(b, 2)} | {bound:.1f} | {fmt_stat(d)} | "
                f"{d['median'] / bound:.2f} |"
            )
    return "\n".join(lines)


def jsonable(x):
    """Tuple keys (PU pairs) as "a->b" strings, recursively."""
    if isinstance(x, dict):
        return {
            ("->".join(k) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in x.items()
        }
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def load_coruns(serial, runtime):
    """{model: co-run dict}: corun-<runtime>/ holds CORUN_MODEL, corun-<runtime>-<model>/ any other."""
    out = {}
    loader = load_llamacpp_corun if runtime == "llamacpp" else load_et_corun
    for path in sorted(glob.glob(os.path.join(DATA, serial, f"corun-{runtime}*"))):
        suffix = os.path.basename(path)[len(f"corun-{runtime}") :]
        out[suffix.lstrip("-") or CORUN_MODEL] = loader(path)
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--json", help="write every grid row here")
    args = ap.parse_args()
    DATA = args.data
    dump, best, unassessed = {}, None, []
    for serial, phone in PHONES.items():
        bdir = os.path.join(DATA, serial, "bandwidth")
        bw = load_bandwidth(bdir) if os.path.isdir(bdir) else {}
        print(f"\n## {phone} ({serial})\n")
        if bw:
            print("### Read bandwidth (GB/s)\n\n| case | CPU | GPU |\n|---|---|---|")
            for case in ("cpu", "gpu", "both"):
                if case in bw:
                    print(
                        f"| {case} | {fmt_stat(bw[case]['cpu'], 2)} | {fmt_stat(bw[case]['gpu'], 2)} |"
                    )
            print()
        handoff_bw = (
            {"cpu": bw["cpu"]["cpu"], "gpu": bw["gpu"]["gpu"]}
            if bw.get("cpu") and bw.get("gpu")
            else None
        )
        for runtime in ("llamacpp", "et"):
            alone = load_alone(serial, runtime)
            if not alone:
                continue
            print(
                f"### {runtime}: alone rates (tok/s, median [min-max] of 3 runs)\n\n{alone_table(alone)}\n"
            )
            print(
                f"### {runtime}: phase affinity (tok/s; CPU = wider set)\n\n"
                f"{affinity_table(serial, runtime, alone)}\n"
            )
            coruns = load_coruns(serial, runtime)
            for model, corun in coruns.items():
                if runtime == "llamacpp":
                    print(f"### {runtime} {model}: co-run (tok/s; ratio = co-run / alone)\n")
                    print(corun_table(serial, alone, corun, model) + "\n")
                else:
                    print(f"### {runtime} {model}: co-run (tok/s; ratio = co-run / solo loop)\n")
                    print(et_corun_table(corun) + "\n")
                    try:
                        print(f"### {runtime} {model}: co-run ratios\n\n{et_ratio_table(corun)}\n")
                    except KeyError as e:
                        print(f"### {runtime} {model}: no co-run ratios (missing {e})\n")
            if runtime == "llamacpp" and handoff_bw:
                print(
                    f"### Decode against the bandwidth bound\n\n{bandwidth_table(serial, alone, bw, coruns.get(CORUN_MODEL, {}))}\n"
                )
            for model in sorted({m for m, _ in alone} - REFERENCE_ONLY):
                rows, why = grid(serial, runtime, model, alone, coruns, handoff_bw)
                if rows is None:
                    print(f"### {runtime} {model}: no prediction grid ({why})\n")
                    unassessed.append(f"{phone} {runtime} {model} (no grid: {why})")
                    continue
                if not rows[0]["gate_eligible"]:
                    unassessed.append(f"{phone} {runtime} {model} (co-run ratios transferred)")
                note = (
                    ""
                    if rows[0]["gate_eligible"]
                    else f" — co-run ratios transferred from {rows[0]['ratios_from']}, not gate-eligible"
                )
                print(
                    f"### {runtime} {model}: predicted request throughput (requests/s){note}\n\n{grid_table(rows)}\n"
                )
                for r in rows:
                    if r["gate_eligible"]:
                        cand = (
                            r["gain"],
                            phone,
                            runtime,
                            model,
                            r["prompt"],
                            r["gen"],
                            r["best_pair"],
                        )
                        best = max(best, cand) if best else cand
                dump[f"{serial}/{runtime}/{model}"] = jsonable(rows)
    if best:
        gain, phone, runtime, model, prompt, gen, pair = best
        verdict = "GO" if gain >= GATE else "NO-GO"
        if verdict == "NO-GO" and unassessed:
            verdict = "NO-GO for the assessed grids"
        print(
            f"\n## Gate\n\n**{verdict}**: best gain {gain:.2f} (gate {GATE}) on {phone}, {runtime}, {model}, "
            f"P={prompt}, D={gen}, prefill on {pair[0].upper()} and decode on {pair[1].upper()}."
        )
    for u in unassessed:
        print(f"Not assessed by the gate: {u}.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dump, f, indent=1, default=str)


if __name__ == "__main__":
    main()
