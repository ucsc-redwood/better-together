#!/usr/bin/env python3
"""Gate for analysis/llm_phase_affinity.py: the design-D5 prediction on a hand-computed mix, the
KV hand-off size, and the co-run overlap filter.

    uv run pytest -q optimizer/tests/test_llm_phase_affinity.py
"""

import json
import math

import pytest
from analysis.llm_phase_affinity import (
    et_rate,
    handoff_seconds,
    load_et_corun,
    overlapped_rate,
    partial_overlap_period,
    predict_mix,
)


def test_predict_mix_hand_computed():
    # P = 512, D = 128. CPU: prefill 100, decode 20 tok/s; GPU: prefill 500, decode 10 tok/s.
    #   T_cpu = 512/100 + 128/20 = 5.12 + 6.4 = 11.52 s;  T_gpu = 512/500 + 128/10 = 1.024 + 12.8 = 13.824 s
    #   best single = CPU, 1/11.52;  ideal data-parallel = 1/11.52 + 1/13.824
    # Both decoding: CPU 16, GPU 8 tok/s -> 1/(5.12 + 8) + 1/(1.024 + 16)
    # GPU prefill + CPU decode: prefill 450, decode 18, hand-off 0.1 s -> 1/(max(512/450, 128/18) + 0.1)
    # CPU prefill + GPU decode: prefill 90, decode 9 -> 1/(max(512/90, 128/9) + 0.1)
    r = predict_mix(
        512,
        128,
        alone={"cpu": (100, 20), "gpu": (500, 10)},
        both_decode={"cpu": 16, "gpu": 8},
        specialized={("gpu", "cpu"): (450, 18), ("cpu", "gpu"): (90, 9)},
        handoff_s=0.1,
    )
    assert r["single_pu"] == "cpu"
    assert math.isclose(r["single"], 1 / 11.52)
    assert math.isclose(r["dp_alone"], 1 / 11.52 + 1 / 13.824)
    dp_corun = 1 / 13.12 + 1 / 17.024
    assert math.isclose(r["dp_corun"], dp_corun)
    gpu_cpu = 1 / (128 / 18 + 0.1)
    assert math.isclose(r["specialized"][("gpu", "cpu")], gpu_cpu)
    assert math.isclose(r["specialized"][("cpu", "gpu")], 1 / (128 / 9 + 0.1))
    assert r["best_pair"] == ("gpu", "cpu")
    assert math.isclose(r["gain"], gpu_cpu / dp_corun)  # 1.0275...
    assert 1.02 < r["gain"] < 1.03


def test_handoff_seconds():
    # 1B: 2 (K, V) x 16 layers x 8 kv heads x 64 dims x 2048 tokens x 2 bytes = 64 MiB, read + write.
    nbytes = 2 * 16 * 8 * 64 * 2048 * 2
    assert nbytes == 64 * 2**20
    assert math.isclose(handoff_seconds("1b", 2048, 10.0), 2 * nbytes / 10e9)


def test_partial_overlap_period():
    # P = 100, D = 10; alone: prefill 100, decode 10 tok/s.
    # Co-run prefill 100 (1 s), decode 5 (2 s): the prefill is shorter, so the decode runs 1 s at 5 and
    # the remaining 5 tokens alone at 10: 1 + 0.5 = 1.5 s.
    assert math.isclose(partial_overlap_period(100, 10, (100, 10), (100, 5)), 1.5)
    # Co-run prefill 25 (4 s), decode 5 (2 s): the decode is shorter, so the prefill runs 2 s at 25 and
    # the remaining 50 tokens alone at 100: 2 + 0.5 = 2.5 s.
    assert math.isclose(partial_overlap_period(100, 10, (100, 10), (25, 5)), 2.5)


def test_overlapped_rate_keeps_only_covered_reps():
    # Partner reps run back to back over [10, 40]; my reps: one starts before the partner, two
    # inside, one ends after it.
    partner = [(10.0, 20.0, 100), (20.01, 30.0, 100), (30.0, 40.0, 100)]
    mine = [(5.0, 12.0, 70), (12.0, 22.0, 100), (22.0, 34.0, 60), (34.0, 45.0, 110)]
    rate, kept, total = overlapped_rate(mine, partner)
    assert (kept, total) == (2, 4)
    assert math.isclose(rate, (100 + 60) / (10 + 12))


def _observer(load_end, start, prompt_end, end, prompt_tokens, generated):
    """One llama_main PyTorchObserver line (epoch seconds in, milliseconds out)."""
    r = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "model_load_start_ms": int((load_end - 1) * 1000),
        "model_load_end_ms": int(load_end * 1000),
        "inference_start_ms": int(start * 1000),
        "prompt_eval_end_ms": int(prompt_end * 1000),
        "inference_end_ms": int(end * 1000),
    }
    return "PyTorchObserver " + json.dumps(r) + "\n"


def test_et_corun_counts_only_fully_overlapped_windows(tmp_path):
    # Prefill process A: warm-up prefill [10, 11], measured prefill [11, 12], then it ends.
    # Decode process B (8-token loop): one window inside A's warm-up, one straddling the end of A's
    # compute, one after A finished.
    trial = tmp_path / "gpuP_cpuDs_p512_t1"
    trial.mkdir()
    (trial / "A.out").write_text(_observer(10.0, 11.0, 12.0, 12.0, 512, 0))
    (trial / "B.out").write_text(
        _observer(10.1, 10.1, 10.2, 10.6, 16, 8)
        + _observer(11.4, 11.4, 11.5, 12.5, 16, 8)
        + _observer(12.9, 12.9, 13.0, 13.4, 16, 8)
    )
    solo = tmp_path / "solo_cpuDs_t1"
    solo.mkdir()
    (solo / "A.out").write_text(
        _observer(0.9, 0.9, 1.0, 1.4, 16, 8) + _observer(1.9, 1.9, 2.0, 3.0, 16, 8)
    )
    (solo / "B.out").write_text("")

    c = load_et_corun(str(tmp_path))
    b = c["gpuP_cpuDs_p512"]["B"]
    assert b["kept"] == [(1, 3)]
    assert math.isclose(b["rate"]["median"], 8 / 0.4)
    # The previous estimate still sees every window: 24 tokens over 1.8 s.
    assert math.isclose(b["rate_all"]["median"], 24 / 1.8)
    # A's prefill [11, 12] lies inside no decode window of B: nothing counts.
    assert c["gpuP_cpuDs_p512"]["A"]["kept"] == [(0, 1)]
    assert c["gpuP_cpuDs_p512"]["A"]["rate"] is None
    # A solo loop counts every window.
    assert c["solo_cpuDs"]["A"]["kept"] == [(2, 2)]
    assert math.isclose(c["solo_cpuDs"]["A"]["rate"]["median"], 16 / 1.4)
    # One trial is not enough for the gate.
    with pytest.raises(KeyError):
        et_rate(c, "gpuP_cpuDs_p512", "B")
