#!/usr/bin/env python3
"""Gate for analysis/llm_phase_affinity.py: the design-D5 prediction on a hand-computed mix, the
KV hand-off size, and the co-run overlap filter.

    uv run pytest -q optimizer/tests/test_llm_phase_affinity.py
"""

import math

from analysis.llm_phase_affinity import handoff_seconds, overlapped_rate, predict_mix


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


def test_overlapped_rate_keeps_only_covered_reps():
    # Partner reps run back to back over [10, 40]; my reps: one starts before the partner, two
    # inside, one ends after it.
    partner = [(10.0, 20.0, 100), (20.01, 30.0, 100), (30.0, 40.0, 100)]
    mine = [(5.0, 12.0, 70), (12.0, 22.0, 100), (22.0, 34.0, 60), (34.0, 45.0, 110)]
    rate, kept, total = overlapped_rate(mine, partner)
    assert (kept, total) == (2, 4)
    assert math.isclose(rate, (100 + 60) / (10 + 12))
