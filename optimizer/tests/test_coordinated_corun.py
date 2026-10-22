#!/usr/bin/env python3
"""Gate for analysis/coordinated_corun.py: the design-D4 prediction on a hand-computed mix, the name
parser, and the co-run filter that keeps only units inside the partner's measured interval.

    uv run pytest -q optimizer/tests/test_coordinated_corun.py
"""

import math

from analysis import llm_phase_affinity as lpa
from analysis.coordinated_corun import (
    Refs,
    dp_throughput,
    parse_key,
    predict,
    side_rates,
    spec_throughput,
    units,
)


def test_parse_key():
    k = parse_key("1b_gpuP_cpuD_p2048_u67_m1c0_c256")
    assert k == {
        "model": "1b",
        "phases": ["gpuP", "cpuD"],
        "P": 2048,
        "u": 0.67,
        "mask": "1c0",
        "C": 256,
    }
    assert parse_key("3b_gpuP_p128_u100")["C"] == 128
    assert parse_key("1b_cpuD_d512_m180") == {
        "model": "1b",
        "phases": ["cpuD"],
        "depth": 512,
        "mask": "180",
        "u": 1.0,
    }


def test_units_and_side_rates():
    # Two prefill repetitions of two 256-token ubatches, each ubatch 1 s of compute and 1 s of gap
    # (duty 0.5). The partner's interval [0, 5] covers the first repetition [0, 4] only.
    steps = [
        {
            "rep": r,
            "step": i,
            "start": 4 * r + 2 * i,
            "end": 4 * r + 2 * i + 1,
            "gap_end": 4 * r + 2 * i + 2,
            "tokens": 256,
        }
        for r in (0, 1)
        for i in (0, 1)
    ]
    us = units(steps, prefill=True)
    assert us == [(0, 3, 4, 2, 512), (4, 7, 8, 2, 512)]
    r = side_rates(us, (0, 5))
    assert r["kept"] == (1, 2)
    assert math.isclose(r["wall"], 512 / 4) and math.isclose(r["compute"], 512 / 2)
    assert math.isclose(r["duty"], 0.5)
    # A repetition missing a ubatch (the process stopped) is not a unit; decode units are tokens.
    assert len(units(steps[:3], prefill=True)) == 1
    assert len(units(steps, prefill=False)) == 4
    assert side_rates(us, (1, 9)) is not None and side_rates(us, (1, 7)) is None


def test_throughputs():
    assert math.isclose(spec_throughput(512, 128, 450, 18, 0.1), 1 / (128 / 18 + 0.1))
    assert math.isclose(spec_throughput(512, 32, 100, 18, 0.0), 1 / 5.12)
    assert math.isclose(dp_throughput(512, 128, [(500, 8), (100, 8)]), 1 / 17.024 + 1 / 21.12)


def test_predict_hand_computed():
    # P = 512, D = 128, hand-off 0.1 s. Alone (duty 1): GPU prefill 500, GPU decode at depth 512: 10;
    # CPU prefill 100 (3f0), CPU decode at depth 512: 20 (3f0), 18 (1c0).
    st = {
        "gpuP_p512_u100": 500,
        "gpuD_d512_u100": 10,
        "cpuP_p512_m3f0": 100,
        "cpuD_d512_m3f0": 20,
        "cpuD_d512_m1c0": 18,
    }
    alone = {f"1b_{k}": {"wall": lpa.stat([v], [k])} for k, v in st.items()}
    sets = {
        # GPU prefill -> CPU decode, (prefill ratio, decode ratio):
        #   a, duty 1:   p' = 450, d' = 10 -> period max(1.138, 12.8) + 0.1 = 12.9 s
        #   b, duty 0.5: p' = 250, d' = 18 -> period max(2.048, 7.111) + 0.1 = 7.2111 s
        "gpu->cpu": [
            {
                "key": "a",
                "u": 1.0,
                "mask": "3f0",
                "P": 512,
                "C": 512,
                "ratios": (0.9, 0.5),
                "files": [],
            },
            {
                "key": "b",
                "u": 0.5,
                "mask": "3f0",
                "P": 512,
                "C": 512,
                "ratios": (0.5, 0.9),
                "files": [],
            },
            {
                "key": "other P",
                "u": 0.5,
                "mask": "3f0",
                "P": 2048,
                "C": 512,
                "ratios": (1, 1),
                "files": [],
            },
        ],
        "cpu->gpu": [],
        # Data-parallel, (GPU decode ratio, CPU decode ratio):
        #   x, duty 1, 3f0:   1/(1.024 + 128/8) + 1/(5.12 + 128/8)    = 0.058741 + 0.047348 = 0.106089
        #   y, duty 0.5, 1c0: 1/(1.024 + 128/5) + 1/(5.12 + 128/16.2) = 0.037560 + 0.076801 = 0.114361
        "dp": [
            {
                "key": "x",
                "u": 1.0,
                "mask": "3f0",
                "P": None,
                "C": None,
                "ratios": (0.8, 0.4),
                "files": [],
            },
            {
                "key": "y",
                "u": 0.5,
                "mask": "1c0",
                "P": None,
                "C": None,
                "ratios": (0.5, 0.9),
                "files": [],
            },
        ],
    }
    # Trial spread: every setting's slowest and fastest trial equal its median, except b's slowest
    # (prefill 0.5, decode 0.8) and y's fastest (GPU 0.5, CPU 1.0).
    for f in sets.values():
        for x in f:
            x["ratios_lo"] = x["ratios_hi"] = x["ratios"]
    sets["gpu->cpu"][1]["ratios_lo"] = (0.5, 0.8)
    sets["dp"][1]["ratios_hi"] = (0.5, 1.0)
    # Uncoordinated (llm-phase-affinity) data-parallel 0.1 and single PU 0.08: both below y.
    unc = {"dp_corun": 0.1, "single": 0.08, "single_pu": "cpu", "specialized": {}}
    r = predict(512, 128, Refs(alone, "1b"), sets, 0.1, unc)
    assert r["spec_setting"] == "b"
    assert math.isclose(r["spec"], 1 / (128 / 18 + 0.1))
    dp_y = 1 / (1.024 + 25.6) + 1 / (5.12 + 128 / 16.2)
    assert r["dp_setting"] == "y" and math.isclose(r["dp"], dp_y)
    # Best single PU: CPU on 3f0, 1/(5.12 + 6.4); GPU 1/13.824, CPU 1c0 1/(5.12 + 7.111).
    assert r["single_pu"] == "CPU 3f0" and math.isclose(r["single"], 1 / 11.52)
    assert r["baseline_from"] == "coordinated data-parallel" and math.isclose(r["baseline"], dp_y)
    assert math.isclose(r["gain"], (1 / (128 / 18 + 0.1)) / dp_y)  # 1.2126...
    assert 1.21 < r["gain"] < 1.22
    # Least favourable: b at 1/(max(2.048, 128/16) + 0.1) = 1/8.1 against y at its fastest,
    # 1/(1.024 + 25.6) + 1/(5.12 + 128/18) = 0.119317; most favourable: the medians (y at its slowest).
    dp_y_hi = 1 / (1.024 + 25.6) + 1 / (5.12 + 128 / 18)
    assert math.isclose(r["gain_lo"], (1 / 8.1) / dp_y_hi)  # 1.0347...
    assert math.isclose(r["gain_hi"], r["gain"])
    # The duty-1 widest-mask pair alone gives a, against x.
    assert math.isclose(r["spec_u1"], 1 / 12.9) and math.isclose(r["dp_u1"], 1 / 17.024 + 1 / 21.12)
    # A larger uncoordinated data-parallel bound becomes the baseline.
    r2 = predict(512, 128, Refs(alone, "1b"), sets, 0.1, {**unc, "dp_corun": 0.2})
    assert r2["baseline"] == 0.2 and r2["baseline_from"].startswith("uncoordinated")
    # The diagnostic against this study's own baseline ignores llm-phase-affinity's values.
    assert math.isclose(r2["baseline_own"], dp_y) and math.isclose(r2["gain_own"], r["gain"])
