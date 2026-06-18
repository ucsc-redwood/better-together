"""P4 drift guard: the generated bt_vocab must stay consistent with vocab.json and with
the sites that used to hand-duplicate the vocabulary (data_loader, baselines, the schema).
Run:  pytest optimizer/tests/test_vocab.py
"""
import json
import pathlib

from smt.bt_vocab import CPU_TIERS, CORE_TYPES, APP_STAGES
from smt.data_loader import define_data
from smt.baselines import get_num_stages_for_app

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_bt_vocab_matches_vocab_json():
    v = json.loads((ROOT / "vocab.json").read_text())
    assert list(CORE_TYPES) == v["solver_core_types"]
    assert list(CPU_TIERS) == [p["name"] for p in v["processor_types"] if p.get("solver_cpu_tier")]
    assert APP_STAGES == v["app_stages"], "committed bt_vocab.py is stale vs vocab.json -- regenerate"


def test_core_types_match_data_loader():
    # define_data() is what feeds z3 the column layout; it must equal CORE_TYPES.
    _, core_types, _ = define_data(app_name="tree")
    assert core_types == list(CORE_TYPES)


def test_cpu_tiers_are_core_types_minus_gpu():
    assert [t.capitalize() for t in CPU_TIERS] + ["GPU"] == list(CORE_TYPES)


def test_app_stages_match_lookup():
    for app, n in APP_STAGES.items():
        assert get_num_stages_for_app(app) == n


def _enums(obj):
    out = []
    if isinstance(obj, dict):
        if isinstance(obj.get("enum"), list):
            out.append(obj["enum"])
        for val in obj.values():
            out += _enums(val)
    elif isinstance(obj, list):
        for item in obj:
            out += _enums(item)
    return out


def test_schema_core_type_enum_matches_vocab():
    schema = json.loads((ROOT / "schemas" / "schedule.schema.json").read_text())
    assert list(CORE_TYPES) in _enums(schema), \
        "schedule.schema.json core_type enum drifted from vocab.json CORE_TYPES"
