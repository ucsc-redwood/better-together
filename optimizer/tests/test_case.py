#!/usr/bin/env python3
"""Unit test for the Case path builder (the data-layout single source of truth).

Run:  uv run python optimizer/tests/test_case.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from orchestrate.case import Case, table_to_scenario, to_short_backend  # noqa: E402


def test_short_long_backend():
    assert Case("jetson", "tree", "cu").backend_long == "cuda"
    assert Case("minipc", "tree", "vk").backend_long == "vulkan"
    assert to_short_backend("cuda") == "cu"
    assert to_short_backend("vulkan") == "vk"
    assert to_short_backend("cu") == "cu"  # idempotent


def test_table_to_scenario():
    # The z3 --table_type token maps to the profiling-store scenario dir.
    assert table_to_scenario("isolated") == "isolated"
    assert table_to_scenario("btpm") == "interference"


def test_paths():
    c = Case("jetson", "tree", "cu")
    assert c.schedule_path("data/schedules", "btpm", "tmax") == (
        "data/schedules/jetson/tree/cu/schedules_btpm_tmax.json"
    )
    # profiling store uses the LONG backend name.
    assert c.profiling_glob("data/profiling", "isolated") == (
        "data/profiling/jetson/tree/cuda/isolated/run-*.jsonl"
    )


def test_from_profiling_relpath():
    c = Case.from_profiling_relpath(("R5CY21Y3VEV", "cifar-dense", "vulkan", "interference"))
    assert c == Case("R5CY21Y3VEV", "cifar-dense", "vk")


if __name__ == "__main__":
    test_short_long_backend()
    test_table_to_scenario()
    test_paths()
    test_from_profiling_relpath()
    print("PASS  Case path builder")
