#!/usr/bin/env python3
"""Unit test for the Case path builder (the data-layout single source of truth).

Run:  uv run python scripts/collect/test_case.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from case import Case, to_short_backend  # noqa: E402


def test_short_long_backend():
    assert Case("jetson", "tree", "cu").backend_long == "cuda"
    assert Case("minipc", "tree", "vk").backend_long == "vulkan"
    assert to_short_backend("cuda") == "cu"
    assert to_short_backend("vulkan") == "vk"
    assert to_short_backend("cu") == "cu"  # idempotent


def test_paths():
    c = Case("jetson", "tree", "cu")
    assert c.schedule_path("data/schedules", "btpm", "tmax") == (
        "data/schedules/jetson/tree/cu/schedules_btpm_tmax.json"
    )
    assert c.csv_path("data/btpm_export", "isolated") == (
        "data/btpm_export/jetson/tree/cu/isolated.csv"
    )
    # interference scenario maps to the btpm.csv name (the paper's BTPM table).
    assert c.csv_path("data/btpm_export", "interference").endswith("/btpm.csv")
    # profiling store uses the LONG backend name.
    assert c.profiling_glob("data/profiling", "isolated") == (
        "data/profiling/jetson/tree/cuda/isolated/run-*.jsonl"
    )


def test_from_profiling_relpath():
    c = Case.from_profiling_relpath(("R5CY21Y3VEV", "cifar-dense", "vulkan", "interference"))
    assert c == Case("R5CY21Y3VEV", "cifar-dense", "vk")


if __name__ == "__main__":
    test_short_long_backend()
    test_paths()
    test_from_profiling_relpath()
    print("PASS  Case path builder")
