#!/usr/bin/env python3
"""Cross-tool round-trip contract test (Lever B).

The schedule JSON is the one artifact the Optimizer (this side) writes and the
Implementer (C++ config_reader.hpp) reads. Nothing used to check that "what the
producer writes == what the consumer accepts" — each side validated in isolation.
This pins both to one shared truth: schemas/schedule.schema.json + a committed
example (tests/fixtures/schedule.contract.json).

  - Python (here): the committed fixture is schema-valid, AND a LIVE solver run
    (with the GPU-hardware injection that 02_gen_schedule_merged.py does) produces
    schema-valid output.
  - C++ (test_schedule.cpp ScheduleContract): readSchedulesFromJson +
    validate_schedule_coverage consume the same fixture.

If the producer drifts from the schema -> this reds. If the consumer drifts from
the contract shape -> the C++ test reds. Run:
    uv run python optimizer/tests/test_schedule_contract.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jsonschema  # noqa: E402
from smt.solution_analyzer import validate_against_schema  # noqa: E402
from smt.solver import solve_optimization_problem  # noqa: E402

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SCHEMA = os.path.join(_ROOT, "schemas", "schedule.schema.json")
_FIXTURE = os.path.join(_ROOT, "tests", "fixtures", "schedule.contract.json")


def test_committed_fixture_is_schema_valid():
    """The shared contract example the C++ side consumes must satisfy the schema."""
    with open(_SCHEMA) as f:
        schema = json.load(f)
    with open(_FIXTURE) as f:
        fixture = json.load(f)
    jsonschema.validate(instance=fixture, schema=schema)  # raises on drift


def test_live_producer_output_is_schema_valid():
    """A real solver run, post GPU-hardware injection (exactly as 02 does), must be
    schema-valid -- so the producer can never emit a schedule the consumer rejects."""
    # 11-stage fixture whose makespan-optimal answer SPLITS across GPU + Big -> the
    # output has both a GPU chunk (needs hardware) and a CPU chunk.
    HUGE = 1000.0
    stages = [
        [HUGE, HUGE, (100.0 if s == 0 else 1.0), (100.0 if s == 10 else 1.0)] for s in range(11)
    ]
    solutions = solve_optimization_problem(stages, 5, "cifar-dense", "max_time")
    assert solutions, "solver returned no solutions"

    # Mirror 02_gen_schedule_merged.py: stamp the GPU backend onto GPU chunks.
    saw_gpu = False
    for sol in solutions:
        for chunk in sol["chunks"]:
            if chunk["core_type"] == "GPU":
                chunk["hardware"] = "gpu_cuda"
                saw_gpu = True
    assert saw_gpu, "fixture was meant to force a GPU chunk; producer path untested"

    validate_against_schema(solutions)  # raises if producer output violates the contract

    # And the consumer's required fields are present on every chunk.
    for sol in solutions:
        for chunk in sol["chunks"]:
            assert {"core_type", "start_stage", "end_stage"} <= set(chunk)
            assert chunk["start_stage"] >= 1 and chunk["end_stage"] >= chunk["start_stage"]


if __name__ == "__main__":
    test_committed_fixture_is_schema_valid()
    print("OK  committed fixture is schema-valid")
    test_live_producer_output_is_schema_valid()
    print("OK  live producer output is schema-valid (incl. GPU hardware)")
    print("PASS  schedule round-trip contract")
