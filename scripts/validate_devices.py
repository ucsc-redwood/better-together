#!/usr/bin/env python3
"""Validate the declarative device specs under devices/.

Two layers of checking:

1. Structural validation against schemas/device-spec.schema.json (a pragmatic,
   dependency-free subset check -- no jsonschema package required).
2. A *characterization* check: every device is compared against the golden core
   topology transcribed from builtin-apps/conf.cpp at commit 109bcf1 (the
   published artifact). This locks the published values so that a future edit to
   a device file -- or to conf.cpp during the C++ migration -- cannot silently
   change a device's affinity map without this test failing.

Run:  uv run scripts/validate_devices.py     (or: python3 scripts/validate_devices.py)
Exit code 0 = all good, 1 = a mismatch or structural error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICES_DIR = REPO_ROOT / "devices"
SCHEMA_PATH = REPO_ROOT / "schemas" / "device-spec.schema.json"

VALID_TYPES = {"little", "medium", "big", "super"}


# --- Golden topology, transcribed verbatim from builtin-apps/conf.cpp@109bcf1 ---
# Each entry is the ordered list of (core_id, type, pinnable) as it appears in
# the DeviceRegistry constructor.
def _run(id_type_pairs):
    """Expand [(ids, type, pinnable), ...] into a flat ordered core list."""
    out = []
    for ids, ctype, pinnable in id_type_pairs:
        for cid in ids:
            out.append((cid, ctype, pinnable))
    return out


GOLDEN = {
    "pc": _run([(range(0, 8), "big", True), (range(8, 24), "little", True)]),
    "jetson": _run([(range(0, 6), "little", True)]),
    "jetsonlowpower": _run([(range(0, 4), "little", True)]),
    "3A021JEHN02756": _run(
        [(range(0, 4), "little", True), (range(4, 6), "medium", True), (range(6, 8), "big", True)]
    ),
    "9b034f1b": _run(
        [(range(0, 3), "little", True), (range(3, 5), "medium", True), (range(5, 8), "big", False)]
    ),
    "ce0717178d7758b00b7e": _run(
        [(range(4, 8), "little", True), (range(0, 4), "big", True)]
    ),
    "R9TR30814KJ": _run(
        [(range(0, 4), "little", True), (range(4, 7), "big", False), ([7], "big", True)]
    ),
    "minipc": _run([(range(0, 16), "big", True)]),
    "R5CY21Y3VEV": _run(
        [
            (range(0, 4), "little", True),
            (range(4, 7), "medium", True),
            (range(7, 9), "big", True),
            ([9], "super", True),
        ]
    ),
}


def structural_errors(name: str, spec: object) -> list[str]:
    errs: list[str] = []
    if not isinstance(spec, dict):
        return [f"{name}: top-level value is not an object"]
    for key in ("id", "cores"):
        if key not in spec:
            errs.append(f"{name}: missing required key '{key}'")
    extra = set(spec) - {"id", "description", "cores"}
    if extra:
        errs.append(f"{name}: unexpected key(s) {sorted(extra)}")
    if spec.get("id") != name:
        errs.append(f"{name}: 'id' ({spec.get('id')!r}) must match file stem")
    cores = spec.get("cores")
    if not isinstance(cores, list) or not cores:
        errs.append(f"{name}: 'cores' must be a non-empty array")
        return errs
    seen_ids = set()
    for i, core in enumerate(cores):
        if set(core) != {"id", "type", "pinnable"}:
            errs.append(f"{name}: core[{i}] keys must be exactly id/type/pinnable, got {sorted(core)}")
            continue
        if not isinstance(core["id"], int) or isinstance(core["id"], bool) or core["id"] < 0:
            errs.append(f"{name}: core[{i}].id must be a non-negative integer")
        if core["type"] not in VALID_TYPES:
            errs.append(f"{name}: core[{i}].type {core['type']!r} not in {sorted(VALID_TYPES)}")
        if not isinstance(core["pinnable"], bool):
            errs.append(f"{name}: core[{i}].pinnable must be a boolean")
        if core.get("id") in seen_ids:
            errs.append(f"{name}: duplicate core id {core['id']}")
        seen_ids.add(core.get("id"))
    return errs


def golden_errors(name: str, spec: object) -> list[str]:
    if not isinstance(spec, dict) or not isinstance(spec.get("cores"), list):
        return []  # structural_errors already reported the problem
    if name not in GOLDEN:
        return [f"{name}: no golden entry (new device) -- add it to GOLDEN in this script "
                "after confirming the topology, so it is locked against regressions"]
    got = [(c["id"], c["type"], c["pinnable"]) for c in spec["cores"]]
    want = GOLDEN[name]
    if got != want:
        return [f"{name}: topology differs from published conf.cpp golden\n"
                f"      golden: {want}\n"
                f"      file:   {got}"]
    return []


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"WARN: schema not found at {SCHEMA_PATH}", file=sys.stderr)
    files = sorted(DEVICES_DIR.glob("*.json"))
    if not files:
        print(f"ERROR: no device specs found in {DEVICES_DIR}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    rows = []
    for f in files:
        name = f.stem
        try:
            spec = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            all_errors.append(f"{name}: invalid JSON ({e})")
            continue
        errs = structural_errors(name, spec) + golden_errors(name, spec)
        all_errors.extend(errs)
        cores = spec.get("cores", []) if isinstance(spec, dict) else []
        tally: dict[str, int] = {}
        for c in cores:
            if isinstance(c, dict):
                tally[c.get("type", "?")] = tally.get(c.get("type", "?"), 0) + 1
        summary = " ".join(f"{t}:{tally[t]}" for t in ("little", "medium", "big", "super") if t in tally)
        rows.append((name, len(cores), summary, "OK" if not errs else "FAIL"))

    # Report any golden devices that have no file at all.
    have = {f.stem for f in files}
    for name in GOLDEN:
        if name not in have:
            all_errors.append(f"{name}: present in golden but missing devices/{name}.json")

    width = max((len(r[0]) for r in rows), default=8)
    print(f"{'device'.ljust(width)}  cores  topology")
    print(f"{'-' * width}  -----  --------")
    for name, n, summary, status in rows:
        mark = "ok " if status == "OK" else "ERR"
        print(f"{name.ljust(width)}  {str(n).rjust(5)}  {summary}   [{mark}]")

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} problem(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {len(rows)} device spec(s) valid and match the published conf.cpp golden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
