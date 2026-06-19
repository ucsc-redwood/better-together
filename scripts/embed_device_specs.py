#!/usr/bin/env python3
"""Embed devices/*.json into a C++ header so the runtime DeviceRegistry can be
data-driven without a runtime file dependency (works on-device, in tests, in the
cross-build -- nothing to push alongside the binary).

Regenerate after changing any devices/*.json:
    uv run scripts/embed_device_specs.py   # or: python3 scripts/embed_device_specs.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEVICES = sorted((ROOT / "devices").glob("*.json"))
OUT = ROOT / "platform" / "registry" / "generated" / "device_specs_embedded.hpp"

if not DEVICES:
    sys.exit("no devices/*.json found")

parts = [
    "#pragma once",
    "// AUTO-GENERATED from devices/*.json by scripts/embed_device_specs.py -- DO NOT EDIT.",
    "// Regenerate after changing any devices/*.json.",
    "",
    "#include <string_view>",
    "#include <vector>",
    "",
    "namespace bt::device_specs {",
    "",
    "// One raw-JSON device spec per registered device (schema: schemas/device-spec.schema.json).",
    "inline const std::vector<std::string_view> kEmbedded = {",
]

for f in DEVICES:
    text = f.read_text(encoding="utf-8").rstrip("\n")
    # raw string delimiter chosen to not collide with JSON contents
    parts.append(f"    // {f.name}")
    parts.append(f'    R"DEVSPEC({text})DEVSPEC",')

parts += ["};", "", "}  // namespace bt::device_specs", ""]

content = "\n".join(parts) + "\n"
# Idempotent write: only touch the file when the content actually changes, so a clean
# tree stays clean and a read-only checkout (CI) is not forced to rewrite an identical
# committed header on every configure (review #20). Encoding pinned to utf-8 (#23).
if not OUT.exists() or OUT.read_text(encoding="utf-8") != content:
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(DEVICES)} device specs")
else:
    print(f"{OUT.relative_to(ROOT)} unchanged ({len(DEVICES)} device specs)")
