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
    text = f.read_text().rstrip("\n")
    # raw string delimiter chosen to not collide with JSON contents
    parts.append(f"    // {f.name}")
    parts.append(f'    R"DEVSPEC({text})DEVSPEC",')

parts += ["};", "", "}  // namespace bt::device_specs", ""]

OUT.write_text("\n".join(parts) + "\n")
print(f"wrote {OUT.relative_to(ROOT)} with {len(DEVICES)} device specs")
