#!/usr/bin/env python3
"""Generate the framework vocabulary from vocab.json into a C++ header and a Python
module, so the PU-tier / backend / app-stage vocabulary has ONE source instead of the
~6 hand-duplicated enum sites (conf.hpp, conf.cpp, config_reader, data_loader.py,
baselines.py, the schema). Same mechanism as scripts/embed_device_specs.py: committed
generated outputs are the fallback, regenerated at build time by the CMake bt_vocab target.

Regenerate after editing vocab.json:
    python3 scripts/embed_vocab.py
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOCAB = ROOT / "vocab.json"
OUT_HPP = ROOT / "platform" / "vocab" / "generated" / "bt_vocab.hpp"
OUT_PY = ROOT / "optimizer" / "smt" / "bt_vocab.py"

if not VOCAB.exists():
    sys.exit("vocab.json not found")

v = json.loads(VOCAB.read_text(encoding="utf-8"))


def _write_if_changed(path, content):
    # Only touch the file when content changes: clean tree stays clean, read-only CI is
    # not forced to rewrite an identical committed file (review #20). utf-8 pinned (#23).
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


pus = v["processor_types"]
named = [p for p in pus if p.get("name")]  # CoreTypeName/parse handle the named tiers
cpu_tiers = [p["name"] for p in pus if p.get("solver_cpu_tier")]

# ---- C++ : builtin-apps/generated/bt_vocab.hpp ----------------------------------
hpp = [
    "#pragma once",
    "// AUTO-GENERATED from vocab.json by scripts/embed_vocab.py -- DO NOT EDIT.",
    "// Regenerate after changing vocab.json.",
    "",
    "#include <stdexcept>",
    "#include <string>",
    "",
    "// Processing-unit class. Values are load-bearing (schedules index them).",
    "enum class ProcessorType {",
]
hpp += [f"  {p['cpp']} = {p['value']}," for p in pus]
hpp += [
    "};",
    "",
    "inline std::string CoreTypeName(const ProcessorType core_type) {",
    "  switch (core_type) {",
]
hpp += [f'    case ProcessorType::{p["cpp"]}:\n      return "{p["name"]}";' for p in named]
hpp += [
    "    default:",
    '      return "unknown";',
    "  }",
    "}",
    "",
    "inline ProcessorType ParseCoreType(const std::string& s) {",
]
hpp += [f'  if (s == "{p["name"]}") return ProcessorType::{p["cpp"]};' for p in named]
hpp += [
    '  throw std::runtime_error("unknown core type \'" + s + "\'");',
    "}",
    "",
    "// Per-app pipeline stage counts (single source for the AppTraits specializations).",
    "namespace bt::vocab {",
]


def _app_const(key):
    # tree -> kTreeStages, cifar-dense -> kCifarDenseStages
    return "k" + "".join(part.capitalize() for part in key.replace("-", "_").split("_")) + "Stages"


hpp += [f"inline constexpr int {_app_const(k)} = {n};" for k, n in v["app_stages"].items()]
hpp += [
    "}  // namespace bt::vocab",
    "",
]
_write_if_changed(OUT_HPP, "\n".join(hpp))

# ---- Python : optimizer/smt/bt_vocab.py -----------------------------------------
py = [
    '"""AUTO-GENERATED from vocab.json by scripts/embed_vocab.py -- DO NOT EDIT.',
    "Regenerate after changing vocab.json.",
    '"""',
    "",
    f"# Solver CPU tiers, in cost-matrix column order (the orphaned 'super' tier is",
    f"# intentionally NOT here -- it is absent from the z3 tier list today).",
    f"CPU_TIERS = {tuple(cpu_tiers)!r}",
    "",
    f"# Solver core-type columns: the CPU tiers (display-cased) + the GPU column.",
    f"CORE_TYPES = {list(v['solver_core_types'])!r}",
    "",
    f"# Application stage counts.",
    f"APP_STAGES = {dict(v['app_stages'])!r}",
    "",
]
_write_if_changed(OUT_PY, "\n".join(py))

print(f"wrote {OUT_HPP.relative_to(ROOT)} and {OUT_PY.relative_to(ROOT)}")
