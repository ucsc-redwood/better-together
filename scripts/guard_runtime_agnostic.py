#!/usr/bin/env python3
"""guard-runtime-agnostic: fail if runtime/ references any application by name in CODE
(comments excluded). Enforces that bt::runtime stays application-agnostic -- the
decoupling P3 achieved (the executor that hardcoded tree::SafeAppData is gone). Catches a
header-only leak that link-scoping cannot (e.g. a runtime header using tree::SafeAppData
or #include "apps/..."). Wired as a CTest (LABEL "guard").

Run:  python3 scripts/guard_runtime_agnostic.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
EXTS = (".hpp", ".cpp", ".cu", ".cuh")

# The forbidden app namespaces are DERIVED from vocab.json (the single source of truth), so
# adding a 4th app there automatically guards its `<app>::` namespace -- no edit here needed.
_apps = json.load(open(os.path.join(ROOT, "vocab.json")))["app_stages"].keys()
_app_ns = [re.escape(a.replace("-", "_")) + "::" for a in _apps]  # tree:: cifar_dense:: ...
# App-internal identifiers vocab.json does not name (sub-namespaces / kernel identifiers).
_extra = [r"octree::", r"cifar::", r"\bmorton\b"]
# Real code leaks: an app namespace/identifier or an apps/ include. (Prose in // and
# /* */ comments is stripped first, so explanatory comments naming apps are allowed.)
FORBID = re.compile("|".join(_app_ns + _extra) + r'|#\s*include\s*"apps/')


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)   # block comments
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


bad = []
for dp, _, files in os.walk(RUNTIME):
    for fn in files:
        if not fn.endswith(EXTS):
            continue
        fp = os.path.join(dp, fn)
        code = strip_comments(open(fp).read())
        for i, line in enumerate(code.splitlines(), 1):
            if FORBID.search(line):
                bad.append(f"{os.path.relpath(fp, ROOT)}: {line.strip()}")

if bad:
    print("guard-runtime-agnostic FAIL -- runtime/ references an app in CODE:")
    print("\n".join(bad))
    sys.exit(1)
print("guard-runtime-agnostic OK -- runtime/ is application-agnostic")
