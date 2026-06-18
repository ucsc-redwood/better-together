#!/usr/bin/env python3
"""guard-runtime-agnostic: fail if runtime/ references any application by name in CODE
(comments excluded). Enforces that bt::runtime stays application-agnostic -- the
decoupling P3 achieved (the executor that hardcoded tree::SafeAppData is gone). Catches a
header-only leak that link-scoping cannot (e.g. a runtime header using tree::SafeAppData
or #include "apps/..."). Wired as a CTest (LABEL "guard").

Run:  python3 scripts/guard_runtime_agnostic.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
EXTS = (".hpp", ".cpp", ".cu", ".cuh")
# Real code leaks: an app namespace/identifier or an apps/ include. (Prose in // and
# /* */ comments is stripped first, so explanatory comments naming apps are allowed.)
FORBID = re.compile(r'tree::|cifar_dense|cifar_sparse|cifar::|octree::|\bmorton\b|#\s*include\s*"apps/')


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
