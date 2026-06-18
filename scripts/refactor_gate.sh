#!/usr/bin/env bash
# refactor_gate.sh -- behavior-preserving gate for the component-structure refactor.
#
# The refactor (docs/reports-for-human/target-structure.md, phases P0-P6) adds NO
# behavior. So the gate is EQUIVALENCE to a pre-refactor baseline, not new tests:
#   * the same set of tests still exists (no target silently dropped),
#   * the test count is non-zero and unchanged ("green" with 0 tests is a lie),
#   * no test flipped PASS->FAIL or RUN->SKIP (a new GTEST_SKIP hides a regression),
#   * known pre-existing failures (e.g. AlternatingBoundary) are captured in the
#     baseline so they are NOT attributed to the refactor.
#
# Usage:
#   scripts/refactor_gate.sh capture-baseline   # run ONCE on a clean tree before P0
#   scripts/refactor_gate.sh check              # run at the end of every phase; exits !=0 on regression
#
# Env knobs:
#   RUN_PRESET   preset to BUILD + RUN locally for pass/skip (default: pc)
#   RUN_LABELS   ctest label filter for the run     (default: omp)   e.g. "unit" or "omp"
#   BUILD_DIRS   space-separated build dirs to inventory (default: auto-detect build/*/)
#
# GPU presets (cuda/vulkan) compile/run only on their targets -- run this script
# ON the fleet (Jetson / rocky) after cross-building to capture & check those rows.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STORE=".refactor-gate"
RUN_PRESET="${RUN_PRESET:-pc}"
RUN_LABELS="${RUN_LABELS:-omp}"

# auto-detect configured build dirs (have a CTestTestfile.cmake)
detect_dirs() {
  if [ -n "${BUILD_DIRS:-}" ]; then printf '%s\n' $BUILD_DIRS; return; fi
  for d in build/*/; do
    [ -f "${d}CTestTestfile.cmake" ] && printf '%s\n' "${d%/}"
  done
}

# sorted list of test NAMES in a build dir (configure-time inventory)
# Tolerate ctest's right-aligned numbering ("Test  #1" vs "Test #10" once a dir crosses
# 10 tests) and drop any path-noise lines (cross-build dirs print /workspace/... entries).
inventory() { ctest --test-dir "$1" -N 2>/dev/null | sed -n 's/^ *Test  *#[0-9]*: //p' | grep -vE '/|^$' | sort; }

# from a ctest run log, emit "<name> <STATUS>" for non-passing outcomes
nonpass() {
  awk '/Test #/{
        name=""; for(i=1;i<=NF;i++) if($i=="Test"){name=$(i+2)}
        if(/Skipped/)                          print name" SKIPPED";
        else if(/Failed|Timeout|Exception/)    print name" FAILED";
      }' "$1" | sort
}

tag() { echo "[$(basename "$RUN_PRESET")] $*"; }

# build (failure here IS a gate failure) + run, capturing inventory + nonpass set
build_and_run() {
  local out="$1"; mkdir -p "$out"
  tag "building preset '$RUN_PRESET' ..."
  if ! cmake --build --preset "$RUN_PRESET" >"$out/build.log" 2>&1; then
    tag "BUILD FAILED -- see $out/build.log"; tail -20 "$out/build.log"; return 1
  fi
  local bdir="build/$RUN_PRESET"
  inventory "$bdir" >"$out/inventory-$RUN_PRESET.txt"
  tag "running ctest -L '$RUN_LABELS' ..."
  ctest --test-dir "$bdir" -L "$RUN_LABELS" --output-on-failure >"$out/run.log" 2>&1
  nonpass "$out/run.log" >"$out/nonpass-$RUN_PRESET.txt"
  # inventory every other configured build dir too (cheap, catches dropped targets)
  for d in $(detect_dirs); do
    local p; p="$(basename "$d")"; [ "$p" = "$RUN_PRESET" ] && continue
    inventory "$d" >"$out/inventory-$p.txt"
  done
  return 0
}

cmd_capture() {
  rm -rf "$STORE";
  build_and_run "$STORE" || { echo "baseline capture aborted (build failed)"; exit 1; }
  echo
  echo "Baseline captured in $STORE/ :"
  for f in "$STORE"/inventory-*.txt; do echo "  $(wc -l <"$f" | tr -d ' ') tests  $(basename "$f")"; done
  echo "  $(wc -l <"$STORE/nonpass-$RUN_PRESET.txt" | tr -d ' ') pre-existing non-pass (skip/fail) on $RUN_PRESET"
  echo "Commit nothing here -- $STORE/ is gitignored. Re-run with 'check' after each phase."
}

cmd_check() {
  [ -d "$STORE" ] || { echo "no baseline -- run 'capture-baseline' first"; exit 2; }
  local cur; cur="$(mktemp -d)"; local rc=0
  build_and_run "$cur" || rc=1   # build failure already fails the gate

  echo; echo "=== gate: equivalence vs baseline ==="

  # 1. inventory diff per preset + non-zero count
  for base in "$STORE"/inventory-*.txt; do
    local p; p="$(basename "$base" .txt)"; p="${p#inventory-}"
    local now="$cur/inventory-$p.txt"
    if [ ! -f "$now" ]; then echo "  [$p] MISSING now (build dir gone?)  <-- FAIL"; rc=1; continue; fi
    local nb nn; nb=$(wc -l <"$base"); nn=$(wc -l <"$now")
    if [ "$nn" -eq 0 ]; then echo "  [$p] 0 tests now ('green' with no tests is a lie)  <-- FAIL"; rc=1; fi
    if ! diff -q "$base" "$now" >/dev/null; then
      echo "  [$p] test set changed ($nb -> $nn)  <-- FAIL"
      diff "$base" "$now" | sed -n 's/^< /      removed: /p;s/^> /      added:   /p'
      rc=1
    else
      echo "  [$p] inventory unchanged ($nn tests)  OK"
    fi
  done

  # 2. no NEW skip/fail vs baseline (pre-existing ones are allowed)
  local base="$STORE/nonpass-$RUN_PRESET.txt" now="$cur/nonpass-$RUN_PRESET.txt"
  if [ -f "$now" ]; then
    local newbad; newbad="$(comm -13 "$base" "$now")"
    if [ -n "$newbad" ]; then
      echo "  [$RUN_PRESET] NEW skip/fail introduced  <-- FAIL"
      printf '      %s\n' $newbad | paste - - 2>/dev/null || echo "$newbad" | sed 's/^/      /'
      rc=1
    else
      echo "  [$RUN_PRESET] no new skip/fail (pre-existing: $(wc -l <"$base" | tr -d ' '))  OK"
    fi
  fi

  rm -rf "$cur"
  echo
  if [ "$rc" -eq 0 ]; then echo "GATE GREEN -- behavior preserved, safe to commit this phase."
  else echo "GATE RED -- do NOT commit; the diff above shows what the refactor broke."; fi
  exit "$rc"
}

case "${1:-}" in
  capture-baseline) cmd_capture ;;
  check)            cmd_check ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
