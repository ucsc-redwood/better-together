#!/usr/bin/env bash
# lib-cell-marker.sh -- shared helpers for the run-on-*.sh deploy scripts to emit a
# machine-greppable per-cell coverage marker. Sourced (not executed).
#
# Why: ctest -L omp (the everyday gate) gives NO signal on the GPU backends, and the
# CUDA/Vulkan differential tests GTEST_SKIP off-fleet -- a never-run cell is exit-0,
# indistinguishable from a pass. These helpers turn each fleet run into an explicit
# RAN/SKIP/FAIL marker line so scripts/check_fleet_coverage.py can fail a silently-absent
# cell against fleet-coverage.json. See docs/reports-for-human/code-review-2026-06-18.md #8.
#
# Marker format (one line per binary):  BT-CELL <app> <backend> <hardware> <RAN|SKIP|FAIL>

# Map a test-target binary name -> "<app> <backend>", or empty if it is not a
# fleet-coverage differential cell (e.g. the e2e pipeline binaries, OMP tests).
# Recognized diff targets: test-<app>-<cu|vk>  (app in tree|cifar-dense|cifar-sparse).
bt_cell_of() {
  case "$1" in
    test-tree-cu) echo "tree cuda" ;;
    test-cifar-dense-cu) echo "cifar-dense cuda" ;;
    test-cifar-sparse-cu) echo "cifar-sparse cuda" ;;
    test-tree-vk) echo "tree vulkan" ;;
    test-cifar-dense-vk) echo "cifar-dense vulkan" ;;
    test-cifar-sparse-vk) echo "cifar-sparse vulkan" ;;
    *) echo "" ;; # not a tracked cell
  esac
}

# Classify a gtest run from (exit_code, captured_output) -> RAN|SKIP|FAIL.
#  - nonzero exit (or any [  FAILED  ] line)         -> FAIL
#  - at least one [       OK ] line (a case ran)     -> RAN
#  - else (only [  SKIPPED ], no OK)                 -> SKIP
bt_classify_run() {
  local rc="$1" out="$2"
  if [ "$rc" -ne 0 ] || printf '%s' "$out" | grep -q '\[  FAILED  \]'; then
    echo FAIL
  elif printf '%s' "$out" | grep -q '\[       OK \]'; then
    echo RAN
  else
    echo SKIP
  fi
}

# Emit the marker line for one binary. Args: target hardware exit_code output
# Prints "BT-CELL <app> <backend> <hardware> <status>" to stdout for tracked cells.
bt_emit_marker() {
  local target="$1" hardware="$2" rc="$3" out="$4"
  local cell
  cell="$(bt_cell_of "$target")"
  [ -n "$cell" ] || return 0
  echo "BT-CELL $cell $hardware $(bt_classify_run "$rc" "$out")"
}

# Post-process a full run-log into per-cell markers. The run loops in run-on-*.sh
# delimit each binary with a "== <target> ==" header; we slice on those and classify
# each binary's gtest section. Args: logfile hardware. Prints BT-CELL lines for the
# tracked diff targets present in the log (the overall run already reported pass/fail).
bt_emit_markers_from_log() {
  local log="$1" hardware="$2"
  awk -v hw="$hardware" '
    function tok(t){
      if(t=="test-tree-cu")         return "tree cuda";
      if(t=="test-cifar-dense-cu")  return "cifar-dense cuda";
      if(t=="test-cifar-sparse-cu") return "cifar-sparse cuda";
      if(t=="test-tree-vk")         return "tree vulkan";
      if(t=="test-cifar-dense-vk")  return "cifar-dense vulkan";
      if(t=="test-cifar-sparse-vk") return "cifar-sparse vulkan";
      return "";
    }
    function flush(){
      if(cur==""){return}
      c=tok(cur);
      if(c!=""){
        st = failed ? "FAIL" : (ok ? "RAN" : "SKIP");
        print "BT-CELL " c " " hw " " st;
      }
      ok=0; failed=0;
    }
    /^== .* ==$/ { flush(); cur=$2; next }
    /\[  FAILED  \]/ { failed=1 }
    /\[       OK \]/ { ok=1 }
    END { flush() }
  ' "$log"
}
