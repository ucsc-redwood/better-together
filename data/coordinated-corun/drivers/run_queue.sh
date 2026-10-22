#!/usr/bin/env bash
# Host-side queue: run stages one after the other on one phone and commit each stage's raw data when it
# ends, so an interruption loses at most the stage in flight (run_plan.sh / run_corun.sh resume it).
# usage: run_queue.sh <serial> <stage-file>
# Stage line: alone|<plan>|<rounds>|<out-subdir>|<commit subject>   (run_plan.sh)
#             corun|<plan>|<trials>|<out-subdir>|<commit subject>   (run_corun.sh)
#             push|<file under the repo>                             (adb push to /data/local/tmp/ccr)
# Commits go to whatever repository GIT_DIR / GIT_WORK_TREE name (see design.md).
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
serial=$1 stages=$2
log=$root/data/coordinated-corun/$serial/queue.log
mkdir -p "$(dirname "$log")"
while IFS='|' read -r kind plan n sub subject; do
  case "$kind" in '' | \#*) continue ;; esac
  echo "$(date -Is) stage $kind $plan $n $sub" >>"$log"
  case "$kind" in
    push)
      adb -s "$serial" push "$root/$plan" /data/local/tmp/ccr/ </dev/null >>"$log" 2>&1
      continue
      ;;
    alone) "$here/run_plan.sh" "$serial" "$here/$plan" "$n" "$root/data/coordinated-corun/$serial/$sub" ;;
    corun) "$here/run_corun.sh" "$serial" "$here/$plan" "$n" "$root/data/coordinated-corun/$serial/$sub" ;;
  esac
  (
    cd "$root" &&
      git add -f "data/coordinated-corun/$serial/$sub" "$log" &&
      git commit -q -m "$subject" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014XEYgCRDc2cZXgcQzhRNnM"
  ) >>"$log" 2>&1
  echo "$(date -Is) stage done $sub (commit rc=$?)" >>"$log"
done <"$stages"
echo "$(date -Is) queue done" >>"$log"
