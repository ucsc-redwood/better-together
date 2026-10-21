#!/usr/bin/env bash
# Host-side driver: run every line of a plan file on one phone, for R rounds.
# usage: run_plan.sh <serial> <plan-file> <rounds> <out-dir>
# Plan line: <name>|<hex-mask>|<command run from /data/local/tmp/lpa>   (# = comment)
# Each run goes through lpa_run.sh (raw stdout/stderr, clock samples, thermal before and
# after). Runs are 20 s apart; before a run the phone must report thermal status 0. A run
# that ends with thermal status >= 1 is flagged, the phone cools down, and it is repeated
# once as <name>_rK_retry; so is a run that exits non-zero (a failure is retried once, and recorded
# if the retry fails too). Raw files land in <out-dir> as <name>_r<K>.{out,err,clk,meta}.
# Resumable: a run whose .meta in <out-dir> is complete (has exit_code) is not run again.
set -u
serial=$1 plan=$2 rounds=$3 outdir=$4
dev=/data/local/tmp/lpa
mkdir -p "$outdir"
log="$outdir/driver.log"

status() { adb -s "$serial" shell "dumpsys thermalservice | sed -n 's/^Thermal Status: //p'" </dev/null | tr -d '\r'; }

wait_cool() {
  local waited=0 s
  while :; do
    s=$(status)
    [ "$s" = "0" ] && break
    if [ "$waited" -ge 900 ]; then
      echo "$(date -Is) WARN thermal status still $s after ${waited}s, running anyway" >>"$log"
      break
    fi
    echo "$(date -Is) cooling: thermal status $s" >>"$log"
    sleep 30
    waited=$((waited + 30))
  done
}

complete() { grep -q '^exit_code=' "$outdir/$1.meta" 2>/dev/null; }

run_one() { # <tag> <mask> <cmd>
  local tag=$1 mask=$2 cmd=$3
  wait_cool
  echo "$(date -Is) start $tag mask=$mask cmd=$cmd" >>"$log"
  adb -s "$serial" shell "cd $dev && ./lpa_run.sh out/$tag $mask $cmd" </dev/null
  local rc=$?
  for ext in out err clk meta; do
    adb -s "$serial" pull "$dev/out/$tag.$ext" "$outdir/" </dev/null >/dev/null 2>&1
  done
  echo "$(date -Is) end $tag rc=$rc" >>"$log"
  sleep 20
}

for k in $(seq 1 "$rounds"); do
  while IFS='|' read -r name mask cmd; do
    case "$name" in '' | \#*) continue ;; esac
    complete "${name}_r$k" || run_one "${name}_r$k" "$mask" "$cmd"
    complete "${name}_r${k}_retry" && continue
    if ! grep -q '^exit_code=0$' "$outdir/${name}_r$k.meta"; then
      echo "$(date -Is) FAILED ${name}_r$k, repeating once" >>"$log"
      run_one "${name}_r${k}_retry" "$mask" "$cmd"
    elif sed -n '/^thermal_after:/,$p' "$outdir/${name}_r$k.meta" | grep -q 'Thermal Status: [1-9]'; then
      echo "$(date -Is) THROTTLED ${name}_r$k, cooling down and repeating once" >>"$log"
      sleep 120
      run_one "${name}_r${k}_retry" "$mask" "$cmd"
    fi
  done <"$plan"
done
echo "$(date -Is) plan done" >>"$log"
