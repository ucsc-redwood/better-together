#!/usr/bin/env bash
# Host-side driver for task 3.1: read bandwidth alone and together, <trials> rounds of cpu, gpu, both.
# usage: run_bw.sh <serial> <cpu-mask> <cpu-core-list> <seconds> <trials> <out-dir>
# Same discipline as run_plan.sh: thermal status 0 before each case, 20 s between cases, a case
# that ends with thermal status >= 1 repeated once after a 120 s cooldown as <case>_t<k>_retry.
set -u
serial=$1 cmask=$2 cores=$3 secs=$4 trials=$5 outdir=$6
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

one() { # <tag> <mode>
  wait_cool
  echo "$(date -Is) start $1" >>"$log"
  adb -s "$serial" shell "cd $dev && rm -rf bw/$1 && ./lpa_bw.sh bw/$1 $2 $cmask $cores $secs" </dev/null
  adb -s "$serial" pull "$dev/bw/$1" "$outdir/" </dev/null >/dev/null 2>&1
  echo "$(date -Is) end $1" >>"$log"
  sleep 20
}

for k in $(seq 1 "$trials"); do
  for mode in cpu gpu both; do
    one "${mode}_t$k" "$mode"
    if sed -n '/^thermal_after:/,$p' "$outdir/${mode}_t$k/bw.meta" | grep -q 'Thermal Status: [1-9]'; then
      echo "$(date -Is) THROTTLED ${mode}_t$k, cooling down and repeating once" >>"$log"
      sleep 120
      one "${mode}_t${k}_retry" "$mode"
    fi
  done
done
echo "$(date -Is) plan done" >>"$log"
