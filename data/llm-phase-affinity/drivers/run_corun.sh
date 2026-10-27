#!/usr/bin/env bash
# Host-side driver for co-run trials (design D3): every line of a plan, <trials> times.
# usage: run_corun.sh <serial> <plan-file> <trials> <out-dir>
# Plan line: <name>|<maskA>|<cmdA>|<maskB>|<cmdB>   (# = comment; commands run from /data/local/tmp/lpa)
# Trial k launches A first when k is odd and B first when k is even (both orders), 20 s between
# trials, thermal status 0 required before each. A trial that ends with thermal status >= 1 is
# repeated once after a 120 s cooldown as <name>_t<k>_retry. Raw files: <out-dir>/<name>_t<k>/.
# Resumable: a trial whose corun.meta in <out-dir> is complete (has thermal_after) is not run again.
set -u
serial=$1 plan=$2 trials=$3 outdir=$4
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

complete() { grep -q '^thermal_after:' "$outdir/$1/corun.meta" 2>/dev/null; }

trial() { # <tag> <first> <maskA> <cmdA> <maskB> <cmdB>
  local tag=$1
  wait_cool
  echo "$(date -Is) start $tag first=$2" >>"$log"
  adb -s "$serial" shell "cd $dev && rm -rf corun/$tag && ./lpa_corun.sh corun/$tag $2 $3 '$4' $5 '$6'" </dev/null
  adb -s "$serial" pull "$dev/corun/$tag" "$outdir/" </dev/null >/dev/null 2>&1
  echo "$(date -Is) end $tag" >>"$log"
  sleep 20
}

for k in $(seq 1 "$trials"); do
  first=A
  [ $((k % 2)) -eq 0 ] && first=B
  while IFS='|' read -r name mA cA mB cB; do
    case "$name" in '' | \#*) continue ;; esac
    complete "${name}_t$k" || trial "${name}_t$k" "$first" "$mA" "$cA" "$mB" "$cB"
    complete "${name}_t${k}_retry" && continue
    if sed -n '/^thermal_after:/,$p' "$outdir/${name}_t$k/corun.meta" | grep -q 'Thermal Status: [1-9]'; then
      echo "$(date -Is) THROTTLED ${name}_t$k, cooling down and repeating once" >>"$log"
      sleep 120
      trial "${name}_t${k}_retry" "$first" "$mA" "$cA" "$mB" "$cB"
    fi
  done <"$plan"
done
echo "$(date -Is) plan done" >>"$log"
