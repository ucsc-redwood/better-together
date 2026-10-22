#!/usr/bin/env bash
# Host-side battery guard (the phones drain under co-runs even on USB power). When the battery is at or below <low>%,
# pause the phone's running run_plan.sh / run_corun.sh (SIGSTOP to that driver's PID, a process this study started)
# but only between runs: while its log's last line is an "end" or "THROTTLED" line, i.e. it sleeps before the next
# run. Resume it (SIGCONT) when the battery is back at <high>%. The battery is read every 60 s (every 3 s only while
# waiting for a pause window), to keep the guard's own load on the phone small. Actions go to battery_guard.log.
# usage: battery_guard.sh <serial> <low> <high>
set -u
serial=$1 low=$2 high=$3
root=$(cd "$(dirname "$0")/../../.." && pwd)
log=$root/data/coordinated-corun/$serial/battery_guard.log
paused=""
level() { adb -s "$serial" shell "dumpsys battery | sed -n 's/^  level: //p'" </dev/null | tr -d '\r'; }
while :; do
  pid=$(ps -eo pid=,args= | awk -v s="$serial" '$2 == "bash" && ($3 ~ /run_corun\.sh$/ || $3 ~ /run_plan\.sh$/) && $4 == s {print $1}' | head -1)
  [ -n "${b:-}" ] && [ "$b" -le "$low" ] && [ -z "$paused" ] || b=$(level)
  if [ -n "$paused" ]; then
    if [ -n "$b" ] && [ "$b" -ge "$high" ]; then
      kill -CONT "$paused" && echo "$(date -Is) battery $b%: resumed $paused" >>"$log"
      paused=""
    fi
  elif [ -n "$pid" ] && [ -n "$b" ] && [ "$b" -le "$low" ]; then
    out=$(ps -o args= -p "$pid" | awk '{print $NF}')
    last=$(tail -1 "$out/driver.log" 2>/dev/null)
    case "$last" in
      *" end "* | *THROTTLED*)
        kill -STOP "$pid" && paused=$pid && echo "$(date -Is) battery $b%: paused $pid after: $last" >>"$log"
        ;;
    esac
  fi
  if [ -z "$paused" ] && [ -n "$b" ] && [ "$b" -le "$low" ]; then sleep 3; else sleep 60; fi
done
