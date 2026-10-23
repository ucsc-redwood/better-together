#!/usr/bin/env bash
# Host-side supervisor: runs run_queue.sh on each stage file in turn, and pauses for the battery (the Samsung drains
# under co-runs even on USB power; SIGSTOP has no effect in this session's sandbox, so pausing is stop-and-resume).
# When the battery is at or below <low>%, it waits until the running driver sleeps between runs (its log's last line
# is an "end", "THROTTLED" or "cooling" line), terminates the queue and that driver (SIGTERM to the PIDs it started
# and their children), waits until the battery is back at <high>%, and starts the same stage file again: run_plan.sh
# and run_corun.sh skip every complete run, so the stage resumes where it stopped. A run is never cut short.
# The battery is read every 60 s (every 3 s while waiting for a pause window). Actions go to supervise.log.
# usage: ccr_supervise.sh <serial> <low> <high> <stage-file>...
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
serial=$1 low=$2 high=$3
shift 3
log=$root/data/coordinated-corun/$serial/supervise.log
level() { adb -s "$serial" shell "dumpsys battery | sed -n 's/^  level: //p'" </dev/null | tr -d '\r'; }
say() { echo "$(date -Is) $*" >>"$log"; }

for stages in "$@"; do
  while :; do
    say "start run_queue.sh $stages"
    "$here/run_queue.sh" "$serial" "$stages" &
    q=$!
    paused=""
    while kill -0 "$q" 2>/dev/null; do
      b=$(level)
      if [ -n "$b" ] && [ "$b" -le "$low" ]; then
        # Wait for a window between runs, then stop the queue and its driver.
        while kill -0 "$q" 2>/dev/null; do
          d=$(pgrep -P "$q" | head -1)
          out=$(ps -o args= -p "$d" 2>/dev/null | awk '{print $NF}')
          last=$(tail -1 "$out/driver.log" 2>/dev/null)
          case "$last" in
            *" end "* | *THROTTLED* | *cooling:*)
              kids=$(pgrep -P "$d" | tr '\n' ' ')
              kill "$q" "$d" $kids 2>/dev/null
              say "battery $b%: stopped queue $q, driver $d ($kids) after: $last"
              paused=1
              break
              ;;
          esac
          sleep 3
        done
        break
      fi
      sleep 60
    done
    wait "$q" 2>/dev/null
    [ -z "$paused" ] && break # the stage file is done
    until b=$(level) && [ -n "$b" ] && [ "$b" -ge "$high" ]; do sleep 120; done
    say "battery $b%: resuming"
  done
done
say "all stage files done"
