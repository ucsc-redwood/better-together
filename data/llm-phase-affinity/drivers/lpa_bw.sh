#!/system/bin/sh
# Device-side read-bandwidth case (llm-phase-affinity, design D4).
# usage: lpa_bw.sh <out-dir> <cpu|gpu|both> <cpu-mask> <cpu-core-list> <seconds>
# CPU: cpu_readbw (one pinned thread per listed core, 256 MiB total) prints "<epoch> <GB/s>"
# every 0.5 s. GPU: the owner's igpu-roofline runner (copied to gpubw/, with
# gpubw_config.json.tmpl as gpubw/config.json; mem_read_v4 over a
# 256 MiB buffer, host thread on core 3) prints one JSON sample per dispatch batch; each stdout
# line is prefixed with the epoch it was read. Both: the two at once.
out=$1
mode=$2
cmask=$3
cores=$4
secs=$5
mkdir -p "$out"

if [ -r /sys/kernel/gpu/gpu_clock ]; then
  gclk=/sys/kernel/gpu/gpu_clock
else
  gclk=/sys/class/misc/mali0/device/cur_freq
fi

thermal() {
  dumpsys thermalservice | sed -n '/^Thermal Status/p; /^Current temperatures from HAL/,/^Current cooling/p' |
    grep -E 'Thermal Status|Temperature\{' | sed 's/^[[:space:]]*//'
}

{
  echo "mode=$mode cpu_mask=$cmask cpu_cores=$cores seconds=$secs"
  echo "thermal_before:"
  thermal
} >"$out/bw.meta"

(
  while :; do
    f=""
    for p in /sys/devices/system/cpu/cpufreq/policy*; do f="$f ${p##*/}=$(cat $p/scaling_cur_freq)"; done
    echo "$(date +%s.%N) gpu=$(cat $gclk)$f"
    sleep 0.5
  done
) >"$out/bw.clk" &
sampler=$!

sed "s/DURATION/$secs/" gpubw/config.json >"$out/gpu.config.json"
pids=""
if [ "$mode" != cpu ]; then
  (cd gpubw && taskset 8 ./roofline "../$out/gpu.config.json" 2>"../$out/gpu.err" |
    taskset 8 sh -c 'while IFS= read -r l; do echo "$(date +%s.%N) $l"; done' >"../$out/gpu.out") &
  pids="$pids $!"
fi
if [ "$mode" != gpu ]; then
  taskset "$cmask" ./cpu_readbw "$cores" 256 "$secs" >"$out/cpu.out" 2>"$out/cpu.err" &
  pids="$pids $!"
fi
for p in $pids; do wait $p; done

kill $sampler
{
  echo "thermal_after:"
  thermal
} >>"$out/bw.meta"
