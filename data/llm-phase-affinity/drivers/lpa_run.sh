#!/system/bin/sh
# Device-side wrapper for one measured run of the llm-phase-affinity study.
# usage: lpa_run.sh <out-prefix> <hex-mask> <cmd...>
# Writes <out-prefix>.out (stdout), .err (stderr), .clk (GPU clock + CPU policy
# frequencies every 0.5 s while the command runs) and .meta (mask, command, exit code,
# start/end epoch, thermal status and HAL temperatures before and after).
out=$1
mask=$2
shift 2

if [ -r /sys/kernel/gpu/gpu_clock ]; then
  gclk=/sys/kernel/gpu/gpu_clock # Samsung, MHz
else
  gclk=/sys/class/misc/mali0/device/cur_freq # Pixel, kHz
fi

thermal() {
  dumpsys thermalservice | sed -n '/^Thermal Status/p; /^Current temperatures from HAL/,/^Current cooling/p' |
    grep -E 'Thermal Status|Temperature\{' | sed 's/^[[:space:]]*//'
}

{
  echo "mask=$mask"
  echo "cmd=$*"
  echo "gpu_clock_path=$gclk"
  echo "start_epoch=$(date +%s.%N)"
  echo "thermal_before:"
  thermal
} >"$out.meta"

(
  while :; do
    f=""
    for p in /sys/devices/system/cpu/cpufreq/policy*; do f="$f ${p##*/}=$(cat $p/scaling_cur_freq)"; done
    echo "$(date +%s.%N) gpu=$(cat $gclk)$f"
    sleep 0.5
  done
) >"$out.clk" &
sampler=$!

taskset "$mask" "$@" >"$out.out" 2>"$out.err"
rc=$?

kill $sampler
{
  echo "end_epoch=$(date +%s.%N)"
  echo "exit_code=$rc"
  echo "thermal_after:"
  thermal
} >>"$out.meta"
exit $rc
