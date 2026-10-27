#!/system/bin/sh
# Device-side co-run of two pinned processes (llm-phase-affinity, design D3).
# usage: lpa_corun.sh <out-dir> <first: A|B> <maskA> <cmdA> <maskB> <cmdB>
# cmdA/cmdB are single strings. Both processes are launched back to back (the one named
# by <first> first) and run concurrently; each one's stderr is prefixed line by line with
# the epoch time it was read, so llama-bench --progress lines ("prompt run i/R",
# "generation run i/R") give each repetition's start and the overlapped interval can be
# computed afterwards. Writes <out-dir>/{A,B}.out (stdout), {A,B}.err (timestamped
# stderr), corun.clk (GPU clock + CPU policy frequencies every 0.5 s) and corun.meta.
out=$1
first=$2
mA=$3
cA=$4
mB=$5
cB=$6
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

stamp() {
  while IFS= read -r l; do echo "$(date +%s.%N) $l"; done
}

launch() { # <A|B> <mask> <cmd>
  echo "launch_$1_epoch=$(date +%s.%N)" >>"$out/corun.meta"
  (taskset "$2" $3 2>&1 >"$out/$1.out" | stamp >"$out/$1.err") &
}

{
  echo "maskA=$mA"
  echo "cmdA=$cA"
  echo "maskB=$mB"
  echo "cmdB=$cB"
  echo "first=$first"
  echo "gpu_clock_path=$gclk"
  echo "thermal_before:"
  thermal
} >"$out/corun.meta"

(
  while :; do
    f=""
    for p in /sys/devices/system/cpu/cpufreq/policy*; do f="$f ${p##*/}=$(cat $p/scaling_cur_freq)"; done
    echo "$(date +%s.%N) gpu=$(cat $gclk)$f"
    sleep 0.5
  done
) >"$out/corun.clk" &
sampler=$!

if [ "$first" = A ]; then
  launch A "$mA" "$cA"
  pa=$!
  launch B "$mB" "$cB"
  pb=$!
else
  launch B "$mB" "$cB"
  pb=$!
  launch A "$mA" "$cA"
  pa=$!
fi
wait $pa
echo "end_A_epoch=$(date +%s.%N)" >>"$out/corun.meta"
wait $pb
echo "end_B_epoch=$(date +%s.%N)" >>"$out/corun.meta"

kill $sampler
{
  echo "thermal_after:"
  thermal
} >>"$out/corun.meta"
