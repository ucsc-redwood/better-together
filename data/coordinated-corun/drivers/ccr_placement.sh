#!/system/bin/sh
# Device-side thread-placement check for a CPU decode mask (task 1.4; placement only, not a
# measurement). Runs lpa_pace's CPU decode under taskset <mask> with <threads> threads for 25 s and,
# between t = 5 s and t = 20 s, takes 4 samples of every thread's state, last CPU and utime
# (/proc/<pid>/task/*/stat fields 3, 39, 14) and the per-core busy fraction from /proc/stat.
# usage: ccr_placement.sh <mask> <threads> <gguf>
mask=$1
n=$2
model=$3
taskset "$mask" ./lpa_pace -m "$model" -ngl 0 -dev none -t "$n" --mode decode -P 0 -C 512 -u 1 -s 25 \
  >/dev/null 2>&1 &
pid=$!
sleep 5
busy() { grep '^cpu[0-9]' /proc/stat | while read -r c us ni sy id io irq sirq rest; do
  echo "$c $((us + ni + sy + irq + sirq)) $((us + ni + sy + id + io + irq + sirq))"
done; }
busy >/data/local/tmp/ccr/.busy0
for k in 1 2 3 4; do
  echo "mask=$mask threads=$n sample=$k: tid state last_cpu utime_ticks"
  for t in /proc/$pid/task/*; do
    set -- $(cat "$t/stat" | sed 's/^.*) //')
    # after "pid (comm) " the fields shift by 2: state=$1, utime=$12, processor=$37
    echo "  ${t##*/} $1 $(eval echo \${37}) $(eval echo \${12})"
  done
  sleep 5
done
busy >/data/local/tmp/ccr/.busy1
echo "per-core busy fraction over the sampled 20 s:"
paste /data/local/tmp/ccr/.busy0 /data/local/tmp/ccr/.busy1 | while read -r c b0 t0 c1 b1 t1; do
  echo "  $c $(((b1 - b0) * 100 / (t1 - t0)))%"
done
rm -f /data/local/tmp/ccr/.busy0 /data/local/tmp/ccr/.busy1
wait $pid
