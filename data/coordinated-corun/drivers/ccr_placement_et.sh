#!/system/bin/sh
# Device-side thread-placement check for an ExecuTorch XNNPACK decode mask (task 3.1, the Pixel; placement only,
# not a measurement). As ccr_placement.sh, but the process is llama_main decoding 2000 tokens (XNNPACK 8da4w 1B,
# --cpu_threads <threads>) under taskset <mask>: 4 samples of every thread's state, last CPU and utime between
# t = 8 s and t = 28 s (after model load), and the per-core busy fraction over that interval.
# usage: ccr_placement_et.sh <mask> <threads>
mask=$1
n=$2
taskset "$mask" ./llama_main --model_path llama3_2-1b_xnnpack_8da4w.pte \
  --tokenizer_path /data/local/tmp/llama/tokenizer.model --prompt_file prompt_16.txt --max_new_tokens 2000 \
  --temperature 0 --ignore_eos --cpu_threads "$n" >/dev/null 2>&1 &
pid=$!
sleep 8
busy() { grep '^cpu[0-9]' /proc/stat | while read -r c us ni sy id io irq sirq rest; do
  echo "$c $((us + ni + sy + irq + sirq)) $((us + ni + sy + id + io + irq + sirq))"
done; }
busy >/data/local/tmp/ccr/.busy0
for k in 1 2 3 4; do
  echo "mask=$mask threads=$n sample=$k: tid state last_cpu utime_ticks"
  for t in /proc/$pid/task/*; do
    set -- $(cat "$t/stat" | sed 's/^.*) //')
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
