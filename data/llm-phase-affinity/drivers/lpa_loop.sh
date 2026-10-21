#!/system/bin/sh
# Run a command again and again until <seconds> have passed (a new run starts only before the
# deadline); stdout and stderr of every run go to this script's stdout. Used for ExecuTorch
# co-runs, whose llama_main prints epoch timestamps (PyTorchObserver) for every generate().
# usage: lpa_loop.sh <seconds> <cmd...>
secs=$1
shift
end=$(($(date +%s) + secs))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "loop_start_epoch=$(date +%s.%N)"
  "$@" 2>&1
  echo "loop_exit=$?"
done
