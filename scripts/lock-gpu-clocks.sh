#!/usr/bin/env bash
# Lock GPU (and CPU) clocks to a fixed high state for profiling -- the gold-standard fix
# for the interference audit's #1 finding: the BTPM GPU column is corrupted by DVFS (the
# bg load boosts the GPU while isolated runs at a low clock, so the GPU measures FASTER
# under load -- impossible for contention). Locking the clock makes isolated and
# interference share an operating point, so the measured delta reflects real contention.
#
# Run with sudo ON THE TARGET before bm-prof, e.g. on rocky:  sudo scripts/lock-gpu-clocks.sh
# Restore the default governor afterwards:                    sudo scripts/lock-gpu-clocks.sh --restore
#
# Best-effort + platform-detecting (AMD iGPU / Tegra / Mali / Adreno). The bm-prof harness
# records the resulting clock in provenance (gpu_clock_mhz), so you can VERIFY it took.
# NOTE: needs root and writes to power/clock sysfs -- it does NOT run inside the binary.
set -uo pipefail

MODE=${1:-lock} # lock | --restore
want() { [ "$MODE" = "--restore" ] && echo "$2" || echo "$1"; }

# --- CPU governor (all cores) -----------------------------------------------
cpu_gov=$(want performance schedutil)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$g" ] && echo "$cpu_gov" >"$g" 2>/dev/null && echo "CPU $(dirname "$g"): $cpu_gov"
done

# --- AMD iGPU (RADV / amdgpu): power_dpm_force_performance_level -------------
amd_level=$(want high auto)
for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  [ -w "$f" ] && echo "$amd_level" >"$f" 2>/dev/null && echo "AMD $f: $amd_level"
done

# --- NVIDIA Jetson (Tegra): jetson_clocks pins GPU+EMC+CPU to max ------------
if command -v jetson_clocks >/dev/null 2>&1; then
  if [ "$MODE" = "--restore" ]; then
    jetson_clocks --restore 2>/dev/null && echo "jetson_clocks: restored"
  else jetson_clocks 2>/dev/null && echo "jetson_clocks: pinned to max"; fi
fi

# --- Mali / Adreno (devfreq): force the performance governor ----------------
gpu_gov=$(want performance simple_ondemand)
for d in /sys/class/devfreq/*gpu* /sys/class/devfreq/*.gpu /sys/devices/platform/*gpu*/devfreq/*; do
  [ -w "$d/governor" ] && echo "$gpu_gov" >"$d/governor" 2>/dev/null && echo "devfreq $d: $gpu_gov"
done
# Adreno (kgsl) also exposes a min clock / force-on:
for k in /sys/class/kgsl/kgsl-3d0/devfreq/governor; do
  [ -w "$k" ] && echo "$gpu_gov" >"$k" 2>/dev/null && echo "kgsl $k: $gpu_gov"
done

echo ">> done ($MODE). Verify with bm-prof's provenance.gpu_clock_mhz, or read the sysfs clock."
