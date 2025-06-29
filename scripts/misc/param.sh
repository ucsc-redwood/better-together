#!/usr/bin/env bash
set -euo pipefail

# ─── configure these lists ─────────────────────────────────────────────────────
devices=(3A021JEHN02756 9b034f1b)
apps=(cifar-sparse cifar-dense tree)
backends=(vk)
# table_type=btpm
# minimize_mode=gapness
# table_type=isolated
# minimize_mode=tmax
num_schedules=25
# ────────────────────────────────────────────────────────────────────────────────

echo "================================================"
echo "now displaying isolated tmax"
echo "================================================"


for device in "${devices[@]}"; do
  echo "================================================"
  for app in "${apps[@]}"; do
    echo "--------------------------------"
    for backend in "${backends[@]}"; do
      # for i in {5..{{num_schedules}}}; do
      i=$num_schedules
        echo -n "$device / $app / $backend / $i = "
        uv run scripts/collect/04_parse_schedules.py -v \
            "data/exe_logs_isolated_tmax/${device}/${app}/${backend}" \
            --schedule-file "data/schedules/${device}/${app}/${backend}/schedules_isolated_tmax.json" \
            --max-schedules $i \
            --time-window 0.25-0.5 \
            2>&1 | rg "Pearson correlation coefficient" | sed -E 's/.*: //'
            # 2>&1 | rg "Coefficient of determination" | sed -E 's/.*: //' 
      # done
    done
  done
done

echo "================================================"
echo "now displaying btpm gapness"
echo "================================================"

for device in "${devices[@]}"; do
  echo "================================================"
  for app in "${apps[@]}"; do
    echo "--------------------------------"
    for backend in "${backends[@]}"; do
      # for i in {5..{{num_schedules}}}; do
      i=$num_schedules
        echo -n "$device / $app / $backend / $i = "
        uv run scripts/collect/04_parse_schedules.py -v \
            "data/exe_logs_btpm_gapness/${device}/${app}/${backend}" \
            --schedule-file "data/schedules/${device}/${app}/${backend}/schedules_btpm_gapness.json" \
            --max-schedules $i \
            2>&1 | rg "Pearson correlation coefficient" | sed -E 's/.*: //'
            # 2>&1 | rg "Coefficient of determination" | sed -E 's/.*: //' 
      # done
    done
  done
done
