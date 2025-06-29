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


for device in "${devices[@]}"; do
  echo "================================================"
  for app in "${apps[@]}"; do
    echo "--------------------------------"
    for backend in "${backends[@]}"; do
      for j in {5..{{num_schedules}}}; do
        i=$num_schedules
          echo -n "$device / $app / $backend / $i = "
          uv run scripts/collect/04_parse_schedules.py -v \
              "data/exe_logs_isolated_tmax/${device}/${app}/${backend}/schedule_run_{j}.log" \
              --schedule-file "data/schedules/${device}/${app}/${backend}/schedules_isolated_tmax.json" \
              --max-schedules $i \
              --time-window 0.25-0.5 \
              2>&1 | rg "Pearson correlation coefficient" | sed -E 's/.*: //'
      done
    done
  done
done
