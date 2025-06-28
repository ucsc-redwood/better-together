#!/usr/bin/env bash
set -euo pipefail

# ─── configure these lists ─────────────────────────────────────────────────────
devices=(3A021JEHN02756 9b034f1b)
apps=(cifar-sparse cifar-dense tree)
backends=(vk)
table_type=btpm
minimize_mode=gapness
# ────────────────────────────────────────────────────────────────────────────────

for device in "${devices[@]}"; do
  echo "================================================"
  for app in "${apps[@]}"; do
    echo "--------------------------------"
    for backend in "${backends[@]}"; do
      for i in {5..30}; do
        echo -n "$device / $app / $backend / $i = "
        uv run scripts/collect/parse_schedules.py -v \
            "data/exe_logs_${table_type}_${minimize_mode}/${device}/${app}/${backend}" \
            --schedule-file "data/schedules/${device}/${app}/${backend}/schedules_${table_type}_${minimize_mode}.json" \
            --max-schedules $i \
          2>&1 | rg "Pearson correlation coefficient" | sed -E 's/.*: //'
      done
    done
  done
done
