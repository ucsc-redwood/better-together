#!/usr/bin/env bash
set -euo pipefail

# ─── configure these lists ─────────────────────────────────────────────────────
devices=(3A021JEHN02756 9b034f1b)
apps=(cifar-sparse cifar-dense tree)
backends=(vk)
# ────────────────────────────────────────────────────────────────────────────────

for device in "${devices[@]}"; do
  echo "================================================"
  for app in "${apps[@]}"; do
    echo "--------------------------------"
    for backend in "${backends[@]}"; do
      for i in {5..30}; do
        echo -n "$device / $app / $backend / $i = "
        uv run scripts/collect/04_parse_schedules_by_widest_advanced.py -v \
            "data/exe_logs_isolated/${device}/${app}/${backend}" \
            --model "data/schedules-isolated/${device}/${app}/${backend}/schedules_normal.json" \
            --max-schedules $i \
          2>&1 | rg "Pearson correlation coefficient" | sed -E 's/.*: //'
      done
    done
  done
done

    # uv run scripts/collect/04_parse_schedules_by_widest_advanced.py -v \
    #     data/exe_logs_isolated/{{device}}/{{app}}/{{backend}} \
    #     --model data/schedules-isolated/{{device}}/{{app}}/{{backend}}/schedules_normal.json \
    #     -o plots-isolated/{{device}}/{{app}}/{{backend}} \
    #     --max-schedules 30

