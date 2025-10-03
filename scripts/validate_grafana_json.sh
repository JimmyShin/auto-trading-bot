#!/usr/bin/env bash
set -euo pipefail
FILE="grafana/dashboards/strategy_analytics.json"
if [ ! -f "$FILE" ]; then
  echo "ERROR: $FILE not found"; exit 1
fi
python - <<'PY'
import json,sys
with open("grafana/dashboards/strategy_analytics.json","r",encoding="utf-8") as f:
    json.load(f)
print("json_ok")
PY
echo "Import grafana/dashboards/strategy_analytics.json in Grafana UI → Dashboards → Import"
