## Strategy Analytics Dashboard (Prometheus)

1. Import the dashboard: open Grafana, navigate to Dashboards → Import, upload or paste `grafana/dashboards/strategy_analytics.json`, and select the Prometheus data source (or keep the templated default).
2. Adjust metric names or labels: if your Prometheus uses different metric or label names, update the PromQL expressions and variable queries inside `grafana/dashboards/strategy_analytics.json` before importing so they match your environment.
3. Change the default time window: edit the `$window` constant variable in the dashboard JSON (under `templating.list`) or through Grafana’s variable editor to set your preferred default (currently `24h`).
4. Validate before importing: run `bash scripts/validate_grafana_json.sh` to confirm the JSON parses correctly—`jq` is used when available, otherwise Python’s `json` module is used as a fallback.
