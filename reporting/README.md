# Reporting Utilities

This directory contains batch analytics helpers that generate trade performance artifacts and gate deployments.

## Local Usage

1. Prepare your virtualenv and install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. Generate reports from raw trade CSV logs:
   ```bash
   python -m reporting.report_metrics --env testnet --out-dir reports/ --min-trades 30
   ```
   Optional flags:
   - `--csv-glob` to override the default `data/<env>/trades_*.csv` pattern.
   - `--windows` to supply a comma-separated list of rolling window sizes (defaults to `10,30,100`).
3. Run the Good-To-Go gate against the latest summary:
   ```bash
   python -m reporting.good_to_go_check --report reports/testnet/latest-summary.json --config reporting/good_to_go_config.yaml
   ```

## GitHub Actions Integration

Example nightly job that generates reports and uploads artifacts:
```yaml
jobs:
  trade-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m reporting.report_metrics --env testnet --out-dir reports/ --min-trades 30
      - uses: actions/upload-artifact@v4
        with:
          name: trade-reports
          path: reports/testnet/
```

Example PR gate that blocks merges when trade risk drifts:
```yaml
jobs:
  good-to-go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m reporting.report_metrics --env testnet --out-dir reports/ --min-trades 30
      - run: python -m reporting.good_to_go_check --report reports/testnet/latest-summary.json --config reporting/good_to_go_config.yaml
```

Both jobs rely on the CSV fixtures under `data/<env>` or uploaded artifacts from backtests/loggers. Ensure the CI job has access to the raw trade CSVs (artifact download step not shown).
