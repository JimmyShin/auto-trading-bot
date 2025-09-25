# Auto Trading Bot

## 1. Project Overview
We are building an automated trading bot. Phase 1 focused on establishing a safety net by locking in baseline behaviour, regression tests, and continuous integration. The current codebase captures the strategy's decisions deterministically, verifies them against a baseline, and runs tests automatically on every push.

## 2. Baseline Dataset
`data/testnet/baseline.json` stores the candles used for analysis, the resulting entry/exit signals, stop-loss values, and any skip reasons. This file represents the expected behaviour of the core strategy and serves as the ground truth for regression testing. Rebuild it only after intentional strategy updates.

## 3. Pytest Regression Tests
Baseline parity is enforced with `pytest`. The primary test regenerates decisions using the current strategy and compares them to `baseline.json`. Run tests locally before committing:

```bash
python -m pytest -v
```

If the test fails, either the baseline is stale or the strategy logic has changed. Re-run the baseline generator after reviewing the changes.

## 4. GitHub Actions CI
The workflow `.github/workflows/pytest.yml` installs dependencies and executes `pytest` on every push or pull request. CI must pass before merges, ensuring the baseline and strategy stay in sync across contributors and environments.

## 5. Working Cycle
1. Modify strategy or related code.
2. Run `python -m pytest -v` locally.
3. If behaviour changes are intentional, regenerate `baseline.json` with the baseline generator.
4. Commit the code and updated baseline.
5. Push to trigger the GitHub Actions workflow.

## 6. Module Layout
- `main.py` wires together the exchange adapter, risk engine, and reporting loop.
- `strategy.py` keeps deterministic signal generation and baseline compatibility.
- `risk.py` owns position sizing, daily guards, structural reset tracking, and state persistence.
- `reporter.py` centralises trade, signal, and diagnostics logging.
- `exchange_api.py` wraps the Binance client so orchestration code depends on a narrow adapter.

## 7. JSON Configuration
`config.py` now reads an optional `config.json` (override the location with `BOT_CONFIG_JSON`). Keys present in the file take precedence over environment variables. Common fields are listed below:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `binance_key` / `binance_secret` | string | `""` | API credentials for Binance Futures. |
| `testnet` | bool | `true` | Use Binance testnet endpoints when `true`. |
| `safe_restart` | bool | `true` | Enforce protective stop placement on restart. |
| `quote` | string | `"USDT"` | Quote currency for universe symbols. |
| `universe` | list or comma string | preset list | Trading symbols monitored by the bot. |
| `risk_pct` | float | `0.025` | Fraction of equity risked per entry. |
| `leverage` | number | `10` | Leverage applied when sizing orders. |
| `enable_pyramiding` | bool | `true` | Allow additive scaling once profit thresholds hit. |
| `pyramid_levels` | array | `[[2.0, 0.3], [3.0, 0.2]]` | Each entry is `[R-multiple, add_ratio]`. |
| `atr_len` | int | `14` | ATR lookback for volatility calculations. |
| `atr_stop_k` / `atr_trail_k` | float | `1.2` / `2.5` | Multipliers for initial and trailing stops. |
| `daily_loss_limit` | float | `0.06` | Max daily drawdown before trades are skipped. |
| `funding_avoid_min` | int | `5` | Minutes before funding to block new entries. |
| `poll_sec` | int | `30` | Delay between main loop iterations. |
| `timeframe` / `lookback` | string, int | `"1h"`, `400` | Candle timeframe and history depth. |
| `state_file` | string | `"state.json"` | Path for persisted strategy state. |
| `data_dir` | string | `"data"` | Base directory for reports and logs. |
| `emergency_policy` | string | `"protect_only"` | Shutdown policy when exits are forced. |
| `emergency_min_pnl_to_close_pct` | float | `0.0` | Minimum PnL required to flatten during emergency. |
| `emergency_stop_fallback_pct` | float | `0.015` | Fallback stop distance when no trailing stop exists. |

Nested keys:

- `position_cap.multiple` (default `2.0`) scales the notional cap vs equity.
- `position_cap.min` / `position_cap.max` bound the cap in USD.

A starter template lives in `config.example.json`; copy it to `config.json` and fill in your credentials before running the bot.

## 8. Roadmap
- **Phase 2 – Refactoring:** Modularise the strategy and state management for easier iteration.
- **Phase 3 – Reporting Enhancements:** Extend analytics/exporter scripts for richer monitoring and historical insight.
- **Phase 4 – CI Improvements:** Add linting, type checks, and coverage gates to the pipeline.
- **Phase 5 – Fail-Safes:** Introduce runtime guards, alerting, and automated kill switches.

Phase 1 is complete; focus shifts to architecture, observability, and resilience in upcoming sprints.
