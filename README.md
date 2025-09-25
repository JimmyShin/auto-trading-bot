# Auto Trading Bot

Modular, test‑driven crypto trading bot with risk management, deterministic signals, rich reporting, and CI/CD + Docker.

## Overview

- Project goal: build a dependable, modular trading bot protected by tests and reproducible baselines.
- Key features:
  - Strategy module producing deterministic signals for regression testing.
  - Risk management for sizing, stops, pyramiding, structural resets, and daily loss limits.
  - Exchange API wrapper to isolate third‑party dependencies.
  - Reporter for logging trades/signals and generating analytics (DataFrame + Excel).
  - CI/CD with pytest on push/PR, Docker build/publish, and optional deploy.

## Architecture

### strategy.py

- Engine: `class DonchianATREngine(persist_state: bool = True, initial_state: dict | None = None, risk_manager: RiskManager | None = None)`
- Responsibilities: deterministic signal generation, entry planning, state persistence, daily risk anchors.
- Key methods:
  - `reset_daily_anchor(equity: float) -> None`
  - `detect_signal(df: pd.DataFrame) -> dict`
  - `make_entry_plan(symbol: str, df: pd.DataFrame, equity: float, price: float, atr_abs: float, is_new_bar: bool, can_reenter: bool, funding_avoid: bool, daily_loss_hit: bool) -> dict`
- Example usage:
```python
from strategy import DonchianATREngine
from indicators import atr

engine = DonchianATREngine(persist_state=False, initial_state={})
engine.reset_daily_anchor(20_000)
atr_abs = float(atr(df, 14))
sig = engine.detect_signal(df)
plan = engine.make_entry_plan(
    symbol="BTC/USDT", df=df, equity=20_000, price=float(df.close.iloc[-1]), atr_abs=atr_abs,
    is_new_bar=True, can_reenter=True, funding_avoid=False, daily_loss_hit=False,
)
```

### risk.py

- Manager: `class RiskManager(state: dict | None = None)`
- Responsibilities: daily anchors and loss limit; ATR‑based sizing; trailing stops; pyramiding; structural reset checks; broker sync.
- Key methods:
  - `calc_qty_by_risk(equity_usdt, price, atr_abs, leverage, symbol="") -> float`
  - `calc_qty_by_risk_adjusted(..., risk_pct: float | None = None) -> float`
  - `trail_stop_price(side, entry_price, last_price, atr_abs, be_promoted) -> float`
  - `update_symbol_state_on_entry(symbol, side, entry_px, qty=0) -> dict`
  - `update_after_move(symbol, atr_abs, last_price) -> dict | None`
  - `reset_daily_anchor(equity) -> bool` and `hit_daily_loss_limit(equity) -> bool`

### exchange_api.py

- Adapter: `class ExchangeAPI(client: Optional[BinanceUSDM] = None, auto_connect: bool = False)`
- Responsibilities: thin wrapper around the Binance Futures client; market data, account state, orders.
- Example:
```python
from exchange_api import ExchangeAPI
ex = ExchangeAPI(auto_connect=True)
ohlcv = ex.fetch_ohlcv("BTC/USDT", "1h", limit=200)
equity = ex.get_equity_usdt()
```

### reporter.py

- Logger: `class Reporter(base_dir: str, environment: str)` with `Reporter.from_config()`.
- CSV logging: `log_trade`, `log_exit`, `log_signal_analysis`, `log_filtered_signal`, `log_detailed_entry` (files under `data/<env>/`).
- Analytics:
  - `generate_report(trades) -> pd.DataFrame` returns a single‑row DataFrame of metrics (see Reporting).
  - `save_report(df, filepath)` writes Excel with `index=False`.
- Example:
```python
from reporter import generate_report, save_report
rep_df = generate_report(trades_df)
save_report(rep_df, "reports/performance.xlsx")
```

## Configuration (config.json)

`config.py` reads an optional `config.json` (override with env `BOT_CONFIG_JSON`). JSON keys override environment variables.

Common keys:
```json
{
  "binance_key": "...",
  "binance_secret": "...",
  "testnet": true,
  "safe_restart": true,
  "quote": "USDT",
  "universe": ["BTC/USDT", "ETH/USDT"],
  "risk_pct": 0.025,
  "leverage": 10,
  "enable_pyramiding": true,
  "pyramid_levels": [[2.0, 0.3], [3.0, 0.2]],
  "atr_len": 14,
  "atr_stop_k": 1.2,
  "atr_trail_k": 2.5,
  "daily_loss_limit": 0.06,
  "funding_avoid_min": 5,
  "poll_sec": 30,
  "timeframe": "1h",
  "lookback": 400,
  "state_file": "state.json",
  "data_dir": "data",
  "emergency_policy": "protect_only",
  "emergency_min_pnl_to_close_pct": 0.0,
  "emergency_stop_fallback_pct": 0.015,
  "position_cap": {"multiple": 2.0, "min": 10000, "max": 25000}
}
```

## Safety & Quality

### Baseline.json regression testing

- `data/testnet/baseline.json` stores the candles, expected signals (decisions), stops and reasons. It is the ground truth.
- Deterministic generator (baseline.py) rebuilds expected outputs from offline OHLCV snapshots.

### Pytest suite

- Signals equality: regenerated symbols’ records must match baseline exactly.
- Metrics comparison: Win Rate, Expectancy, MDD, Profit Factor, Sharpe must match baseline metrics within tolerance (rel/abs = 1e‑2).
- Relevant tests:
  - `tests/test_regression.py` (signals exact + metrics approx)
  - `test_baseline_regression.py` (baseline structure)
  - `test_report_metrics.py` (unit validation of metrics)

Run locally:
```bash
python -m pytest -v
```

### GitHub Actions

- `.github/workflows/pytest.yml` runs tests on `push` and `pull_request`. Failures block PR merges.
- `.github/workflows/docker-publish.yml` builds/pushes Docker images to GHCR and can deploy via SSH on main.

## Reporting

### Core metrics

- Trade count = number of closed trades
- Win rate = wins / total
- RRR (payoff ratio) = avg(win) / abs(avg(loss))
- Daily/weekly/monthly change = aggregate returns over those periods
  - Daily report implemented via `scripts/performance_report.py`
  - Weekly/monthly can be added by grouping returns by calendar week/month

### Extended metrics (generate_report)

- Expectancy = (win_rate*avg(win)) − (loss_rate*avg(loss))
- MDD = max peak‑to‑trough drawdown / peak (ratio)
- Profit Factor = gross profit / gross loss
- Sharpe Ratio = mean(returns) / std(returns) (no annualisation)
- Win/Loss streaks = max consecutive wins/losses
- Average holding time = mean(exit_ts − entry_ts) in seconds
- Long/Short performance = per‑side trades, win rate, expectancy, profit factor

### Output formats

- Pandas DataFrame: `generate_report(...) -> pd.DataFrame`
- Excel export: `save_report(df, path)` with `index=False`
- CSV summaries:
  - `scripts/performance_report.py` (by day / by symbol / by day+symbol)
  - `scripts/daily_report.py` (daily CSV from trade logs)
  - `scripts/refresh_baseline_metrics.py [--regen]` (embed `report_metrics` into baseline)

## Operations

### Simulation / Backtest

- Offline baseline decisions: `python baseline.py --symbols BTC/USDT ETH/USDT --timeframe 1h --bars 180 --output data/testnet/baseline.json`
- Train/test helpers: see `backtest/` (`sim.py`, `run_train_test.py`, `run_from_config.py`, `metrics.py`).
- Ensure `data/backtest/ohlcv/` exists (offline candles) — it’s ignored by git but required for baseline regeneration.

#### Backtesting usage

- Quick per-symbol simulation (using offline OHLCV in parquet/csv):
  ```bash
  python -m backtest.sim --parquet data/backtest/ohlcv/1h/BTC_USDT.parquet
  # or CSV if parquet engine unavailable
  python -m backtest.sim --parquet data/backtest/ohlcv/1h/BTC_USDT.csv
  ```

- Run end-to-end from config (collect, validate, tune on train, evaluate OOS):
  ```bash
  python -m backtest.run_from_config --days 365 --oos_days 90 --base_dir data/backtest/ohlcv
  ```

Tests include checks that simulated backtest metrics align with baseline-derived metrics.

### Live trading (testnet/live)

- Configure credentials and parameters in `config.json` or environment variables.
- Run: `python main.py`

### Developer quickstart

```bash
python -m pytest -v
python baseline.py --symbols BTC/USDT ETH/USDT --timeframe 1h --bars 180 --output data/testnet/baseline.json
python scripts/refresh_baseline_metrics.py --regen
python main.py
```

### Docker

Build/run locally:
```bash
docker build -t auto-trading-bot:local .
docker run --rm -e BINANCE_KEY=... -e BINANCE_SECRET=... -e TESTNET=true \
  -v $(pwd)/data:/app/data auto-trading-bot:local
```

GHCR publishing (on `main`) and optional remote deploy are configured in `.github/workflows/docker-publish.yml`.

### Tools (MCP)

- Extra utilities under `tools/` for MCP integration:
  - `tools/mcp_server.py` provides bot inspection and safe control endpoints.
  - See `tools/MCP_SETUP.md` for setup.

Run locally:
```bash
python tools/mcp_server.py
```

## Future Extensions

- Strategy diversification (multi‑system, regime switching)
- Advanced risk models (Kelly fraction, Monte Carlo bands)
- Reporting to Slack/Telegram and dashboards (Grafana/Prometheus)
- Scaling with cloud/VPS (Docker Compose, k8s, horizontal workers)

## Instructions for Contributors

### Add a new strategy

- Implement another engine class alongside `DonchianATREngine` exposing:
  - `detect_signal(df) -> dict`
  - `make_entry_plan(...) -> dict`
- Keep outputs deterministic and serialisable so they can be captured in the baseline.

### Update baseline.json

1. Regenerate with intentional logic changes:
   ```bash
   python baseline.py --symbols BTC/USDT ETH/USDT --timeframe 1h --bars 180 --output data/testnet/baseline.json
   ```
2. Embed report metrics to be used by regression tests:
   ```bash
   python scripts/refresh_baseline_metrics.py --regen
   ```
3. Run tests and commit both code and updated baseline.

### Verify with pytest + CI

```bash
python -m pytest -v
```

Open a PR; the GitHub Actions test workflow runs on push/PR and must pass before merge.
