Backtest Module (9+3 months OOS)

Goal
- Verify the live strategy objectively on the last ~1 year of data
- Avoid leakage: 9 months tuning, 3 months pure OOS (with 1–2h embargo)

Structure
- collect_ohlcv.py: Fetch 1y OHLCV via ccxt and store Parquet per symbol/timeframe.
- validate_data.py: Ensure UTC ordering, drop duplicates, report gaps.
- sim.py: Simple bar-level simulator (entry on MA 5/20 alignment, ATR stop + trail).
- metrics.py: Sharpe, MaxDD, trades count, simple fee/slippage model.
- run_train_test.py: Split data (Train=[start:T0), Test=[T0:T0+90d)), tune on Train, evaluate once on Test.

Data Layout
- data/backtest/ohlcv/{timeframe}/{symbol}.parquet (UTC, columns: ts, open, high, low, close, volume)

Usage (example)
1) Collect 1-year OHLCV
   python backtest/collect_ohlcv.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 15m --days 365

2) Validate
   python backtest/validate_data.py --symbols BTC/USDT,ETH/USDT --timeframe 15m

3) Train+Test
   python backtest/run_train_test.py --symbols BTC/USDT,ETH/USDT --timeframe 15m \
     --fees_bps 5 --slip_bps 2 --oos_days 90

Notes
- No writes to live state.json; fully isolated from main.py runtime.
- Fees/slippage default to taker 5 bps per side and 2 bps slippage; adjust via CLI.
- Only bar-close decisions; intrabar stop-outs modeled using bar high/low crossing.

