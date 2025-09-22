import argparse
from datetime import datetime, timezone, timedelta
import ccxt
import pandas as pd
import os, sys

# Ensure repo root on sys.path so we can import config/strategy/indicators
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import UNIVERSE, TF
from backtest.utils import out_path, ensure_utc, dedup_sort, check_gaps, timeframe_to_ms, to_parquet_or_csv, read_parquet_or_csv
from backtest.collect_ohlcv import fetch_ohlcv
from backtest.run_train_test import split_train_test, tune_on_train, evaluate


def collect(symbols, timeframe: str, days: int, base_dir: str):
    ex = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'defaultType': 'future' },
        'timeout': 30_000,
    })
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)
    for sym in symbols:
        print(f"Collect {sym} {timeframe} for {days}d...")
        df = fetch_ohlcv(ex, sym, timeframe, since_ms, until_ms)
        df = ensure_utc(dedup_sort(df))
        p = out_path(base_dir, timeframe, sym)
        actual = to_parquet_or_csv(df, p)
        print(f"Saved {len(df)} -> {actual}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--base_dir', default='data')
    ap.add_argument('--fees_bps', type=float, default=5.0)
    ap.add_argument('--slip_bps', type=float, default=2.0)
    ap.add_argument('--oos_days', type=int, default=90)
    args = ap.parse_args()

    symbols = UNIVERSE
    timeframe = TF
    print(f"Universe from config: {symbols} | TF={timeframe}")

    # 1) Collect
    collect(symbols, timeframe, args.days, args.base_dir)

    # 2) Validate
    for sym in symbols:
        p = out_path(args.base_dir, timeframe, sym)
        df = read_parquet_or_csv(p)
        report = check_gaps(df, timeframe)
        print(f"{sym} rows={report['rows']} gaps={report['gap_count']} max_gap_ms={report['max_gap_ms']}")

    # 3) Train/OOS
    for sym in symbols:
        p = out_path(args.base_dir, timeframe, sym)
        df = read_parquet_or_csv(p)
        train, test, t0, t_end = split_train_test(df, args.oos_days)
        tuned = tune_on_train(train, args.fees_bps, args.slip_bps)
        print(f"{sym} tuned: {tuned}")
        # lock params and evaluate OOS once
        import config as cfg
        prev_stop, prev_trail = cfg.ATR_STOP_K, cfg.ATR_TRAIL_K
        cfg.ATR_STOP_K, cfg.ATR_TRAIL_K = tuned['ATR_STOP_K'], tuned['ATR_TRAIL_K']
        try:
            oos = evaluate(test, args.fees_bps, args.slip_bps)
        finally:
            cfg.ATR_STOP_K, cfg.ATR_TRAIL_K = prev_stop, prev_trail
        ok = (oos['sharpe'] > 1.0) and (oos['maxdd_pct'] < 20.0) and (oos['trades'] >= 30)
        print(f"{sym} OOS: {oos} PASS={ok}")

if __name__ == '__main__':
    main()
