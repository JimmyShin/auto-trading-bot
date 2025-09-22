import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import os, sys

# Ensure repo root on sys.path for config/strategy/indicators imports inside submodules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.utils import out_path, read_parquet_or_csv
from backtest.sim import simulate_symbol, SimParams
from backtest.metrics import equity_curve, sharpe, max_drawdown


def split_train_test(df: pd.DataFrame, oos_days: int = 90):
    df = df.sort_values('ts').reset_index(drop=True)
    t_end = df['ts'].iloc[-1]
    t0 = t_end - pd.Timedelta(days=oos_days)
    train = df[df['ts'] < t0].copy()
    test = df[(df['ts'] >= t0) & (df['ts'] < t_end)].copy()
    return train, test, t0, t_end


def split_train_test_with_embargo(df: pd.DataFrame, oos_days: int = 90, embargo_hours: int = 0):
    df = df.sort_values('ts').reset_index(drop=True)
    t_end = df['ts'].iloc[-1]
    t0 = t_end - pd.Timedelta(days=oos_days)
    t0_test = t0 + pd.Timedelta(hours=embargo_hours)
    train = df[df['ts'] < t0].copy()
    test = df[(df['ts'] >= t0_test) & (df['ts'] < t_end)].copy()
    return train, test, t0, t_end


def evaluate(df: pd.DataFrame, fees_bps: float, slip_bps: float, symbol: str = "SIM"):
    trades = simulate_symbol(df, SimParams(fees_bps=fees_bps, slip_bps=slip_bps), symbol=symbol)
    curve = equity_curve(trades, fees_bps, slip_bps)
    if curve.empty:
        return {
            'trades': 0,
            'sharpe': 0.0,
            'maxdd_pct': 0.0,
        }
    ret = curve['equity'].pct_change().dropna()
    shp = sharpe(ret, periods_per_year=365*24*4)
    mdd = max_drawdown(curve['equity'])
    return {
        'trades': len(trades),
        'sharpe': shp,
        'maxdd_pct': mdd,
    }


def tune_on_train(train: pd.DataFrame, fees_bps: float, slip_bps: float):
    # Minimal grid for ATR multipliers — extend as needed
    from config import ATR_STOP_K, ATR_TRAIL_K
    grid = [
        (1.0, 2.0), (1.0, 2.5), (1.2, 2.0), (1.2, 2.5), (1.5, 3.0)
    ]
    best = None
    for stop_k, trail_k in grid:
        # temporarily monkey-patch config values in local namespace
        import config as cfg
        prev_stop, prev_trail = cfg.ATR_STOP_K, cfg.ATR_TRAIL_K
        cfg.ATR_STOP_K, cfg.ATR_TRAIL_K = stop_k, trail_k
        try:
            res = evaluate(train, fees_bps, slip_bps)
        finally:
            cfg.ATR_STOP_K, cfg.ATR_TRAIL_K = prev_stop, prev_trail
        score = res['sharpe'] - max(0, (res['maxdd_pct'] - 20.0)/20.0)  # soft constraint on DD (maxdd_pct is +%)
        if not best or score > best[0]:
            best = (score, stop_k, trail_k, res)
    return {
        'ATR_STOP_K': best[1],
        'ATR_TRAIL_K': best[2],
        'train_metrics': best[3]
    }


def tune_on_train_expanded(train: pd.DataFrame, fees_bps: float, slip_bps: float):
    # Expanded grid across stop, trail, risk, and candle-position thresholds
    stop_grid = [1.0, 1.2]
    trail_grid = [2.0, 2.5, 3.0]
    risk_grid = [0.02, 0.025]
    long_pos_grid = [0.65, 0.75]
    short_pos_grid = [0.25, 0.35]
    best = None
    import config as cfg
    for stop_k in stop_grid:
        for trail_k in trail_grid:
            for risk_pct in risk_grid:
                for long_pos in long_pos_grid:
                    for short_pos in short_pos_grid:
                        prev = (cfg.ATR_STOP_K, cfg.ATR_TRAIL_K, cfg.RISK_PCT,
                                cfg.CANDLE_LONG_MAX_POS_RATIO, cfg.CANDLE_SHORT_MIN_POS_RATIO)
                        cfg.ATR_STOP_K = stop_k
                        cfg.ATR_TRAIL_K = trail_k
                        cfg.RISK_PCT = risk_pct
                        cfg.CANDLE_LONG_MAX_POS_RATIO = long_pos
                        cfg.CANDLE_SHORT_MIN_POS_RATIO = short_pos
                        try:
                            res = evaluate(train, fees_bps, slip_bps)
                        finally:
                            (cfg.ATR_STOP_K, cfg.ATR_TRAIL_K, cfg.RISK_PCT,
                             cfg.CANDLE_LONG_MAX_POS_RATIO, cfg.CANDLE_SHORT_MIN_POS_RATIO) = prev
                        score = res['sharpe'] - max(0, (res['maxdd_pct'] - 20.0)/20.0)
                        if not best or score > best[0]:
                            best = (score, stop_k, trail_k, risk_pct, long_pos, short_pos, res)
    return {
        'ATR_STOP_K': best[1],
        'ATR_TRAIL_K': best[2],
        'RISK_PCT': best[3],
        'CANDLE_LONG_MAX_POS_RATIO': best[4],
        'CANDLE_SHORT_MIN_POS_RATIO': best[5],
        'train_metrics': best[6]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True)
    ap.add_argument('--timeframe', default='15m')
    ap.add_argument('--base_dir', default='data')
    ap.add_argument('--fees_bps', type=float, default=5.0)
    ap.add_argument('--slip_bps', type=float, default=2.0)
    ap.add_argument('--oos_days', type=int, default=90)
    ap.add_argument('--embargo_hours', type=int, default=2)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    all_reports = []
    for sym in symbols:
        p = out_path(args.base_dir, args.timeframe, sym)
        df = read_parquet_or_csv(p)
        train, test, t0, t_end = split_train_test_with_embargo(df, args.oos_days, args.embargo_hours)
        print(f"{sym}: Train={train['ts'].iloc[0]}..{train['ts'].iloc[-1]} | Test={test['ts'].iloc[0]}..{test['ts'].iloc[-1]} (embargo {args.embargo_hours}h)")

        tuned = tune_on_train_expanded(train, args.fees_bps, args.slip_bps)
        print(f"{sym}: tuned stop={tuned['ATR_STOP_K']} trail={tuned['ATR_TRAIL_K']} risk={tuned['RISK_PCT']} CL={tuned['CANDLE_LONG_MAX_POS_RATIO']} CS={tuned['CANDLE_SHORT_MIN_POS_RATIO']} train={tuned['train_metrics']}")

        # lock params and evaluate once on OOS
        import config as cfg
        prev_vals = (cfg.ATR_STOP_K, cfg.ATR_TRAIL_K, cfg.RISK_PCT,
                     cfg.CANDLE_LONG_MAX_POS_RATIO, cfg.CANDLE_SHORT_MIN_POS_RATIO)
        cfg.ATR_STOP_K = tuned['ATR_STOP_K']
        cfg.ATR_TRAIL_K = tuned['ATR_TRAIL_K']
        cfg.RISK_PCT = tuned['RISK_PCT']
        cfg.CANDLE_LONG_MAX_POS_RATIO = tuned['CANDLE_LONG_MAX_POS_RATIO']
        cfg.CANDLE_SHORT_MIN_POS_RATIO = tuned['CANDLE_SHORT_MIN_POS_RATIO']
        try:
            oos = evaluate(test, args.fees_bps, args.slip_bps, symbol=sym)
        finally:
            (cfg.ATR_STOP_K, cfg.ATR_TRAIL_K, cfg.RISK_PCT,
             cfg.CANDLE_LONG_MAX_POS_RATIO, cfg.CANDLE_SHORT_MIN_POS_RATIO) = prev_vals

        print(f"{sym}: OOS metrics {oos}")
        all_reports.append((sym, tuned, oos))

    print("\nSummary (OOS):")
    for sym, tuned, oos in all_reports:
        ok = (oos['sharpe'] > 1.0) and (oos['maxdd_pct'] < 20.0) and (oos['trades'] >= 30)
        print(f"{sym}: trades={oos['trades']} sharpe={oos['sharpe']:.2f} maxdd={oos['maxdd_pct']:.1f}% PASS={ok}")

if __name__ == '__main__':
    main()
