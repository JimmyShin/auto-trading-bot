import argparse
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from datetime import timedelta
import os, sys

# Ensure repo root on sys.path for top-level modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from indicators import atr as atr_func, ma_crossover_signal, is_near_funding
from strategy import DonchianATREngine
from backtest.utils import ensure_utc, dedup_sort
from backtest.metrics import equity_curve, sharpe, max_drawdown


@dataclass
class SimParams:
    fees_bps: float = 5.0
    slip_bps: float = 2.0
    leverage: int = 1


def simulate_symbol(df: pd.DataFrame, params: SimParams, symbol: str = "SIM") -> pd.DataFrame:
    """Single-symbol, bar-close simulation using current strategy rules.

    Returns a trades DataFrame with columns: ts, side, entry, exit, qty
    """
    df = ensure_utc(dedup_sort(df))
    if len(df) < 200:
        return pd.DataFrame(columns=['ts','side','entry','exit','qty'])

    eng = DonchianATREngine()  # use only stateless helpers (no state writes here)

    in_pos = False
    side: Optional[str] = None
    entry = 0.0
    qty = 0.0
    be = False
    trail_prev: Optional[float] = None
    initial_stop: Optional[float] = None
    trades = []
    equity = 10_000.0
    # Daily loss control
    from config import DAILY_LOSS_LIMIT
    current_day = None
    daily_anchor = equity
    daily_blocked = False

    for i in range(200, len(df)):
        window = df.iloc[:i+1]
        row = window.iloc[-1]
        # daily anchor reset on UTC day change
        day = row['ts'].to_pydatetime().date()
        if current_day is None or day != current_day:
            current_day = day
            daily_anchor = equity
            daily_blocked = False
        else:
            # update block if drawdown from anchor exceeds limit
            if daily_anchor > 0:
                dd = max(0.0, (daily_anchor - equity) / daily_anchor)
                if dd >= DAILY_LOSS_LIMIT:
                    daily_blocked = True
        price = float(row['close'])
        atr_abs = float(atr_func(window, n=14))

        if not in_pos:
            # replicate engine decision at bar close
            plan = eng.make_entry_plan(
                symbol=symbol,
                df=window,
                equity=equity,
                price=price,
                atr_abs=atr_abs,
                is_new_bar=True,
                can_reenter=True,
                funding_avoid=is_near_funding(row['ts'].to_pydatetime()),
                daily_loss_hit=daily_blocked,
            )
            decision = plan.get('decision', 'SKIP')
            if decision in ("ENTER_LONG", "ENTER_SHORT"):
                side = 'long' if decision == 'ENTER_LONG' else 'short'
                qty = float(plan.get('qty') or 0.0)
                if qty <= 0:
                    continue
                entry = price
                be = False
                trail_prev = None
                initial_stop = float(plan.get('stop_price') or 0.0)
                in_pos = True
        else:
            # promote to BE after +1.5*ATR move
            from config import ATR_STOP_K
            if side == 'long' and (price - entry) >= 1.5 * atr_abs:
                be = True
            if side == 'short' and (entry - price) >= 1.5 * atr_abs:
                be = True

            if not be and initial_stop:
                stop_line = float(initial_stop)
            else:
                trail = eng.trail_stop_price(side, entry, price, atr_abs, be)
                # monotonic constraint once trailing
                if trail_prev is not None:
                    if side == 'long':
                        trail = max(trail_prev, trail)
                    else:
                        trail = min(trail_prev, trail)
                trail_prev = trail
                stop_line = trail

            high = float(row['high'])
            low = float(row['low'])
            exit_hit = False
            if side == 'long' and low <= stop_line:
                exit_price = stop_line
                exit_hit = True
            elif side == 'short' and high >= stop_line:
                exit_price = stop_line
                exit_hit = True

            if exit_hit:
                trades.append({
                    'ts': row['ts'],
                    'side': side,
                    'entry': entry,
                    'exit': exit_price,
                    'qty': qty,
                })
                in_pos = False
                side = None
                qty = 0.0
                entry = 0.0
                be = False
                trail_prev = None
                initial_stop = None

    return pd.DataFrame(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True, help='Path to parquet for a symbol')
    ap.add_argument('--fees_bps', type=float, default=5.0)
    ap.add_argument('--slip_bps', type=float, default=2.0)
    args = ap.parse_args()

    # allow CSV fallback if parquet engine missing
    from backtest.utils import read_parquet_or_csv
    df = read_parquet_or_csv(args.parquet)
    trades = simulate_symbol(df, SimParams(fees_bps=args.fees_bps, slip_bps=args.slip_bps))
    curve = equity_curve(trades, args.fees_bps, args.slip_bps)
    if not curve.empty:
        ret = curve['equity'].pct_change().dropna()
        shp = sharpe(ret, periods_per_year=365*24*4)  # 15m bars
        mdd = max_drawdown(curve['equity'])
        print(f"Trades={len(trades)} Sharpe={shp:.2f} MaxDD={mdd:.1f}%")
    else:
        print("No trades generated")

if __name__ == '__main__':
    main()
