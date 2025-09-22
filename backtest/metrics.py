import numpy as np
import pandas as pd


def equity_curve(trades: pd.DataFrame, fees_bps: float = 5.0, slip_bps: float = 2.0, init_equity: float = 10_000.0) -> pd.DataFrame:
    """Build equity curve from executed trades DataFrame.

    trades columns expected:
    - ts (datetime64[ns, UTC])
    - side ('long'|'short')
    - entry (float)
    - exit (float)
    - qty (float)
    """
    if trades.empty:
        return pd.DataFrame({'ts': [], 'equity': []})

    # Model fees and slippage per side (entry + exit)
    fee_rate = fees_bps / 10_000.0
    slip_rate = slip_bps / 10_000.0

    rows = []
    eq = init_equity
    for _, r in trades.sort_values('ts').iterrows():
        side = r['side']
        entry = float(r['entry']) * (1 + slip_rate if side == 'long' else 1 - slip_rate)
        exitp = float(r['exit']) * (1 - slip_rate if side == 'long' else 1 + slip_rate)
        qty = float(r['qty'])
        pos_val_entry = abs(entry * qty)
        pos_val_exit = abs(exitp * qty)
        gross = (exitp - entry) * qty if side == 'long' else (entry - exitp) * qty
        fees = (pos_val_entry + pos_val_exit) * fee_rate
        pnl = gross - fees
        eq = eq + pnl
        rows.append({'ts': r['ts'], 'pnl': pnl, 'equity': eq})
    return pd.DataFrame(rows)


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return (mean / std) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Return Max Drawdown magnitude as a positive percent.

    Example: 0.395 -> 39.5 (%). Returns 0.0 if curve is empty or flat.
    """
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    min_dd = float(dd.min()) if not dd.empty else 0.0
    return abs(min_dd) * 100.0
