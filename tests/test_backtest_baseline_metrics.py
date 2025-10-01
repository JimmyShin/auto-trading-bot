import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from auto_trading_bot.reporter import generate_report
from backtest.metrics import equity_curve, max_drawdown
from baseline import DEFAULT_BASELINE_PATH

# Mark this module as regression (computes equity curve from baseline trades)
pytestmark = pytest.mark.regression


def _extract_trades_from_records(symbol_records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] | None = None
    for r in symbol_records:
        ts = pd.to_datetime(r.get("ts"), utc=True)
        decision = r.get("decision") or ""
        side = (r.get("side") or "").lower()
        qty = float(r.get("qty") or 0.0)
        close_price = float(r.get("close") or 0.0)

        if open_trade is None and decision in ("ENTER_LONG", "ENTER_SHORT") and qty > 0:
            open_trade = {
                "ts": ts,
                "side": (
                    side
                    if side in ("long", "short")
                    else ("long" if "LONG" in decision else "short")
                ),
                "entry": close_price,
                "qty": qty,
            }
            continue

        exit_info = r.get("exit") or None
        if open_trade is not None and exit_info is not None:
            try:
                exit_price = float(exit_info.get("price") or 0.0)
            except Exception:
                exit_price = 0.0
            rows.append({**open_trade, "exit": exit_price})
            open_trade = None

    return pd.DataFrame(rows)


@pytest.mark.parametrize("baseline_path", [DEFAULT_BASELINE_PATH])
def test_backtest_equity_metrics_match_reporter(baseline_path: Path):
    if not baseline_path.exists():
        pytest.skip(f"Baseline file missing: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    symbols = list((baseline.get("symbols") or {}).keys())
    if not symbols:
        pytest.skip("No symbols in baseline")

    # Concatenate all symbol trades from baseline to build a unified equity curve
    all_trades: List[pd.DataFrame] = []
    for sym in symbols:
        records = baseline["symbols"][sym]
        df_sym = _extract_trades_from_records(records)
        if not df_sym.empty:
            all_trades.append(df_sym)
    if not all_trades:
        pytest.skip("Baseline has no trades across symbols")
    trades = pd.concat(all_trades, ignore_index=True)

    # Reporter metrics (ground truth) from baseline trades
    rep = generate_report(trades).iloc[0]

    # Backtest equity metrics computed from baseline trades (no fees/slippage to align)
    baseline_equity = float(baseline.get("equity") or 10_000.0)
    curve = equity_curve(trades, fees_bps=0.0, slip_bps=0.0, init_equity=baseline_equity)
    if curve.empty or curve["equity"].nunique() <= 1:
        pytest.skip("Not enough trade equity data to compute metrics")

    # Max drawdown: backtest returns percent; reporter stores ratio
    mdd_ratio_from_backtest = max_drawdown(curve["equity"]) / 100.0
    # Allow tolerance: equity-based MDD vs return-based MDD can differ due to sizing
    assert float(mdd_ratio_from_backtest) == pytest.approx(float(rep["mdd"]), rel=0.25, abs=0.1)

    # Non-annualized Sharpe to match reporter (mean/std of equity returns)
    returns = curve["equity"].pct_change().dropna()
    if len(returns) >= 2 and float(returns.std(ddof=1)) > 0:
        sharpe_na = float(returns.mean() / returns.std(ddof=1))
        # Allow tolerance for equity-based vs per-trade return differences
        assert sharpe_na == pytest.approx(float(rep["sharpe"]), rel=0.25, abs=0.25)
    else:
        assert float(rep["sharpe"]) == pytest.approx(0.0, abs=1e-9)
