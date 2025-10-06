import json
import csv
import os
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone

import pandas as pd
import pytest

from auto_trading_bot.reporter import Reporter, generate_report
from auto_trading_bot.main import _persist_primary_entry
from baseline import DEFAULT_BASELINE_PATH, generate_baseline, looks_like_fallback

# Mark all tests in this module as regression (slow)
pytestmark = pytest.mark.regression


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture(scope="session")
def baseline_path() -> Path:
    return DEFAULT_BASELINE_PATH


@pytest.fixture(scope="session")
def baseline_data(baseline_path: Path) -> Dict[str, Any]:
    os.environ.setdefault("ALLOW_SYNTHETIC_BASELINE", "1")
    if not baseline_path.exists():
        pytest.skip(f"Baseline file missing: {baseline_path}")
    with baseline_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def current_generated(baseline_data: Dict[str, Any], tmp_path_factory) -> Dict[str, Any]:
    # Use parameters captured in the baseline to regenerate
    timeframe = baseline_data.get("timeframe")
    bars = int(baseline_data.get("bars") or 180)
    equity = float(baseline_data.get("equity") or 20000.0)
    symbols = list((baseline_data.get("symbols") or {}).keys())
    assert symbols, "Baseline has no symbols to validate"
    regen_dir = tmp_path_factory.mktemp("regen_baseline")
    regen_path = regen_dir / "baseline.json"
    regenerated = generate_baseline(
        symbols=symbols, timeframe=timeframe, bars=bars, equity=equity, output_path=regen_path
    )
    if looks_like_fallback(regenerated.get("symbols", {})):
        pytest.skip("Synthetic baseline fixtures detected; skipping regression comparison.")
    return regenerated


# --------------------------
# Helpers
# --------------------------


def _extract_trades_from_records(symbol_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build trades from per-candle baseline records for a single symbol.

    - Entry when decision is ENTER_LONG/ENTER_SHORT and qty > 0
    - Exit when record contains non-null 'exit' with 'price'
    - Open trades without an exit are ignored
    """
    trades: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] | None = None

    for r in symbol_records:
        ts = pd.to_datetime(r.get("ts"), utc=True)
        decision = r.get("decision") or ""
        side = (r.get("side") or "").lower()
        qty = float(r.get("qty") or 0.0)
        close_price = float(r.get("close") or 0.0)

        # Detect entry
        if open_trade is None and decision in ("ENTER_LONG", "ENTER_SHORT") and qty > 0:
            open_trade = {
                "side": (
                    side
                    if side in ("long", "short")
                    else ("long" if "LONG" in decision else "short")
                ),
                "entry": close_price,
                "entry_ts": ts,
                "qty": qty,
            }
            continue

        # Detect exit
        exit_info = r.get("exit") or None
        if open_trade is not None and exit_info is not None:
            try:
                exit_price = float(exit_info.get("price") or 0.0)
            except Exception:
                exit_price = 0.0
            trade = {
                **open_trade,
                "exit": exit_price,
                "exit_ts": ts,
            }
            trades.append(trade)
            open_trade = None

    return trades


def _trades_from_baseline(baseline_obj: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sym, records in (baseline_obj.get("symbols") or {}).items():
        rows.extend(_extract_trades_from_records(records))
    return pd.DataFrame(rows)


# --------------------------
# Tests
# --------------------------


def test_signals_match_exact(baseline_data: Dict[str, Any], current_generated: Dict[str, Any]):
    assert (
        current_generated["symbols"] == baseline_data["symbols"]
    ), "Signals diverged from baseline"


def test_report_metrics_regression(
    baseline_data: Dict[str, Any], current_generated: Dict[str, Any]
):
    # Current metrics from regenerated decisions -> trades -> reporter.generate_report
    current_trades = _trades_from_baseline(current_generated)
    current_df = generate_report(current_trades)
    assert len(current_df) == 1

    # Expected metrics either present in baseline or computed from baseline's own trades
    expected_metrics = baseline_data.get("report_metrics")
    if not expected_metrics:
        baseline_trades = _trades_from_baseline(baseline_data)
        expected_df = generate_report(baseline_trades)
        expected_metrics = expected_df.iloc[0].to_dict()

    # Allow both ratio (0..1) or percent (0..100) for win_rate if provided in baseline
    def norm_rate(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return v / 100.0 if v > 1.0 else v

    expected = {
        "win_rate": norm_rate(expected_metrics.get("win_rate")),
        "expectancy": float(expected_metrics.get("expectancy", 0.0)),
        "mdd": float(expected_metrics.get("mdd", 0.0)),
        "profit_factor": float(expected_metrics.get("profit_factor", 0.0)),
        "sharpe": float(expected_metrics.get("sharpe", 0.0)),
    }

    actual = {
        "win_rate": float(current_df.loc[0, "win_rate"]),
        "expectancy": float(current_df.loc[0, "expectancy"]),
        "mdd": float(current_df.loc[0, "mdd"]),
        "profit_factor": float(current_df.loc[0, "profit_factor"]),
        "sharpe": float(current_df.loc[0, "sharpe"]),
    }

    import math

    for key in expected.keys():
        exp = expected[key]
        act = actual[key]
        if math.isinf(exp) or math.isinf(act):
            assert math.isinf(exp) and math.isinf(act), f"Metric {key} mismatch (infinite check)"
        else:
            assert act == pytest.approx(exp, rel=1e-2, abs=1e-2), f"Metric {key} mismatch"



def test_primary_entry_logging_creates_trade_artifacts(tmp_path):
    base_dir = tmp_path / "data"
    fixed_now = datetime(2025, 10, 1, 12, 0, tzinfo=timezone.utc)
    reporter = Reporter(str(base_dir), "testnet", clock=lambda: fixed_now)

    plan_tags = {
        "strategy_name": "ma_cross_5_20",
        "strategy_version": "1",
        "strategy_params_hash": "hash123",
        "strategy_params": {"fast": 5, "slow": 20},
    }
    plan = {
        "decision": "ENTER_LONG",
        "risk_multiplier": 0.75,
        "signal": {"fast_ma": 114108.92, "slow_ma": 113538.97},
        "reasons": ["trend_alignment", "candle_safe"],
        "strategy_tags": plan_tags,
    }

    _persist_primary_entry(
        reporter,
        "BTC/USDT",
        "long",
        114000.0,
        0.219,
        113200.0,
        plan,
        plan_tags,
        equity=15000.0,
        atr_abs=600.0,
        risk_usdt=175.0,
    )

    env_dir = base_dir / "testnet"
    trades_files = list(env_dir.glob("trades_*.csv"))
    assert len(trades_files) == 1
    with trades_files[0].open("r", encoding="utf-8", newline="") as handle:
        trade_rows = list(csv.DictReader(handle))
    assert len(trade_rows) == 1
    trade_row = trade_rows[0]
    assert trade_row["symbol"] == "BTC/USDT"
    assert trade_row["status"] == "OPEN"
    assert trade_row["stop_basis"] == "atr"
    assert trade_row["strategy_name"] == plan_tags["strategy_name"]
    assert pytest.approx(float(trade_row["risk_usdt_planned"]), rel=1e-9) == 175.0

    detail_files = list(env_dir.glob("detailed_entries_*.csv"))
    assert len(detail_files) == 1
    with detail_files[0].open("r", encoding="utf-8", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))
    assert len(detail_rows) == 1
    detail_row = detail_rows[0]
    assert detail_row["strategy_name"] == plan_tags["strategy_name"]
    assert detail_row["strategy_params_hash"] == plan_tags["strategy_params_hash"]
    assert "trend_alignment" in detail_row["reasons"]
    assert pytest.approx(float(detail_row["risk_multiplier"]), rel=1e-9) == 0.75
