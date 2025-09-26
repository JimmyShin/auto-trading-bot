import os
import csv
from datetime import datetime

from auto_trading_bot.reporter import Reporter


def test_r_atr_long_and_short(tmp_path, monkeypatch):
    base = tmp_path / "data" / "testnet"
    base.mkdir(parents=True, exist_ok=True)
    rep = Reporter(base_dir=str(tmp_path / "data"), environment="testnet")

    # Long: entry=100 exit=110; ATR=5, stop_k=2 -> denom=10; R=+1.0
    rep.log_trade(
        "ETH/USDT", "long", 100.0, 2.0,
        timeframe="1h", leverage=5,
        entry_atr_abs=5.0, atr_period=14, atr_source="hlc3",
        stop_basis="atr", stop_k=2.0, fallback_pct=0.004,
        stop_distance=10.0, risk_usdt_planned=20.0,
    )
    rep.log_exit("ETH/USDT", "long", 110.0, 10.0, reason="test")

    # Short: entry=200 exit=180; ATR=10, stop_k=1 -> denom=10; R=+2.0
    rep.log_trade(
        "BTC/USDT", "short", 200.0, 1.0,
        timeframe="1h", leverage=10,
        entry_atr_abs=10.0, atr_period=14, atr_source="hlc3",
        stop_basis="atr", stop_k=1.0, fallback_pct=0.004,
        stop_distance=10.0, risk_usdt_planned=10.0,
    )
    rep.log_exit("BTC/USDT", "short", 180.0, 10.0, reason="test")

    today = datetime.utcnow().strftime("%Y-%m-%d")
    trades_path = base / f"trades_{today}.csv"
    with trades_path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    # Last row corresponds to BTC short
    assert rows[-1].get("R_atr_expost") in ("2.0", "2", "2.000000", "2.000")
    # First closed row corresponds to ETH long -> R=1
    # Find first CLOSED row for ETH
    eth_rows = [row for row in rows if row.get("symbol") == "ETH/USDT" and (row.get("status") or "").upper() == "CLOSED"]
    assert eth_rows, "ETH closed row missing"
    assert eth_rows[-1].get("R_atr_expost") in ("1.0", "1", "1.000000", "1.000")


def test_percent_basis_risk_and_fees(tmp_path):
    # Set up a trade where percent fallback dominates
    base = tmp_path / "data" / "testnet"
    base.mkdir(parents=True, exist_ok=True)
    rep = Reporter(base_dir=str(tmp_path / "data"), environment="testnet")

    # Price=100, ATR very small -> fallback_pct=0.04*100=4 > stop_k*atr_abs=1
    rep.log_trade(
        "SOL/USDT", "long", 100.0, 5.0,
        timeframe="1h", leverage=5,
        entry_atr_abs=1.0, atr_period=14, atr_source="hlc3",
        stop_basis="percent", stop_k=1.0, fallback_pct=0.04,
        stop_distance=4.0, risk_usdt_planned=20.0,
    )
    # Exit with small win: 101 -> pnl_distance=1 -> pnl_quote=5; fees 0.5 -> pnl_quote_expost=4.5
    rep.log_exit("SOL/USDT", "long", 101.0, 1.0, reason="test", fees_quote_actual=0.5)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    trades_path = base / f"trades_{today}.csv"
    with trades_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    row = rows[-1]
    # ATR-based denominator is still stop_k*entry_atr_abs=1.0 so R_atr_expost = pnl_distance(1)/1 = 1
    assert row.get("R_atr_expost") in ("1.0", "1", "1.000000")
    # R_usd_expost = pnl_quote_expost / risk_usdt_planned = 4.5/20 = 0.225
    val = float(row.get("R_usd_expost") or 0.0)
    assert abs(val - 0.225) < 1e-6


