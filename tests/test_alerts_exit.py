import os
import csv
from datetime import datetime

from auto_trading_bot import alerts
from auto_trading_bot.reporter import Reporter


def test_slack_notify_exit_uses_trade_logs(tmp_path, monkeypatch):
    # Point data dir to a temp folder
    base = tmp_path / "data" / "testnet"
    base.mkdir(parents=True, exist_ok=True)

    # Use Reporter to log an entry with snapshot and then exit
    rep = Reporter(base_dir=str(tmp_path / "data"), environment="testnet")
    # Entry: short, entry=200, ATR=10, stop_k=1.0 -> denom=10; fallback_pct=0.004 -> 0.8 so ATR basis applies
    # risk_usdt_planned = 10*1 = 10
    rep.log_trade(
        "BTC/USDT",
        "short",
        200.0,
        1.0,
        reason="test",
        risk_usdt=10.0,
        timeframe="1h",
        leverage=10,
        entry_atr_abs=10.0,
        atr_period=14,
        atr_source="hlc3",
        stop_basis="atr",
        stop_k=1.0,
        fallback_pct=0.004,
        stop_distance=10.0,
        risk_usdt_planned=10.0,
    )
    # Exit: 180 => pnl_distance = 20 -> R_atr_expost = 20/10 = 2.0
    rep.log_exit("BTC/USDT", "short", 180.0, 10.0, reason="test", fees_quote_actual=0.0)

    # Monkeypatch config to use our temp data dir
    import importlib
    import config as cfg
    monkeypatch.setattr(cfg, "DATA_BASE_DIR", str(tmp_path / "data"), raising=False)
    monkeypatch.setattr(cfg, "TESTNET", True, raising=False)
    importlib.reload(alerts)

    # Capture Slack request
    os.environ["SLACK_WEBHOOK_URL"] = "https://example.com/hook"
    payloads = {}

    class DummyResp:
        def __init__(self, status):
            self.status_code = status

    def fake_post(url, data=None, headers=None, timeout=None):
        payloads["url"] = url
        payloads["data"] = data
        return DummyResp(200)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.slack_notify_exit(
        symbol="BTC/USDT",
        side="short",
        entry_price=200.0,
        exit_price=180.0,
        qty=1.0,
        pnl_usdt=20.0,
        pnl_pct=10.0,
        equity_usdt=10000.0,
    )
    assert ok is True
    data = payloads.get("data", "")
    assert "EXIT BTC/USDT SHORT" in data
    assert "PnL:" in data
    assert "Avg R: 2.000" in data
    # Verify CSV contains computed R_atr_expost = 2.0
    today = datetime.utcnow().strftime("%Y-%m-%d")
    trades_path = base / f"trades_{today}.csv"
    with trades_path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
        assert rows[-1].get("R_atr_expost") in ("2.0", "2", "2.000000", "2.000")
    os.environ.pop("SLACK_WEBHOOK_URL", None)

