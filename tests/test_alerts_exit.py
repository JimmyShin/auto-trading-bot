import sys

try:
    import prometheus_client  # type: ignore
except ModuleNotFoundError:
    import types

    class _DummyMetric:
        def labels(self, **kwargs):
            return self

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    prometheus_client = types.SimpleNamespace(
        Counter=lambda *args, **kwargs: _DummyMetric(),
        Gauge=lambda *args, **kwargs: _DummyMetric(),
        Histogram=lambda *args, **kwargs: _DummyMetric(),
        start_http_server=lambda *args, **kwargs: None,
    )
    sys.modules['prometheus_client'] = prometheus_client


import importlib
from datetime import datetime

import pytest

import config
import auto_trading_bot.alerts as alerts
from auto_trading_bot.reporter import Reporter


def reload_for_exit(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    importlib.reload(config)
    module = importlib.reload(alerts)
    module._SLACK = None
    return module


def test_slack_notify_exit_suppresses_sentinel(monkeypatch):
    alerts_mod = reload_for_exit(
        monkeypatch,
        OBS_ACCOUNT_LABEL="ops",
        OBS_SERVER_ENV="dev",
        OBS_RUN_ID="exit001",
        TESTNET="true",
    )

    calls = []

    def sender(text, *, blocks=None):
        calls.append((text, blocks))
        return True

    ok = alerts_mod.slack_notify_exit(
        symbol="BTC/USDT",
        side="long",
        entry_price=200.0,
        exit_price=200.25,
        qty=1.0,
        pnl_usdt=0.0,
        pnl_pct=0.0,
        equity_usdt=1000.0,
        sender=sender,
    )

    assert ok is False
    assert calls == []


def test_slack_notify_exit_posts_blocks_for_valid_fill(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    alerts_mod = reload_for_exit(
        monkeypatch,
        OBS_ACCOUNT_LABEL="ops",
        OBS_SERVER_ENV="staging",
        OBS_RUN_ID="exitABC",
        TESTNET="true",
    )
    monkeypatch.setattr(config, "DATA_BASE_DIR", str(data_dir), raising=False)

    reporter = Reporter(base_dir=str(data_dir), environment="testnet")
    reporter.log_trade(
        "BTC/USDT",
        "short",
        200.0,
        1.0,
        timeframe="1h",
        leverage=5,
        entry_atr_abs=10.0,
        stop_k=1.0,
        fallback_pct=0.01,
        risk_usdt_planned=10.0,
    )
    reporter.log_exit("BTC/USDT", "short", 180.0, 10.0, fees_quote_actual=0.5)

    calls = []

    def sender(text, *, blocks=None):
        calls.append({"text": text, "blocks": blocks})
        return True

    ok = alerts_mod.slack_notify_exit(
        symbol="BTC/USDT",
        side="short",
        entry_price=200.0,
        exit_price=180.0,
        qty=1.0,
        pnl_usdt=20.0,
        pnl_pct=10.0,
        equity_usdt=10000.0,
        sender=sender,
    )

    assert ok is True
    assert len(calls) == 1
    message = calls[0]["text"]
    assert message.startswith("[staging] ✅ EXIT BTC/USDT SHORT")
    assert "run:exitABC" in message
    assert "acct:ops" in message
    blocks = calls[0]["blocks"]
    assert blocks[0]["type"] == "header"
    assert blocks[1]["fields"]
    context_text = " ".join(elem["text"] for elem in blocks[2]["elements"])
    assert "thread:exitABC:EXIT" in context_text
    assert "fill_id:" in context_text
