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
import json
import math

import pytest

import config
import auto_trading_bot.alerts as alerts


class DummyResp:
    def __init__(self, status: int) -> None:
        self.status_code = status


@pytest.fixture(autouse=True)
def reset_slack_singleton():
    alerts._SLACK = None
    yield
    alerts._SLACK = None


def reload_modules(monkeypatch, **env) -> alerts.__class__:
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    importlib.reload(config)
    module = importlib.reload(alerts)
    module._SLACK = None
    return module


def test_slack_notify_success(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/hook")
    payload = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        payload["url"] = url
        payload["json"] = json
        payload["headers"] = headers
        return DummyResp(200)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.slack_notify_safely(
        "hello",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}],
    )
    assert ok is True
    assert payload["url"] == "https://example.com/hook"
    assert payload["json"]["text"] == "hello"
    assert payload["json"]["blocks"][0]["type"] == "section"


def test_slack_notify_failure(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/hook")

    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResp(500)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.slack_notify_safely("hello")
    assert ok is False


def test_heartbeat_formats_account_label_from_env(monkeypatch):
    alerts_mod = reload_modules(
        monkeypatch,
        OBS_ACCOUNT_LABEL="desk",
        OBS_SERVER_ENV="local",
        OBS_RUN_ID="testrn1",
        TESTNET="false",
    )

    snapshot = {
        "equity": float("nan"),
        "avg_r": float("nan"),
        "trade_total": None,
        "signal_totals": {("BTC/USDT", "1h"): 4},
        "order_errors": {},
        "time_drift": {},
        "heartbeat_ts": None,
        "daily_dd": float("nan"),
    }

    class FakeMetrics:
        def __init__(self, snap):
            self.snapshot = snap

        def get_snapshot(self):
            return self.snapshot

    slack_calls = []

    def sender(text, *, blocks=None):
        slack_calls.append({"text": text, "blocks": blocks})
        return True

    cfg = dict(config.OBSERVABILITY)
    cfg['heartbeat_interval_sec'] = 60

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        cfg,
        sender,
        interval_sec=60.0,
        now_fn=lambda: 2000.0,
    )
    scheduler.last_heartbeat_sent = 0.0
    scheduler.run_step()

    assert len(slack_calls) == 1
    text = slack_calls[0]["text"]
    assert text.startswith("[local] HB ✅")
    assert "acct:desk" in text
    assert "eq:NA" in text and "rATR30:NA" in text
    context_block = slack_calls[0]["blocks"][-1]
    joined = " ".join(elem["text"] for elem in context_block["elements"])
    assert "thread:testrn1:HB" in joined


def test_heartbeat_fallbacks_to_testnet_when_label_missing(monkeypatch):
    alerts_mod = reload_modules(
        monkeypatch,
        OBS_ACCOUNT_LABEL=None,
        OBS_SERVER_ENV="staging",
        TESTNET="true",
        OBS_RUN_ID="testrun2",
    )

    snapshot = {
        "equity": 10500.12,
        "avg_r": 1.234,
        "trade_total": 10,
        "signal_totals": {("BTC/USDT", "1h"): 10},
        "order_errors": {},
        "time_drift": {},
        "heartbeat_ts": 1234567890,
        "daily_dd": 0.01,
    }

    class FakeMetrics:
        def __init__(self, snap):
            self.snapshot = snap

        def get_snapshot(self):
            return self.snapshot

    slack_calls = []

    def sender(text, *, blocks=None):
        slack_calls.append({"text": text, "blocks": blocks})
        return True

    cfg = dict(config.OBSERVABILITY)
    cfg['heartbeat_interval_sec'] = 60

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        cfg,
        sender,
        interval_sec=60.0,
        now_fn=lambda: 5000.0,
    )
    scheduler.last_heartbeat_sent = 0.0
    scheduler.run_step()

    assert len(slack_calls) == 1
    text = slack_calls[0]["text"]
    assert "acct:testnet" in text
    assert text.startswith("[staging] HB ✅")


def test_emergency_dedupe_requires_escalation(monkeypatch):
    alerts_mod = reload_modules(
        monkeypatch,
        OBS_ACCOUNT_LABEL="ops",
        OBS_SERVER_ENV="dev",
        OBS_RUN_ID="dedupe1",
        TESTNET="false",
    )

    class FakeMetrics:
        def __init__(self):
            self.snapshot = {
                "equity": 10000.0,
                "avg_r": 0.5,
                "trade_total": 5,
                "signal_totals": {},
                "order_errors": {},
                "time_drift": {"exchange": 0.0},
                "heartbeat_ts": None,
                "daily_dd": 0.0,
            }

        def get_snapshot(self):
            return self.snapshot

    metrics = FakeMetrics()
    slack_calls = []

    def sender(text, *, blocks=None):
        slack_calls.append(text)
        return True

    config_copy = dict(config.OBSERVABILITY)
    config_copy["clock_drift_warn_ms"] = 5000.0
    config_copy["alert_cooldown_sec"] = 600

    current_time = 0.0

    def fake_now():
        return current_time

    scheduler = alerts_mod.AlertScheduler(
        metrics,
        config_copy,
        sender,
        interval_sec=60.0,
        now_fn=fake_now,
    )

    drift_series = [7000, 7000, 7000, 7000, 7000, 7000, 7000, 9000]
    for step, drift in enumerate(drift_series):
        current_time = step * 60.0
        metrics.snapshot["time_drift"] = {"exchange": drift}
        scheduler.run_step()

    assert len(slack_calls) == 2
    assert "7000ms" in slack_calls[0]
    assert "9000ms" in slack_calls[1]
