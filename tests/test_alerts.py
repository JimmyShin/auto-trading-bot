import datetime
import logging
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
    sys.modules["prometheus_client"] = prometheus_client


import importlib
import json

import pytest

import auto_trading_bot.alerts as alerts
import config
from auto_trading_bot import slack_notifier
from auto_trading_bot.state_store import StateStore


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

    monkeypatch.setenv("SLACK_DRY_RUN", "false")

    class DummyRequest:
        def __init__(self, url, data=None, headers=None):
            payload["url"] = url
            payload["data"] = data
            payload["headers"] = headers

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        return DummyResponse()

    monkeypatch.setattr(slack_notifier.urllib.request, "Request", DummyRequest)
    monkeypatch.setattr(slack_notifier.urllib.request, "urlopen", fake_urlopen)

    ok = slack_notifier.SlackNotifier(logger_name="test").send(
        "hello", blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
    )
    assert ok is True
    assert payload["url"] == "https://example.com/hook"
    body = json.loads(payload["data"].decode("utf-8"))
    assert body["text"] == "hello"
    assert body["blocks"][0]["type"] == "section"


def test_slack_notify_failure(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com/hook")

    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResp(500)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.slack_notify_safely("hello")
    assert ok is False


def test_heartbeat_formats_account_label_from_env(monkeypatch, tmp_path):
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
    cfg["heartbeat_interval_sec"] = 60

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        cfg,
        sender,
        interval_sec=60.0,
        now_fn=lambda: 2000.0,
        guard_action=None,
    )
    scheduler.last_heartbeat_sent = 0.0
    monkeypatch.setattr(
        "auto_trading_bot.metrics.compute_daily_dd_ratio",
        lambda state=None, snap=None, exchange=None: 0.0,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics.get_state", lambda: StateStore(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type(
            "Stub",
            (),
            {
                "fetch_equity_snapshot": staticmethod(
                    lambda: type(
                        "Snap",
                        (),
                        {
                            "margin_balance": 0.0,
                            "ts_utc": datetime.datetime.now(datetime.timezone.utc),
                        },
                    )()
                )
            },
        )(),
    )
    scheduler.run_step()

    assert len(slack_calls) == 1
    text = slack_calls[0]["text"]
    assert text.startswith("[local] HB ✅")
    body_block = slack_calls[0]["blocks"][0]["text"]["text"]
    assert "acct:live" in body_block
    assert "Equity\n—" in body_block


def test_heartbeat_fallbacks_to_testnet_when_label_missing(monkeypatch, tmp_path):
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
    cfg["heartbeat_interval_sec"] = 60

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        cfg,
        sender,
        interval_sec=60.0,
        now_fn=lambda: 5000.0,
        guard_action=None,
    )
    scheduler.last_heartbeat_sent = 0.0
    monkeypatch.setattr(
        "auto_trading_bot.metrics.compute_daily_dd_ratio",
        lambda state=None, snap=None, exchange=None: 0.0,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics.get_state", lambda: StateStore(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type(
            "Stub",
            (),
            {
                "fetch_equity_snapshot": staticmethod(
                    lambda: type(
                        "Snap",
                        (),
                        {
                            "margin_balance": 0.0,
                            "ts_utc": datetime.datetime.now(datetime.timezone.utc),
                        },
                    )()
                )
            },
        )(),
    )
    scheduler.run_step()

    assert len(slack_calls) == 1
    text = slack_calls[0]["text"]
    assert "acct:testnet" in text
    assert text.startswith("[staging] HB ✅")


def test_emergency_dedupe_requires_escalation(monkeypatch, tmp_path):
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
        guard_action=None,
    )
    monkeypatch.setattr(
        "auto_trading_bot.alerts.compute_daily_dd_ratio", lambda state=None, snap=None: 0.6
    )
    monkeypatch.setattr(
        "auto_trading_bot.alerts.get_state", lambda: StateStore(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type(
            "Stub",
            (),
            {
                "fetch_equity_snapshot": staticmethod(
                    lambda: type(
                        "Snap",
                        (),
                        {
                            "margin_balance": 12000.0,
                            "ts_utc": datetime.datetime.now(datetime.timezone.utc),
                        },
                    )()
                )
            },
        )(),
    )

    class StubNotifier:
        def send_markdown(self, message):
            slack_calls.append(message)
            return True

    monkeypatch.setattr(alerts_mod, "_get_notifier", lambda: StubNotifier())
    scheduler._check_auto_testnet(0, scheduler.metrics.get_snapshot())

    assert len(slack_calls) == 1
    assert "AUTO_TESTNET_ON_DD" in slack_calls[0]
    assert "daily dd 60.0%" in slack_calls[0]


def test_alerts_input_logging_includes_metrics(monkeypatch, caplog):
    alerts_mod = reload_modules(
        monkeypatch,
        OBS_ACCOUNT_LABEL="ops",
        OBS_SERVER_ENV="dev",
        OBS_RUN_ID="log123",
        TESTNET="false",
        OBS_DEBUG_ALERTS="1",
    )

    snapshot = {
        "equity": 15000.12,
        "avg_r": 1.25,
        "trade_total": 12.0,
        "signal_totals": {("BTC/USDT", "1h"): 4.0},
        "order_errors": {"auth": 2.0},
        "time_drift": {"exchange": 42.0},
        "heartbeat_ts": 1700000000.0,
        "daily_dd": 0.02,
    }

    class FakeMetrics:
        def __init__(self, snap):
            self.snapshot = snap

        def get_snapshot(self):
            return self.snapshot

    slack_calls = []

    def sender(text, *, blocks=None):
        slack_calls.append(text)
        return True

    cfg = dict(config.OBSERVABILITY)
    cfg["heartbeat_interval_sec"] = 60

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        cfg,
        sender,
        interval_sec=60.0,
        now_fn=lambda: 2000.0,
        guard_action=None,
    )
    scheduler.last_heartbeat_sent = 0.0
    scheduler.last_heartbeat_trades = 7.0
    monkeypatch.setattr(
        "auto_trading_bot.metrics.compute_daily_dd_ratio",
        lambda state=None, snap=None, exchange=None: 0.2,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics.get_state", lambda: StateStore(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type(
            "Stub",
            (),
            {
                "fetch_equity_snapshot": staticmethod(
                    lambda: type(
                        "Snap",
                        (),
                        {
                            "margin_balance": 15000.12,
                            "ts_utc": datetime.datetime.now(datetime.timezone.utc),
                        },
                    )()
                )
            },
        )(),
    )
    with caplog.at_level(logging.INFO, logger="auto_trading_bot.alerts"):
        scheduler.run_step()

    alert_logs = [rec.message for rec in caplog.records if rec.message.startswith("ALERTS_INPUT ")]
    assert alert_logs, "Expected ALERTS_INPUT instrumentation"
    payload = json.loads(alert_logs[0].split(" ", 1)[1])
    assert payload["acct"] == "ops"
    assert payload["equity"] == pytest.approx(15000.12)
    assert payload["trade_count_total"] == pytest.approx(12.0)
    assert payload["signal_emitted_total"] == pytest.approx(4.0)
    assert payload["trades_delta"] == pytest.approx(5.0)


def test_auto_testnet_dd_requires_valid_equity_and_trades(monkeypatch):
    alerts_mod = reload_modules(
        monkeypatch,
        OBS_ACCOUNT_LABEL="ops",
        OBS_SERVER_ENV="dev",
        OBS_RUN_ID="ddchk",
        TESTNET="false",
        OBS_DEBUG_ALERTS="1",
    )
    monkeypatch.setenv("AUTO_TESTNET_ON_DD", "true")
    monkeypatch.setenv("DAILY_DD_LIMIT", "0.05")

    snapshot = {
        "equity": None,
        "avg_r": 0.5,
        "trade_total": 10.0,
        "signal_totals": {("BTC/USDT", "1h"): 5.0},
        "order_errors": {},
        "time_drift": {},
        "heartbeat_ts": None,
        "daily_dd": 0.2,
    }

    class FakeMetrics:
        def __init__(self, snap):
            self.snapshot = snap

        def get_snapshot(self):
            return self.snapshot

    slack_calls: list[str] = []

    def sender(text, *, blocks=None):
        slack_calls.append(text)
        return True

    config_copy = dict(config.OBSERVABILITY)
    config_copy["alert_cooldown_sec"] = 60

    current_time = 0.0

    def fake_now():
        return current_time

    scheduler = alerts_mod.AlertScheduler(
        FakeMetrics(snapshot),
        config_copy,
        sender,
        interval_sec=60.0,
        now_fn=fake_now,
        guard_action=None,
    )
    scheduler.last_heartbeat_sent = -999.0
    monkeypatch.setattr(
        "auto_trading_bot.metrics.compute_daily_dd_ratio",
        lambda state=None, snap=None, exchange=None: 0.2,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics.get_state", lambda: StateStore(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type(
            "Stub",
            (),
            {
                "fetch_equity_snapshot": staticmethod(
                    lambda: type(
                        "Snap",
                        (),
                        {
                            "margin_balance": 10000.0,
                            "ts_utc": datetime.datetime.now(datetime.timezone.utc),
                        },
                    )()
                )
            },
        )(),
    )

    class StubNotifier:
        def send_markdown(self, message):
            slack_calls.append(message)
            return True

    monkeypatch.setattr(alerts_mod, "_get_notifier", lambda: StubNotifier())
    scheduler._check_auto_testnet(0, scheduler.metrics.get_snapshot())
