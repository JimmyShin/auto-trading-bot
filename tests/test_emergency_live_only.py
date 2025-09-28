import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from auto_trading_bot.alerts import AlertScheduler
from auto_trading_bot.metrics import Metrics
from auto_trading_bot.state_store import StateStore


@pytest.fixture
def fake_store(tmp_path):
    path = tmp_path / "state.json"
    store = StateStore(path)
    return store


class DummyMetrics(Metrics):
    def __init__(self, dd_ratio: float):
        super().__init__()
        self._ratio = dd_ratio

    def get_snapshot(self):
        return {"daily_dd": self._ratio}

    def bot_daily_drawdown(self):
        return self._ratio


def test_emergency_suppressed_on_testnet(monkeypatch, fake_store):
    metrics = DummyMetrics(dd_ratio=0.5)
    sender = Mock(return_value=True)
    scheduler = AlertScheduler(metrics, {"heartbeat_interval_sec": 0}, sender, now_fn=lambda: 0, guard_action=None)
    monkeypatch.setattr("auto_trading_bot.alerts.config.get_trading_mode", lambda: "testnet")
    monkeypatch.setenv("AUTO_TESTNET_ON_DD_THRESHOLD", "0.25")
    monkeypatch.setattr("auto_trading_bot.alerts.get_state", lambda: fake_store)
    monkeypatch.setattr(
        "auto_trading_bot.alerts.compute_daily_dd_ratio",
        lambda state=None, snap=None: 0.0,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type("Dummy", (), {"fetch_equity_snapshot": staticmethod(lambda: type("Snap", (), {"margin_balance": 1000.0, "ts_utc": datetime.datetime.now(datetime.timezone.utc)})())})(),
    )
    monkeypatch.setattr(
        "auto_trading_bot.alerts._get_notifier",
        lambda: type(
            "Stub",
            (),
            {
                "send_markdown": lambda self, msg: sender(msg),
            },
        )(),
    )
    scheduler._check_auto_testnet(0, metrics.get_snapshot())
    sender.assert_not_called()


def test_emergency_triggers_once_in_live(monkeypatch, fake_store):
    metrics = DummyMetrics(dd_ratio=0.5)
    sender = Mock(return_value=True)
    scheduler = AlertScheduler(metrics, {"heartbeat_interval_sec": 0}, sender, now_fn=lambda: 0, guard_action=None)
    monkeypatch.setattr("auto_trading_bot.alerts.config.get_trading_mode", lambda: "live")
    monkeypatch.setenv("AUTO_TESTNET_ON_DD_THRESHOLD", "0.25")
    monkeypatch.setattr("auto_trading_bot.alerts.get_state", lambda: fake_store)
    monkeypatch.setattr(
        "auto_trading_bot.alerts.compute_daily_dd_ratio",
        lambda state=None, snap=None: 0.5,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type("Dummy", (), {"fetch_equity_snapshot": staticmethod(lambda: type("Snap", (), {"margin_balance": 1000.0, "ts_utc": datetime.datetime.now(datetime.timezone.utc)})())})(),
    )
    monkeypatch.setattr(
        "auto_trading_bot.alerts._get_notifier",
        lambda: type(
            "Stub",
            (),
            {
                "send_markdown": lambda self, msg: sender(msg),
            },
        )(),
    )
    scheduler._check_auto_testnet(0, metrics.get_snapshot())
    assert sender.call_count == 1
    scheduler._check_auto_testnet(0, metrics.get_snapshot())
    assert sender.call_count == 1


def test_emergency_allows_after_rollover(monkeypatch, fake_store):
    metrics = DummyMetrics(dd_ratio=0.5)
    sender = Mock(return_value=True)
    scheduler = AlertScheduler(metrics, {"heartbeat_interval_sec": 0}, sender, now_fn=lambda: 0, guard_action=None)
    monkeypatch.setattr("auto_trading_bot.alerts.config.get_trading_mode", lambda: "live")
    monkeypatch.setenv("AUTO_TESTNET_ON_DD_THRESHOLD", "0.25")
    monkeypatch.setattr("auto_trading_bot.alerts.get_state", lambda: fake_store)
    monkeypatch.setattr(
        "auto_trading_bot.alerts.compute_daily_dd_ratio",
        lambda state=None, snap=None: 0.5,
    )
    monkeypatch.setattr(
        "auto_trading_bot.metrics._fetch_snapshot",
        type("Dummy", (), {"fetch_equity_snapshot": staticmethod(lambda: type("Snap", (), {"margin_balance": 1000.0, "ts_utc": datetime.datetime.now(datetime.timezone.utc)})())})(),
    )
    monkeypatch.setattr(
        "auto_trading_bot.alerts._get_notifier",
        lambda: type(
            "Stub",
            (),
            {
                "send_markdown": lambda self, msg: sender(msg),
            },
        )(),
    )
    scheduler._check_auto_testnet(0, metrics.get_snapshot())
    assert sender.call_count == 1
    next_day = datetime.date.today() + datetime.timedelta(days=1)
    fake_store.update_daily_peak_equity(0, next_day)
    fake_store._save({"daily": {}, "dedupe": {}})
    scheduler._check_auto_testnet(0, metrics.get_snapshot())
    assert sender.call_count == 2
