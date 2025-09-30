import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pytest

from auto_trading_bot.alerts import RUNBOOK_HEADER, Alerts
from auto_trading_bot.metrics import Metrics

pytestmark = pytest.mark.unit


class FixedMetrics(Metrics):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self._dd_ratio = ratio
        self._dd_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def set_ratio(self, value: float) -> None:
        self._dd_ratio = value


def _logs(caplog, record_type):
    out = []
    for rec in caplog.records:
        try:
            payload = json.loads(rec.message)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == record_type:
            out.append(payload)
    return out


def test_alerts_input_always_emitted(caplog):
    metrics = FixedMetrics(0.05)
    alerts = Alerts(metrics, heartbeat_interval_sec=30, dedupe_window_sec=120)

    now = datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc)
    caplog.set_level(logging.INFO, logger="auto_trading_bot.alerts")
    alerts.evaluate(equity=1000.0, peak_equity=1100.0, now_utc=now)

    inputs = _logs(caplog, "ALERTS_INPUT")
    assert len(inputs) == 1
    payload = inputs[0]
    assert set(payload.keys()) == {
        "type",
        "ts_utc",
        "dd_ratio",
        "equity",
        "peak_equity",
        "account_mode",
        "source",
    }
    assert payload["dd_ratio"] == pytest.approx(0.05)
    assert payload["equity"] == pytest.approx(1000.0)
    assert payload["peak_equity"] == pytest.approx(1100.0)
    assert payload["ts_utc"].endswith("Z")


def test_heartbeat_interval(caplog, monkeypatch):
    metrics = FixedMetrics(0.12)
    alerts = Alerts(metrics, heartbeat_interval_sec=60)

    base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    caplog.set_level(logging.INFO, logger="auto_trading_bot.alerts")

    monkeypatch.setattr(time, "time", lambda: base.timestamp())
    assert alerts.maybe_emit_heartbeat(now_utc=base) is True
    heartbeats = _logs(caplog, "HEARTBEAT")
    assert len(heartbeats) == 1
    assert heartbeats[0]["dd_ratio"] == pytest.approx(0.12)

    caplog.clear()
    monkeypatch.setattr(time, "time", lambda: base.timestamp() + 10)
    assert alerts.maybe_emit_heartbeat(now_utc=base + timedelta(seconds=10)) is False
    assert _logs(caplog, "HEARTBEAT") == []

    caplog.clear()
    monkeypatch.setattr(time, "time", lambda: base.timestamp() + 61)
    assert alerts.maybe_emit_heartbeat(now_utc=base + timedelta(seconds=61)) is True
    heartbeats = _logs(caplog, "HEARTBEAT")
    assert len(heartbeats) == 1
    assert heartbeats[0]["dd_ratio"] == pytest.approx(0.12)
    assert heartbeats[0]["ts_utc"].endswith("Z")


def test_guardrail_dedupe_and_runbook(caplog, monkeypatch):
    metrics = FixedMetrics(0.25)
    called = []

    def guard_action(value: float) -> None:
        called.append(value)

    alerts = Alerts(
        metrics,
        heartbeat_interval_sec=60,
        dd_guard_threshold=0.2,
        dedupe_window_sec=300,
        guard_action=guard_action,
    )

    ts = datetime(2025, 1, 3, 15, 0, tzinfo=timezone.utc)
    base_time = ts.timestamp()
    monkeypatch.setattr(time, "time", lambda: base_time)

    caplog.set_level(logging.INFO, logger="auto_trading_bot.alerts")
    alerts.evaluate(now_utc=ts)

    guard_logs = _logs(caplog, "GUARDRAIL_TRIP")
    assert len(guard_logs) == 1
    payload = guard_logs[0]
    assert set(payload.keys()) == {
        "type",
        "ts_utc",
        "guard",
        "dd_ratio",
        "threshold",
        "account_mode",
        "source",
    }
    assert payload["dd_ratio"] == pytest.approx(0.25)
    assert payload["threshold"] == pytest.approx(0.2)
    assert payload["ts_utc"].endswith("Z")

    assert len(called) == 1
    assert called[0] == pytest.approx(0.25)

    runbook_logs = [
        rec.message for rec in caplog.records if RUNBOOK_HEADER.splitlines()[0] in rec.message
    ]
    assert runbook_logs
    assert runbook_logs[0].startswith(RUNBOOK_HEADER.splitlines()[0])

    caplog.clear()
    metrics.set_ratio(0.3)
    alerts.evaluate(now_utc=ts + timedelta(seconds=30))
    assert _logs(caplog, "GUARDRAIL_TRIP") == []
    assert len(called) == 1

    caplog.clear()
    metrics.set_ratio(0.35)
    monkeypatch.setattr(time, "time", lambda: base_time + 600)
    alerts.evaluate(now_utc=ts + timedelta(seconds=600))
    guard_logs = _logs(caplog, "GUARDRAIL_TRIP")
    assert len(guard_logs) == 1
    assert guard_logs[0]["dd_ratio"] == pytest.approx(0.35)
    assert len(called) == 2


def test_clamp_dd_ratio_in_logs(caplog, monkeypatch):
    metrics = FixedMetrics(1.7)
    alerts = Alerts(metrics, heartbeat_interval_sec=30)

    ts = datetime(2025, 1, 4, 9, 0, tzinfo=timezone.utc)
    caplog.set_level(logging.INFO, logger="auto_trading_bot.alerts")
    alerts.evaluate(now_utc=ts)
    alerts.maybe_emit_heartbeat(now_utc=ts)

    guard_logs = _logs(caplog, "GUARDRAIL_TRIP")
    input_logs = _logs(caplog, "ALERTS_INPUT")
    hb_logs = _logs(caplog, "HEARTBEAT")

    if guard_logs:
        assert guard_logs[0]["dd_ratio"] <= 1.0
    assert input_logs[0]["dd_ratio"] <= 1.0
    assert hb_logs and hb_logs[0]["dd_ratio"] <= 1.0

    caplog.clear()
    metrics.set_ratio(-0.5)
    alerts.evaluate(now_utc=ts + timedelta(minutes=1))
    alerts.maybe_emit_heartbeat(now_utc=ts + timedelta(minutes=1))

    input_logs = _logs(caplog, "ALERTS_INPUT")
    hb_logs = _logs(caplog, "HEARTBEAT")
    assert input_logs[0]["dd_ratio"] >= 0.0
    assert hb_logs[0]["dd_ratio"] >= 0.0
