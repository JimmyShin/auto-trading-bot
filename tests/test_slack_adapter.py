import json
import logging
from datetime import datetime, timezone

import pytest

from auto_trading_bot.alerts import RUNBOOK_HEADER, Alerts
from auto_trading_bot.metrics import Metrics
from auto_trading_bot.slack_notifier import SlackNotifier

pytestmark = pytest.mark.unit


def test_slack_notifier_dry_run(monkeypatch, caplog):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("SLACK_DRY_RUN", "true")

    notifier = SlackNotifier()
    caplog.set_level(logging.INFO, logger="auto_trading_bot.slack_notifier")
    notifier.send_markdown("hello")

    records = [json.loads(rec.message) for rec in caplog.records]
    assert len(records) == 1
    payload = records[0]
    assert payload["type"] == "SLACK_DRY_RUN"
    assert payload["payload"]["text"] == "hello"


class FixedMetrics(Metrics):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self._dd_ratio = ratio
        self._dd_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def bot_daily_drawdown(self) -> float:
        return self._dd_ratio


def test_alerts_guardrail_emits_slack(monkeypatch, caplog):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("SLACK_DRY_RUN", "true")

    metrics = FixedMetrics(0.25)
    alerts = Alerts(metrics, dd_guard_threshold=0.2, dedupe_window_sec=9999)

    caplog.set_level(logging.INFO)
    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    alerts.evaluate(now_utc=now)

    guard_logs = [
        json.loads(rec.message)
        for rec in caplog.records
        if "type" in rec.message and "GUARDRAIL_TRIP" in rec.message
    ]
    assert guard_logs

    dry_logs = [
        json.loads(rec.message)
        for rec in caplog.records
        if '"type": "SLACK_DRY_RUN"' in rec.message
    ]
    assert len(dry_logs) == 1
    text = dry_logs[0]["payload"]["text"]
    assert text.startswith(RUNBOOK_HEADER.splitlines()[0])
    assert "dd=25.0%" in text
    assert "threshold=20.0%" in text
    assert "mode=testnet" in text
    assert "source=binance-usdm-testnet" in text
