import logging
from datetime import datetime

import pytest

from auto_trading_bot import main as bot_main


@pytest.fixture(autouse=True)
def reset_daily_report(monkeypatch):
    monkeypatch.setattr(bot_main, "DAILY_REPORT_ENABLED", False, raising=False)
    monkeypatch.setattr(bot_main, "_gen_daily_report", None, raising=False)
    return


def test_emit_daily_report_disabled_skips(monkeypatch):
    calls = []

    def fake_gen(date_str, env, base_dir):
        calls.append((date_str, env, base_dir))
        return "ignored", {}

    messages = []

    monkeypatch.setattr(bot_main, "_gen_daily_report", fake_gen, raising=False)
    monkeypatch.setattr(bot_main, "slack_notify_safely", lambda msg: messages.append(msg), raising=False)
    monkeypatch.setattr(bot_main, "DAILY_REPORT_ENABLED", False, raising=False)

    bot_main._emit_daily_report(datetime(2025, 9, 30, 12, 0, 0), "testnet", "data")

    assert calls == []
    assert messages == []


def test_emit_daily_report_missing_generator_logs(monkeypatch, caplog):
    messages = []

    monkeypatch.setattr(bot_main, "_gen_daily_report", None, raising=False)
    monkeypatch.setattr(bot_main, "slack_notify_safely", lambda msg: messages.append(msg), raising=False)
    monkeypatch.setattr(bot_main, "DAILY_REPORT_ENABLED", True, raising=False)

    with caplog.at_level(logging.WARNING):
        bot_main._emit_daily_report(datetime(2025, 9, 30, 12, 0, 0), "testnet", "data")

    assert messages == []
    assert any("Daily report generation skipped" in record.message for record in caplog.records)


def test_emit_daily_report_enabled_sends(monkeypatch):
    messages = []
    captured_args = []

    summary = {
        "trades": 5,
        "wins": 3,
        "losses": 2,
        "win_rate_pct": 60.0,
        "gross_win_pct": 1.5,
        "gross_loss_pct": 0.5,
        "profit_factor": 3.0,
    }

    def fake_gen(date_str, env, base_dir):
        captured_args.append((date_str, env, base_dir))
        return "/tmp/report.csv", summary

    monkeypatch.setattr(bot_main, "_gen_daily_report", fake_gen, raising=False)
    monkeypatch.setattr(bot_main, "slack_notify_safely", lambda msg: messages.append(msg), raising=False)
    monkeypatch.setattr(bot_main, "DAILY_REPORT_ENABLED", True, raising=False)

    now = datetime(2025, 9, 30, 12, 0, 0)
    bot_main._emit_daily_report(now, "testnet", "data")

    assert captured_args == [("2025-09-30", "testnet", "data")]
    assert len(messages) == 1
    assert "Daily testnet summary 2025-09-30" in messages[0]
    assert "trades=5" in messages[0]
