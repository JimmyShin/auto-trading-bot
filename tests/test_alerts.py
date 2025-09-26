import os
from types import SimpleNamespace

from auto_trading_bot import alerts


class DummyResp:
    def __init__(self, status):
        self.status_code = status


def test_slack_notify_success(monkeypatch):
    os.environ["SLACK_WEBHOOK_URL"] = "https://example.com/hook"
    calls = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        calls["called"] = True
        assert url == os.environ["SLACK_WEBHOOK_URL"]
        assert "application/json" in headers.get("Content-Type", "")
        return DummyResp(200)

    monkeypatch.setattr(alerts.requests, "post", fake_post)
    try:
        ok = alerts.slack_notify_safely("hello")
        assert ok is True
        assert calls.get("called") is True
    finally:
        os.environ.pop("SLACK_WEBHOOK_URL", None)


def test_slack_notify_failure(monkeypatch):
    os.environ["SLACK_WEBHOOK_URL"] = "https://example.com/hook"

    def fake_post(url, data=None, headers=None, timeout=None):
        return DummyResp(500)

    monkeypatch.setattr(alerts.requests, "post", fake_post)
    try:
        ok = alerts.slack_notify_safely("hello")
        assert ok is False
    finally:
        os.environ.pop("SLACK_WEBHOOK_URL", None)


