from __future__ import annotations

import os
import json
from typing import Optional

import requests


class SlackNotifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = (webhook_url or "").strip()

    @classmethod
    def from_env(cls) -> "SlackNotifier":
        return cls(os.getenv("SLACK_WEBHOOK_URL"))

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, text: str) -> bool:
        if not self.enabled():
            return False
        try:
            payload = {"text": str(text)}
            resp = requests.post(self.webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception as e:
            # Do not raise; caller must not crash bot on alert failure
            try:
                print(f"[WARN] Slack notify failed: {e}")
            except Exception:
                pass
            return False


_SLACK = None  # lazy singleton


def _get_notifier() -> SlackNotifier:
    global _SLACK
    if _SLACK is None:
        _SLACK = SlackNotifier.from_env()
    return _SLACK


def slack_notify_safely(message: str) -> bool:
    """Send a Slack message if configured. Never raises; returns True on success."""
    n = _get_notifier()
    return n.send(message)

