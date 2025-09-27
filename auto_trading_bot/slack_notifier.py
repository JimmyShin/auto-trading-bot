from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional


class SlackNotifier:
    """Minimal Slack webhook adapter (dry-run friendly)."""

    def __init__(self, *, logger_name: str = __name__) -> None:
        self._logger = logging.getLogger(logger_name)
        self._url = os.environ.get("SLACK_WEBHOOK_URL")
        dry_env = os.environ.get("SLACK_DRY_RUN", "false").strip().lower()
        self._dry = dry_env == "true" or not self._url

    def _dispatch(self, payload: Dict[str, Any]) -> bool:
        if self._dry:
            self._logger.info(json.dumps({"type": "SLACK_DRY_RUN", "payload": payload}, sort_keys=True))
            return True
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            return True
        except Exception as exc:  # pragma: no cover - network failure surfaces in logs
            self._logger.warning("Slack webhook failed: %s", exc)
            return False

    def send_markdown(self, text: str) -> bool:
        payload = {"text": text}
        return self._dispatch(payload)

    def send(self, text: str, *, blocks: Optional[list] = None) -> bool:
        if blocks:
            payload: Dict[str, Any] = {"text": text, "blocks": blocks}
            return self._dispatch(payload)
        return self.send_markdown(text)
