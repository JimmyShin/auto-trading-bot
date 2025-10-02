import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from auto_trading_bot.restart import mark_restart_intent, peek_restart_intent, consume_restart_intent


def test_mark_and_consume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "intent.json"
    monkeypatch.setenv("RESTART_INTENT_PATH", str(state_file))

    intent = mark_restart_intent(mode="graceful", ttl_seconds=60)
    assert state_file.exists()

    peeked = peek_restart_intent(ttl_seconds=60)
    assert peeked is not None
    assert abs((peeked.created_at - intent.created_at).total_seconds()) < 1

    consumed = consume_restart_intent(ttl_seconds=60)
    assert consumed is not None
    assert state_file.exists() is False


def test_expired_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "intent.json"
    monkeypatch.setenv("RESTART_INTENT_PATH", str(state_file))

    intent = mark_restart_intent(mode="graceful", ttl_seconds=1)
    # manually rewind created_at to simulate expiry
    payload = {
        "created_at": (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat(),
        "mode": intent.mode,
    }
    state_file.write_text(json.dumps(payload), encoding="utf-8")

    assert peek_restart_intent(ttl_seconds=10) is None
    assert consume_restart_intent(ttl_seconds=10) is None
    assert state_file.exists() is False
