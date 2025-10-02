import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_DEFAULT_TTL_SEC = 600


def _state_path() -> Path:
    return Path(os.getenv("RESTART_INTENT_PATH", "data/restart_intent.json"))


@dataclass
class RestartIntent:
    created_at: datetime
    mode: str = "graceful"

    def is_valid(self, ttl_seconds: int = _DEFAULT_TTL_SEC) -> bool:
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() < ttl_seconds


def _read_state() -> Optional[RestartIntent]:
    path = _state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        created_str = raw.get("created_at")
        mode = raw.get("mode", "graceful")
        if not created_str:
            return None
        created = datetime.fromisoformat(created_str)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        else:
            created = created.astimezone(timezone.utc)
        return RestartIntent(created_at=created, mode=mode)
    except Exception:
        return None


def mark_restart_intent(mode: str = "graceful", *, ttl_seconds: int = _DEFAULT_TTL_SEC) -> RestartIntent:
    intent = RestartIntent(datetime.now(timezone.utc), mode=mode)
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": intent.created_at.isoformat(), "mode": mode, "ttl": ttl_seconds}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return intent


def peek_restart_intent(ttl_seconds: int = _DEFAULT_TTL_SEC) -> Optional[RestartIntent]:
    path = _state_path()
    intent = _read_state()
    if intent and intent.is_valid(ttl_seconds):
        return intent
    if intent:
        try:
            path.unlink()
        except Exception:
            pass
    return None


def consume_restart_intent(ttl_seconds: int = _DEFAULT_TTL_SEC) -> Optional[RestartIntent]:
    path = _state_path()
    intent = _read_state()
    if intent is None or not intent.is_valid(ttl_seconds):
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return None
    try:
        path.unlink()
    except Exception:
        pass
    return intent


def _run_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage graceful restart intent")
    parser.add_argument("action", choices=["mark", "consume", "peek"], help="Action to perform")
    parser.add_argument("--mode", default="graceful", help="Mode label to store (default: graceful)")
    parser.add_argument("--ttl", type=int, default=_DEFAULT_TTL_SEC, help="TTL seconds for intent")
    args = parser.parse_args()

    if args.action == "mark":
        intent = mark_restart_intent(args.mode, ttl_seconds=args.ttl)
        print(json.dumps({"created_at": intent.created_at.isoformat(), "mode": intent.mode}))
        return 0
    if args.action == "consume":
        intent = consume_restart_intent(ttl_seconds=args.ttl)
        if intent is None:
            print("null")
            return 1
        print(json.dumps({"created_at": intent.created_at.isoformat(), "mode": intent.mode}))
        return 0
    intent = peek_restart_intent(ttl_seconds=args.ttl)
    if intent is None:
        print("null")
        return 1
    print(json.dumps({"created_at": intent.created_at.isoformat(), "mode": intent.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
