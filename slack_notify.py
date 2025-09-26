from __future__ import annotations

"""Centralised emergency Slack helpers."""

from typing import Any, Callable, Dict, List, Optional

try:
    from auto_trading_bot.alerts import slack_notify_safely
except Exception:  # pragma: no cover - fallback when alerts unavailable during tests
    def slack_notify_safely(message: str, *, blocks: Optional[List[Dict[str, Any]]] = None) -> bool:  # type: ignore
        raise RuntimeError("slack_notify_safely is unavailable") from None


DEFAULT_PREFIX = ":rotating_light:"


def _format_message(message: str, prefix: str = DEFAULT_PREFIX) -> str:
    if not message:
        return prefix
    return message if message.startswith(prefix) else f"{prefix} {message}"


def notify_emergency(
    message: str,
    *,
    prefix: str = DEFAULT_PREFIX,
    blocks: Optional[List[Dict[str, Any]]] = None,
    sender: Callable[..., bool] = slack_notify_safely,
) -> bool:
    """Send a Slack emergency notification with a consistent prefix."""
    formatted = _format_message(message, prefix=prefix)
    try:
        return sender(formatted, blocks=blocks)
    except TypeError:
        return sender(formatted)
    except Exception:
        return False
