from __future__ import annotations

from typing import Any, Dict

from auto_trading_bot.strategy.base import Signal, Strategy, register_strategy


@register_strategy("donchian_atr")
class DonchianATRStrategy(Strategy):
    """Placeholder Donchian+ATR strategy hook returning neutral signals."""

    def on_bar(self, bar: Dict[str, Any], state: Dict[str, Any]) -> Signal:
        state.setdefault("last_signal", "neutral")
        return Signal(side="neutral", strength=0.0, meta={"reason": "stub"})
