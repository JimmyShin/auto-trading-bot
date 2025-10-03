from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from indicators import ma_crossover_signal
from auto_trading_bot.strategy.base import Signal, Strategy, register_strategy
from auto_trading_bot.strategy.utils import strategy_tags


@register_strategy("ma_cross_5_20")
class MovingAverageCrossStrategy(Strategy):
    """Simple 5/20 MA crossover strategy registered via the strategy registry."""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("MA periods must be positive integers")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be shorter than slow period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, bar: Dict[str, Any], state: Dict[str, Any]) -> Signal:
        history = self._extract_history(bar)
        params = {"fast_period": self.fast_period, "slow_period": self.slow_period}
        tags = strategy_tags("ma_cross_5_20", "1", params)
        if history is None:
            meta = {"reason": "missing_history"}
            meta.update(tags)
            return Signal(side="neutral", strength=0.0, meta=meta)
        if len(history) < self.slow_period:
            meta = {"reason": "insufficient_history"}
            meta.update(tags)
            return Signal(side="neutral", strength=0.0, meta=meta)

        analysis = ma_crossover_signal(history, fast_period=self.fast_period, slow_period=self.slow_period)

        if analysis.get("long"):
            state["ma_cross_last"] = {"analysis": analysis, "side": "long"}
            meta = {"analysis": analysis}
            meta.update(tags)
            return Signal(side="long", strength=1.0, meta=meta)
        if analysis.get("short"):
            state["ma_cross_last"] = {"analysis": analysis, "side": "short"}
            meta = {"analysis": analysis}
            meta.update(tags)
            return Signal(side="short", strength=1.0, meta=meta)

        state["ma_cross_last"] = {"analysis": analysis, "side": "neutral"}
        meta = {"analysis": analysis, "reason": "no_signal"}
        meta.update(tags)
        return Signal(side="neutral", strength=0.0, meta=meta)

    @staticmethod
    def _extract_history(bar: Dict[str, Any]) -> pd.DataFrame | None:
        if not isinstance(bar, dict):
            return None

        history = None
        for key in ("history", "df", "data"):
            candidate = bar.get(key)
            if isinstance(candidate, pd.DataFrame):
                history = candidate
                break

        if history is None:
            return None

        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(history.columns):
            return None
        return history
