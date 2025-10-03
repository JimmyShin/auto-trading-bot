from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from config import (
    ATR_GATE,
    ATR_LEN,
    BAND_EXIT,
    DONCHIAN_LEN,
    MIDLINE_EXIT,
    REENTRY_COOLDOWN_BARS,
)
from auto_trading_bot.strategy.base import Signal, Strategy, register_strategy


@dataclass
class DonchianSnapshot:
    high: float
    low: float
    mid: float


@register_strategy("donchian_atr")
class DonchianATRStrategy(Strategy):
    """Donchian breakout filtered by ATR, with configurable exits and cooldown."""

    def __init__(self) -> None:
        self._cooldown_remaining: int = 0

    def on_bar(self, bar: Dict[str, Any], state: Dict[str, Any]) -> Signal:
        if not bar and not state:
            return Signal(side="neutral", strength=0.0, meta={"reason": "stub"})

        df = self._extract_history(bar)
        required = max(DONCHIAN_LEN, ATR_LEN)
        if df is None or len(df) < required:
            return Signal(side="neutral", strength=0.0, meta={"reason": "insufficient_history"})

        df = df.astype(float)
        snapshot = self._compute_donchian(df)
        atr_abs = self._compute_atr(df)

        if snapshot is None or atr_abs is None:
            return Signal(side="neutral", strength=0.0, meta={"reason": "insufficient_buffers"})

        close = float(df["close"].iloc[-1])
        atr_pct = atr_abs / close if close else 0.0

        gate = getattr(self, "_atr_gate", None)
        if gate is None:
            gate = ATR_GATE
        cooldown_max = getattr(self, "_rcd_max", None)
        if cooldown_max is None:
            cooldown_max = REENTRY_COOLDOWN_BARS

        state[f"donch_high_{DONCHIAN_LEN}"] = snapshot.high
        state[f"donch_low_{DONCHIAN_LEN}"] = snapshot.low
        state[f"donch_mid_{DONCHIAN_LEN}"] = snapshot.mid
        state[f"atr_{ATR_LEN}"] = atr_abs
        state["atr_pct"] = atr_pct
        state["price"] = close

        cooldown_left = max(self._cooldown_remaining, 0)
        side = "neutral"
        reasons: list[str] = []

        if cooldown_left > 0:
            reasons.append("cooldown_active")
            self._cooldown_remaining = cooldown_left - 1
        elif atr_pct < gate:
            reasons.append("atr_gate")
        else:
            if close > snapshot.high:
                side = "long"
            elif close < snapshot.low:
                side = "short"
            else:
                reasons.append("inside_channel")

        if side != "neutral" and cooldown_max > 0:
            self._cooldown_remaining = cooldown_max
            cooldown_left = cooldown_max
        elif side == "neutral" and cooldown_left <= 0:
            self._cooldown_remaining = 0
            cooldown_left = 0

        state["last_signal"] = side
        state["cooldown_left"] = max(self._cooldown_remaining, 0)

        exits = {
            "midline": snapshot.mid,
            "opp_band": snapshot.low if side == "long" else snapshot.high if side == "short" else snapshot.mid,
        }

        meta = {
            "dc_len": DONCHIAN_LEN,
            "atr_len": ATR_LEN,
            "atr_gate": gate,
            "atr_pct": atr_pct,
            "donch_high": snapshot.high,
            "donch_low": snapshot.low,
            "donch_mid": snapshot.mid,
            "cooldown_left": max(self._cooldown_remaining, 0),
            "exit_policy": {"midline": MIDLINE_EXIT, "band": BAND_EXIT},
            "sl_atr_k": 1.5,
            "reasons": reasons,
            "exits": exits,
        }

        return Signal(side=side, strength=1.0 if side != "neutral" else 0.0, meta=meta)

    def _extract_history(self, bar: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if not isinstance(bar, dict):
            return None
        history: Optional[Any] = None
        for key in ("history", "df", "data"):
            if key in bar:
                candidate = bar[key]
                if candidate is None:
                    continue
                history = candidate
                break
        if history is None:
            return None
        if isinstance(history, pd.DataFrame):
            required = {"open", "high", "low", "close"}
            if required.issubset(history.columns):
                return history.copy()
            return None
        if isinstance(history, Iterable):
            prices = list(history)
            if not prices:
                return None
            df = pd.DataFrame({
                "close": prices,
            })
            df["open"] = df["close"].shift(1, fill_value=df["close"].iloc[0])
            df["high"] = df[["open", "close"]].max(axis=1)
            df["low"] = df[["open", "close"]].min(axis=1)
            return df[["open", "high", "low", "close"]]
        return None

    def _compute_donchian(self, df: pd.DataFrame) -> Optional[DonchianSnapshot]:
        window = df.iloc[-DONCHIAN_LEN:]
        if len(window) < 2:
            return None
        channel = window.iloc[:-1]
        if channel.empty:
            return None
        high = float(channel["high"].max())
        low = float(channel["low"].min())
        mid = (high + low) / 2.0
        return DonchianSnapshot(high=high, low=low, mid=mid)

    def _compute_atr(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < 2:
            return None
        if len(df) >= ATR_LEN + 1:
            window = df.iloc[-(ATR_LEN + 1):]
        else:
            window = df.iloc[-(len(df)) :]
        high = window["high"]
        low = window["low"]
        close = window["close"]
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        window_len = min(ATR_LEN, len(tr) - 1) if len(tr) > 1 else 1
        atr_series = tr.tail(window_len) if window_len > 0 else tr.tail(1)
        return float(atr_series.mean()) if not atr_series.empty else None
