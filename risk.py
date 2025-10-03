from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import logging

import pandas as pd

from config import (
    ATR_STOP_K,
    ATR_TRAIL_K,
    DAILY_LOSS_LIMIT,
    ENABLE_PYRAMIDING,
    PYRAMID_LEVELS,
    RISK_PCT,
    get_position_cap,
)


logger = logging.getLogger(__name__)

__all__ = ["RiskManager"]


class RiskManager:
    """Encapsulates position sizing and position-state management."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self.bind_state(state or {})

    def bind_state(self, state: Dict[str, Any]) -> None:
        """Attach the mutable strategy state dictionary."""
        self.state = state

    # ------------------------------------------------------------------
    # Daily risk controls
    # ------------------------------------------------------------------
    def reset_daily_anchor(self, equity: float) -> bool:
        today = datetime.utcnow().date().isoformat()
        daily = self.state.get("daily")
        if not daily or daily.get("date") != today:
            self.state["daily"] = {"date": today, "anchor": float(equity)}
            return True
        return False

    def hit_daily_loss_limit(self, equity: float) -> bool:
        daily = self.state.get("daily")
        if not daily:
            return False
        anchor = float(daily.get("anchor", equity))
        drawdown = max(0.0, (anchor - equity) / max(anchor, 1e-9))
        return drawdown >= DAILY_LOSS_LIMIT

    # ------------------------------------------------------------------
    # Position sizing helpers
    # ------------------------------------------------------------------
    def calc_qty_by_risk(
        self,
        equity_usdt: float,
        price: float,
        atr_abs: float,
        leverage: int,
        symbol: str = "",
    ) -> float:
        return self.calc_qty_by_risk_adjusted(
            equity_usdt=equity_usdt,
            price=price,
            atr_abs=atr_abs,
            leverage=leverage,
            symbol=symbol,
            risk_pct=RISK_PCT,
        )

    def calc_qty_by_risk_adjusted(
        self,
        equity_usdt: float,
        price: float,
        atr_abs: float,
        leverage: int,
        symbol: str = "",
        risk_pct: Optional[float] = None,
    ) -> float:
        if risk_pct is None:
            risk_pct = RISK_PCT

        risk_quote = equity_usdt * risk_pct
        stop_dist = max(ATR_STOP_K * atr_abs, price * 0.004)
        qty = risk_quote / stop_dist if stop_dist > 0 else 0.0

        min_qty = 0.001
        min_notional = 20.0
        min_qty_by_notional = min_notional / max(price, 1e-9)

        symbol_upper = symbol.upper()
        if "ETH" in symbol_upper:
            min_qty = 0.01
        elif "SOL" in symbol_upper:
            min_qty = 0.01
        elif "BNB" in symbol_upper:
            min_qty = 0.01
        elif "AVAX" in symbol_upper:
            min_qty = 0.1
        elif "ADA" in symbol_upper:
            min_qty = 1.0

        final_min_qty = max(min_qty, min_qty_by_notional)
        if 0 < qty < final_min_qty:
            logger.info(
                "Risk qty below exchange minimum; adjusting",
                extra={
                    "event": "risk_qty_floor",
                    "symbol": symbol,
                    "qty": float(qty),
                    "min_qty": float(final_min_qty),
                },
            )
            qty = final_min_qty

        position_cap = get_position_cap(equity_usdt)
        max_qty_by_cap = position_cap / max(price, 1e-9)
        if qty > max_qty_by_cap:
            logger.info(
                "Risk qty exceeds position cap; trimming",
                extra={
                    "event": "risk_qty_cap",
                    "symbol": symbol,
                    "qty": float(qty),
                    "position_cap": float(position_cap),
                    "trimmed_qty": float(max_qty_by_cap),
                },
            )
            qty = max_qty_by_cap

        return max(qty, 0.0)

    def trail_stop_price(
        self,
        side: str,
        entry_price: float,
        last_price: float,
        atr_abs: float,
        be_promoted: bool,
    ) -> float:
        if side == "long":
            base = last_price - ATR_TRAIL_K * atr_abs
            be_price = entry_price
            return max(base, be_price if be_promoted else -1e18)
        base = last_price + ATR_TRAIL_K * atr_abs
        be_price = entry_price
        return min(base, be_price if be_promoted else 1e18)

    # ------------------------------------------------------------------
    # Position state mutations
    # ------------------------------------------------------------------
    def update_symbol_state_on_entry(
        self,
        symbol: str,
        side: str,
        entry_px: float,
        qty: float = 0,
        entry_stop_price: Optional[float] = None,
        risk_usdt: float = 0.0,
    ) -> Dict[str, Any]:
        symbol_state = self.state.get(symbol, {})
        symbol_state.update(
            {
                "in_position": True,
                "side": side,
                "entry_price": float(entry_px),
                "entry_time": datetime.utcnow().isoformat(),
                "be_promoted": False,
                "original_qty": float(qty),
                "pyramid_level": 0,
                "pyramid_added": [],
                "pyramid_locked_limit": {},
            }
        )
        if entry_stop_price is not None:
            try:
                symbol_state["entry_stop_price"] = float(entry_stop_price)
            except Exception:
                pass
        try:
            symbol_state["risk_usdt"] = float(risk_usdt or 0.0)
        except Exception:
            symbol_state["risk_usdt"] = 0.0
        self.state[symbol] = symbol_state
        return symbol_state

    def update_after_move(self, symbol: str, atr_abs: float, last_price: float) -> Optional[Dict[str, Any]]:
        symbol_state = self.state.get(symbol)
        if not symbol_state or not symbol_state.get("in_position"):
            return None
        if atr_abs <= 0:
            return symbol_state

        side = symbol_state["side"]
        entry_price = float(symbol_state["entry_price"])

        if side == "long":
            profit_r = (last_price - entry_price) / max(atr_abs, 1e-9)
            if last_price - entry_price >= atr_abs * 1.5:
                symbol_state["be_promoted"] = True
        else:
            profit_r = (entry_price - last_price) / max(atr_abs, 1e-9)
            if entry_price - last_price >= atr_abs * 1.5:
                symbol_state["be_promoted"] = True

        if ENABLE_PYRAMIDING:
            current_level = int(symbol_state.get("pyramid_level", 0))
            if current_level < len(PYRAMID_LEVELS):
                target_r, add_ratio = PYRAMID_LEVELS[current_level]
                if profit_r >= target_r:
                    symbol_state["pyramid_level"] = current_level + 1
                    symbol_state["pyramid_trigger_r"] = profit_r
                    symbol_state["pyramid_locked_limit"] = self.calc_locked_profit_pyramid_limit(
                        symbol,
                        atr_abs,
                        last_price,
                        add_ratio,
                    )
                    logger.info(
                        "Pyramid trigger reached",
                        extra={
                            "event": "pyramid_trigger",
                            "symbol": symbol,
                            "profit_r": float(profit_r),
                            "level": int(symbol_state.get("pyramid_level", 0)),
                        },
                    )

        self.state[symbol] = symbol_state
        return symbol_state

    def calc_locked_profit_pyramid_limit(
        self,
        symbol: str,
        atr_abs: float,
        current_price: float,
        add_ratio: float,
    ) -> Dict[str, float]:
        symbol_state = self.state.get(symbol, {})
        side = symbol_state.get("side")
        entry_price = float(symbol_state.get("entry_price", current_price))
        original_qty = float(symbol_state.get("original_qty", 0.0))
        be_promoted = bool(symbol_state.get("be_promoted", False))

        trail_stop = self.trail_stop_price(side, entry_price, current_price, atr_abs, be_promoted)

        if side == "long":
            locked_profit_per_unit = max(0.0, trail_stop - entry_price)
        else:
            locked_profit_per_unit = max(0.0, entry_price - trail_stop)

        total_locked_profit = locked_profit_per_unit * original_qty
        traditional_add_qty = original_qty * add_ratio

        if total_locked_profit > 0:
            max_giveback = total_locked_profit * 0.5
            safe_add_qty = max_giveback / max(current_price, 1e-9)
        else:
            safe_add_qty = 0.0

        final_qty = min(traditional_add_qty, safe_add_qty)

        return {
            "locked_profit": total_locked_profit,
            "traditional_qty": traditional_add_qty,
            "safe_qty": safe_add_qty,
            "final_qty": final_qty,
        }

    def clear_position_state(self, symbol: str) -> Dict[str, Any]:
        symbol_state = self.state.get(symbol, {})
        symbol_state["in_position"] = False
        self.state[symbol] = symbol_state
        return symbol_state

    def record_stop_loss_exit(self, symbol: str, side: str, exit_price: float) -> Dict[str, Any]:
        symbol_state = self.state.get(symbol, {})
        symbol_state.update(
            {
                "in_position": False,
                "last_stop_loss": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "side": side,
                    "exit_price": float(exit_price),
                    "bars_since_stop": 0,
                    "reset_qualified": False,
                },
            }
        )
        self.state[symbol] = symbol_state
        return symbol_state

    # ------------------------------------------------------------------
    # Structural reset logic
    # ------------------------------------------------------------------
    def update_reset_tracking(self, symbol: str, df: pd.DataFrame) -> bool:
        symbol_state = self.state.get(symbol, {})
        last_stop = symbol_state.get("last_stop_loss")
        if not last_stop or last_stop.get("reset_qualified"):
            return False

        last_stop["bars_since_stop"] = int(last_stop.get("bars_since_stop", 0)) + 1
        reset_conditions = self.check_reset_conditions(symbol, df, last_stop)
        if reset_conditions["qualified"]:
            last_stop["reset_qualified"] = True
            logger.info(
                "Structural reset qualified",
                extra={
                    "event": "structural_reset",
                    "symbol": symbol,
                    "reasons": list(reset_conditions['reasons']),
                },
            )

        symbol_state["last_stop_loss"] = last_stop
        self.state[symbol] = symbol_state
        return True

    def check_reset_conditions(self, symbol: str, df: pd.DataFrame, last_stop: Dict[str, Any]) -> Dict[str, Any]:
        if len(df) < 25:
            return {"qualified": False, "reasons": []}

        from indicators import sma  # local import to avoid circular dependencies

        fast_ma = sma(df, 5)
        slow_ma = sma(df, 20)

        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]

        reasons = []
        side = last_stop.get("side", "")
        bars_since = int(last_stop.get("bars_since_stop", 0))

        if bars_since >= 2:
            reasons.append(f"min_bars({bars_since})")

        if side == "long":
            cooling_occurred = self.check_cooling_period(df, "long", bars_since)
            new_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
        else:
            cooling_occurred = self.check_cooling_period(df, "short", bars_since)
            new_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

        if cooling_occurred:
            reasons.append("cooling_done")
        if new_cross:
            reasons.append("new_cross")

        if self.check_ma_retest(df, side):
            reasons.append("retest_ok")

        qualified = bars_since >= 2 and (cooling_occurred or new_cross) and len(reasons) >= 2
        return {"qualified": qualified, "reasons": reasons}

    def check_cooling_period(self, df: pd.DataFrame, original_side: str, bars_back: int) -> bool:
        if len(df) < bars_back + 5:
            return False

        from indicators import sma

        fast_ma = sma(df, 5)
        slow_ma = sma(df, 20)

        for i in range(min(bars_back, 5)):
            idx = -(i + 1)
            fast_val = fast_ma.iloc[idx]
            slow_val = slow_ma.iloc[idx]
            if original_side == "long" and fast_val < slow_val:
                return True
            if original_side == "short" and fast_val > slow_val:
                return True
        return False

    def check_ma_retest(self, df: pd.DataFrame, side: str) -> bool:
        if len(df) < 5:
            return True

        from indicators import sma

        ma20 = sma(df, 20)
        closes = df["close"]

        for i in range(1, min(4, len(df))):
            close_val = closes.iloc[-i]
            ma_val = ma20.iloc[-i]
            distance_pct = abs(close_val - ma_val) / max(ma_val, 1e-9)
            if distance_pct < 0.005:
                return True
        return True

    def can_re_enter_after_stop(self, symbol: str) -> bool:
        symbol_state = self.state.get(symbol, {})
        last_stop = symbol_state.get("last_stop_loss")
        if not last_stop:
            return True
        return bool(last_stop.get("reset_qualified", False))

    # ------------------------------------------------------------------
    # Broker synchronisation
    # ------------------------------------------------------------------
    def sync_position_state(self, broker: Any, symbol: str) -> Optional[bool]:
        try:
            actual_size = broker.get_actual_position_size(symbol)
            has_actual_position = abs(float(actual_size)) > 1e-8

            symbol_state = self.state.get(symbol, {})
            state_has_position = symbol_state.get("in_position", False)

            if has_actual_position and not state_has_position:
                symbol_state.update(
                    {
                        "in_position": True,
                        "side": "long" if actual_size > 0 else "short",
                        "original_qty": abs(float(actual_size)),
                    }
                )
                self.state[symbol] = symbol_state
                return True

            if not has_actual_position and state_has_position:
                return False

            return None
        except Exception as exc:
            logger.warning(
                "Failed to sync position state",
                extra={
                    "event": "sync_position_state_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            return None
