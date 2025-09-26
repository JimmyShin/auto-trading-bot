"""Test stub for Binance USDM client.

This module provides a deterministic in-memory replacement for the real
``broker_binance`` dependency so unit tests can exercise
``auto_trading_bot.exchange_api`` without touching the network.  It is *not*
intended for production use.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class BalanceAuthError(Exception):
    """Raised when the exchange rejects the API credentials."""


class BalanceSyncError(Exception):
    """Raised when client/server clocks drift beyond an acceptable threshold."""


class _MockExchange:
    """Lightweight helper exposing ``fetch_my_trades`` like ccxt does."""

    def __init__(self, parent: "BinanceUSDM") -> None:
        self._parent = parent

    def fetch_my_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Tests only assert that the method exists. Return an empty list.
        return []


class BinanceUSDM:
    """Very small in-memory stand-in for the real Binance futures client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = bool(testnet)
        self._equity = 10_000.0
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._prices: Dict[str, float] = {}
        self.exchange = _MockExchange(self)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def set_testnet(self, flag: bool) -> bool:
        self.testnet = bool(flag)
        return True

    def load_markets(self) -> None:  # pragma: no cover - no external state
        return None

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Iterable[Iterable[Any]]:
        price = self._prices.get(symbol, 100.0)
        candles = []
        for i in range(limit):
            ts = 1_600_000_000_000 + i * 60_000
            base = price + i * 0.25
            candles.append([ts, base, base + 1.0, base - 1.0, base + 0.5, 10.0])
        return candles

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        price = self._prices.get(symbol, 100.0)
        return {"symbol": symbol, "last": price, "close": price, "bid": price, "ask": price}

    # ------------------------------------------------------------------
    # Account state API (mirrors methods used by ExchangeAPI)
    # ------------------------------------------------------------------
    def get_equity_usdt(self) -> float:
        return float(self._equity)

    # Convenience aliases required by the prompt
    def get_equity(self) -> float:
        return self.get_equity_usdt()

    def set_equity(self, equity: float) -> None:
        self._equity = float(equity)

    def position_for(self, symbol: str) -> Dict[str, Any]:
        return dict(self._positions.get(symbol, {}))

    def fetch_positions(self) -> List[Dict[str, Any]]:
        return [dict(v, symbol=s) for s, v in self._positions.items()]

    def get_actual_position_size(self, symbol: str) -> float:
        return float(self._positions.get(symbol, {}).get("positionAmt", 0.0))

    def remaining_addable_qty_under_risk_limit(self, symbol: str) -> float:
        # No risk limits in the stub environment.
        return float("inf")

    def close_all_positions(self) -> None:
        for pos in self._positions.values():
            pos["positionAmt"] = 0.0

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    def cancel_all(self, symbol: str) -> None:  # pragma: no cover - no queued orders
        return None

    def create_market_order_safe(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        qty = float(qty)
        side = side.lower()
        state = self._positions.setdefault(symbol, {"positionAmt": 0.0, "entryPrice": self._prices.get(symbol, 100.0)})
        amount = float(state.get("positionAmt", 0.0))
        if side == "buy":
            amount += qty
        else:
            amount -= qty
        state["positionAmt"] = amount
        return {"symbol": symbol, "side": side, "amount": qty, "filled": qty}

    def create_stop_market_safe(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: float,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:  # pragma: no cover - used only for interface compatibility
        return {
            "symbol": symbol,
            "side": side,
            "stopPrice": stop_price,
            "amount": qty,
            "reduce_only": reduce_only,
        }

    # ------------------------------------------------------------------
    # Misc helpers / utilities for tests
    # ------------------------------------------------------------------
    def fetch_my_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self.exchange.fetch_my_trades(symbol, limit=limit)

    # Simple mutators used by tests -------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def set_position(self, symbol: str, amount: float, entry_price: Optional[float] = None) -> None:
        state = self._positions.setdefault(symbol, {})
        state["positionAmt"] = float(amount)
        if entry_price is not None:
            state["entryPrice"] = float(entry_price)
