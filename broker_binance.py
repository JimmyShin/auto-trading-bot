"""Minimal broker_binance stubs for tests.

These stubs provide the small surface area that auto_trading_bot.exchange_api
expects, without requiring the full ccxt dependency or remote API calls.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class BalanceAuthError(Exception):
    """Raised when authentication with the exchange fails."""


class BalanceSyncError(Exception):
    """Raised when time sync or balance retrieval fails."""


class _MockExchange:
    """Lightweight object exposing fetch_my_trades for compatibility."""

    def __init__(self, parent: "BinanceUSDM") -> None:
        self._parent = parent

    def fetch_my_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Integration tests only assert that the method exists; return empty history.
        return []


class BinanceUSDM:
    """Very small stand-in for the real Binance USDM client.

    The implementation keeps in-memory state so that unit tests can interact with
    ExchangeAPI without talking to the network. Only the methods that the test
    suite exercises are implemented here.
    """

    def __init__(self) -> None:
        self._equity = 10_000.0
        self._prices: Dict[str, float] = {}
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._testnet_enabled = True
        self.exchange = _MockExchange(self)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def load_markets(self) -> None:  # pragma: no cover - no-op
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Iterable[Iterable[Any]]:
        # Produce deterministic candles for tests.
        candles = []
        price = self._prices.get(symbol, 100.0)
        for i in range(limit):
            ts = 1_600_000_000_000 + i * 60_000
            open_ = price + i * 0.1
            high = open_ + 1.0
            low = open_ - 1.0
            close = open_ + 0.2
            volume = 10.0
            candles.append([ts, open_, high, low, close, volume])
        return candles

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        price = self._prices.get(symbol, 100.0)
        return {"symbol": symbol, "last": price, "close": price, "bid": price, "ask": price}

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------
    def get_equity_usdt(self) -> float:
        return float(self._equity)

    def position_for(self, symbol: str) -> Dict[str, Any]:
        return dict(self._positions.get(symbol, {}))

    def get_actual_position_size(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0
        try:
            return float(pos.get("positionAmt") or pos.get("contracts") or 0.0)
        except Exception:
            return 0.0

    def remaining_addable_qty_under_risk_limit(self, symbol: str) -> float:
        # No risk limits in the stubbed environment.
        return float("inf")

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    def cancel_all(self, symbol: str) -> None:  # pragma: no cover - no-op
        return None

    def create_market_order_safe(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        qty = float(qty)
        side = side.lower()
        current = self._positions.setdefault(symbol, {"positionAmt": 0.0, "entryPrice": self._prices.get(symbol, 100.0)})
        amt = float(current.get("positionAmt") or 0.0)
        if side == "buy":
            amt += qty
        else:
            amt -= qty
        current["positionAmt"] = amt
        return {"symbol": symbol, "side": side, "amount": qty, "filled": qty}

    def create_stop_market_safe(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: float,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:  # pragma: no cover - behaviour not required in tests
        return {
            "symbol": symbol,
            "side": side,
            "stopPrice": stop_price,
            "amount": qty,
            "reduce_only": reduce_only,
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def set_testnet(self, enabled: bool) -> bool:
        self._testnet_enabled = bool(enabled)
        return True

    # Utilities for tests -------------------------------------------------
    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def set_position(self, symbol: str, amount: float, entry_price: Optional[float] = None) -> None:
        entry = self._positions.setdefault(symbol, {})
        entry["positionAmt"] = float(amount)
        if entry_price is not None:
            entry["entryPrice"] = float(entry_price)

    def set_equity(self, equity: float) -> None:
        self._equity = float(equity)
