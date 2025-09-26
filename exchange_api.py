from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from broker_binance import BinanceUSDM, BalanceAuthError, BalanceSyncError


__all__ = [
    "ExchangeAPI",
    "BalanceAuthError",
    "BalanceSyncError",
]


class ExchangeAPI:
    """Thin wrapper around the BinanceUSDM client to ease dependency injection."""

    def __init__(self, client: Optional[BinanceUSDM] = None, auto_connect: bool = False) -> None:
        self.client = client or BinanceUSDM()
        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def connect(self) -> None:
        self.client.load_markets()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Iterable[Iterable[Any]]:
        return self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.client.fetch_ticker(symbol)

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------
    def get_equity_usdt(self) -> float:
        return float(self.client.get_equity_usdt())

    def position_for(self, symbol: str) -> Dict[str, Any]:
        return self.client.position_for(symbol)

    def get_actual_position_size(self, symbol: str) -> float:
        return float(self.client.get_actual_position_size(symbol))

    def remaining_addable_qty_under_risk_limit(self, symbol: str) -> float:
        return float(self.client.remaining_addable_qty_under_risk_limit(symbol))

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    def cancel_all(self, symbol: str) -> None:
        self.client.cancel_all(symbol)

    def create_market_order_safe(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        return self.client.create_market_order_safe(symbol, side, qty)

    def create_stop_market_safe(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: float,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:
        return self.client.create_stop_market_safe(symbol, side, stop_price, qty, reduce_only=reduce_only)

    def fetch_my_trades(self, symbol: str, limit: int = 5):
        return self.client.exchange.fetch_my_trades(symbol, limit=limit)

    def set_testnet(self, enabled: bool) -> bool:
        """Toggle Binance futures testnet mode via underlying client."""
        try:
            return bool(self.client.set_testnet(bool(enabled)))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Direct access helpers
    # ------------------------------------------------------------------
    @property
    def raw(self) -> BinanceUSDM:
        return self.client
