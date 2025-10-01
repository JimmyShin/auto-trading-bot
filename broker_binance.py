"""Binance USDⓈ-M client adapter backed by ccxt.

This module provides a thin wrapper used by ``auto_trading_bot.ExchangeAPI`` so the
rest of the codebase does not depend directly on ccxt.  It replaces the legacy
in-memory stub so production builds can talk to Binance futures endpoints.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

import ccxt  # type: ignore

logger = logging.getLogger(__name__)


class BalanceAuthError(Exception):
    """Raised when the exchange rejects the API credentials."""


class BalanceSyncError(Exception):
    """Raised when client/server clocks drift beyond an acceptable threshold."""


class BinanceUSDM:
    """ccxt-backed adapter exposing methods required by ExchangeAPI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = bool(testnet)
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if self.testnet:
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception as exc:  # pragma: no cover - best effort only
                logger.warning("Failed to enable sandbox mode: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def set_testnet(self, flag: bool) -> bool:
        self.testnet = bool(flag)
        try:
            self.exchange.set_sandbox_mode(self.testnet)
        except Exception as exc:  # pragma: no cover
            logger.warning("set_sandbox_mode failed: %s", exc)
        return True

    def load_markets(self) -> None:  # pragma: no cover - ccxt handles caching
        self.exchange.load_markets()

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Iterable[Iterable[Any]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    def fapiPrivateGetBalance(self) -> List[Dict[str, Any]]:  # pragma: no cover - passthrough
        return self.exchange.fapiPrivateV2GetBalance()

    def fapi_private_get_balance(self) -> List[Dict[str, Any]]:  # ccxt method alias
        return self.fapiPrivateGetBalance()

    def fapiPrivateGetAccount(self) -> Dict[str, Any]:  # pragma: no cover
        return self.exchange.fapiPrivateV2GetAccount()

    def fapi_private_get_account(self) -> Dict[str, Any]:  # alias for compatibility
        return self.fapiPrivateGetAccount()

    def fapiPrivateGetPositionRisk(self) -> List[Dict[str, Any]]:  # pragma: no cover
        return self.exchange.fapiPrivateV2GetPositionRisk()

    # ------------------------------------------------------------------
    # Account state API (mirrors methods used by ExchangeAPI)
    # ------------------------------------------------------------------
    def get_equity_usdt(self) -> float:
        return float(self.exchange.fetch_balance().get("total", {}).get("USDT", 0.0))

    def get_equity(self) -> float:
        return self.get_equity_usdt()

    def set_equity(self, equity: float) -> None:  # pragma: no cover - not supported via API
        logger.debug("set_equity called with %s; ignoring (not supported)", equity)

    def set_balance(self, wallet_balance: float, unrealized: float) -> None:  # pragma: no cover
        logger.debug(
            "set_balance called with wallet=%s unrealized=%s; ignoring (not supported)",
            wallet_balance,
            unrealized,
        )

    def get_balance_snapshot(self) -> Dict[str, Any]:
        balance = self.exchange.fapiPrivateGetAccount()
        now_ms = int(time.time() * 1000)
        return {
            "totalWalletBalance": float(balance.get("totalWalletBalance", 0.0)),
            "totalUnrealizedProfit": float(balance.get("totalUnrealizedProfit", 0.0)),
            "totalMarginBalance": float(balance.get("totalMarginBalance", 0.0)),
            "updateTime": int(balance.get("updateTime", now_ms)),
        }

    def position_for(self, symbol: str) -> Dict[str, Any]:
        positions = self.exchange.fapiPrivateGetPositionRisk()
        for pos in positions:
            if pos.get("symbol") == symbol.replace("/", ""):
                return pos
        return {}

    def fetch_positions(self) -> List[Dict[str, Any]]:
        return self.exchange.fetch_positions()

    def get_actual_position_size(self, symbol: str) -> float:
        pos = self.position_for(symbol)
        try:
            return float(pos.get("positionAmt", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def remaining_addable_qty_under_risk_limit(self, symbol: str) -> float:
        return float("inf")  # Binance exposes limits but ccxt does not map them 1:1

    def close_all_positions(self) -> None:
        self.exchange.cancel_all_orders(symbol=None)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------
    def cancel_all(self, symbol: str) -> None:
        self.exchange.cancel_all_orders(symbol)

    def create_market_order_safe(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        order = self.exchange.create_order(symbol, "market", side.lower(), qty)
        return order

    def create_stop_market_safe(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        qty: float,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:
        params = {"stopPrice": stop_price, "reduceOnly": reduce_only}
        order = self.exchange.create_order(symbol, "stop_market", side.lower(), qty, params=params)
        return order

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def fetch_my_trades(self, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self.exchange.fetch_my_trades(symbol, limit=limit)
