from __future__ import annotations



import logging
import os
import json

import math

import time



from typing import Any, Dict, Iterable, List, Optional



from broker_binance import BinanceUSDM, BalanceAuthError, BalanceSyncError





__all__ = [

    "ExchangeAPI",

    "BalanceAuthError",

    "BalanceSyncError",

]





logger = logging.getLogger(__name__)

_DEBUG_VALUES = {"1", "true", "yes", "on"}

def _debug_enabled() -> bool:
    return os.getenv("OBS_DEBUG_ALERTS", "").strip().lower() in _DEBUG_VALUES

def _log_equity_fetch(mode: str, equity: float) -> None:
    if not _debug_enabled():
        return
    try:
        import config as cfg
        acct = getattr(cfg, "OBS_ACCOUNT_LABEL", "unknown")
    except Exception:
        acct = "unknown"
    payload = {"acct": acct, "mode": mode, "equity": equity, "ts": time.time()}
    try:
        logger.info("EQUITY_FETCH %s", json.dumps(payload, sort_keys=True))
    except Exception:
        logger.info("EQUITY_FETCH %s", payload)




POSITION_EPS = 1e-8



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
        value = float(self.client.get_equity_usdt())
        mode = "testnet" if getattr(self.client, "testnet", False) else "live"
        _log_equity_fetch(mode, value)
        return value
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



    @staticmethod

    def _extract_position_amount(position: Dict[str, Any]) -> float:

        if not position:

            return 0.0

        for key in ("positionAmt", "contracts", "amount", "size"):

            value = position.get(key)

            if value in (None, ""):

                continue

            try:

                return float(value)

            except (TypeError, ValueError):

                continue

        return 0.0



    def flatten_all(

        self,

        symbols: Iterable[str],

        *,

        retries: int = 0,

        backoff_sec: float = 1.0,

    ) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        for symbol in symbols:

            outcome = self._flatten_symbol(symbol, retries=retries, backoff_sec=backoff_sec)

            results.append(outcome)

        return results



    def _flatten_symbol(

        self,

        symbol: str,

        *,

        retries: int,

        backoff_sec: float,

    ) -> Dict[str, Any]:

        attempts = 0

        last_error: Optional[str] = None

        executed_qty = 0.0



        try:

            position = self.position_for(symbol) or {}

        except Exception as exc:

            reason = f"position_fetch:{exc}"

            logger.warning("Failed to fetch position for %s during flatten attempt: %s", symbol, exc)

            return {

                "symbol": symbol,

                "status": "error",

                "attempts": 0,

                "remaining_qty": None,

                "order_side": None,

                "order_qty": 0.0,

                "reason": reason,

            }



        amt_signed = self._extract_position_amount(position)

        if abs(amt_signed) <= POSITION_EPS:

            return {

                "symbol": symbol,

                "status": "closed",

                "attempts": 0,

                "remaining_qty": 0.0,

                "order_side": None,

                "order_qty": 0.0,

                "reason": None,

            }



        order_side = "sell" if amt_signed > 0 else "buy"



        while attempts <= retries:

            order_qty = abs(amt_signed)

            attempts += 1



            try:

                self.cancel_all(symbol)

            except Exception as exc:

                logger.debug("Cancel all failed for %s: %s", symbol, exc)



            try:

                order_resp = self.create_market_order_safe(symbol, order_side, order_qty)

                executed_qty = float(order_resp.get("amount") or order_resp.get("filled") or order_qty)

            except Exception as exc:

                last_error = f"order:{exc}"

                logger.warning("Flatten attempt %s for %s failed to submit order: %s", attempts, symbol, exc)

                if attempts > retries:

                    break

                if backoff_sec > 0:

                    time.sleep(backoff_sec * attempts)

                try:

                    position = self.position_for(symbol) or {}

                    amt_signed = self._extract_position_amount(position)

                except Exception as fetch_exc:

                    last_error = f"position_fetch:{fetch_exc}"

                    logger.error("Unable to refresh position for %s after failure: %s", symbol, fetch_exc)

                    break

                order_side = "sell" if amt_signed > 0 else "buy"

                continue



            try:

                position = self.position_for(symbol) or {}

                amt_signed = self._extract_position_amount(position)

                remaining_qty = abs(amt_signed)

            except Exception as exc:

                last_error = f"verify:{exc}"

                logger.error("Post-flatten verification failed for %s: %s", symbol, exc)

                return {

                    "symbol": symbol,

                    "status": "error",

                    "attempts": attempts,

                    "remaining_qty": None,

                    "order_side": order_side,

                    "order_qty": executed_qty,

                    "reason": last_error,

                }



            if math.isfinite(remaining_qty) and remaining_qty <= POSITION_EPS:

                return {

                    "symbol": symbol,

                    "status": "closed",

                    "attempts": attempts,

                    "remaining_qty": 0.0,

                    "order_side": order_side,

                    "order_qty": executed_qty,

                    "reason": None,

                }



            last_error = f"remaining:{remaining_qty:.8f}"

            logger.warning(

                "Flatten attempt %s for %s left %.8f contracts; retrying.",

                attempts,

                symbol,

                remaining_qty,

            )



            if attempts > retries:

                break



            if backoff_sec > 0:

                time.sleep(backoff_sec * attempts)

            order_side = "sell" if amt_signed > 0 else "buy"



        try:

            position = self.position_for(symbol) or {}

            remaining_qty = abs(self._extract_position_amount(position))

        except Exception as exc:

            remaining_qty = float("nan")

            last_error = f"position_fetch_final:{exc}"

            logger.error("Failed to fetch final position for %s: %s", symbol, exc)



        return {

            "symbol": symbol,

            "status": "failed",

            "attempts": attempts,

            "remaining_qty": remaining_qty,

            "order_side": order_side,

            "order_qty": executed_qty,

            "reason": last_error or "unknown",

        }



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
