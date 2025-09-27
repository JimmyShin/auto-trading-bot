from __future__ import annotations

"""Exchange API helpers.

Developer tip:
    python -c "from auto_trading_bot.exchange_api import ExchangeAPI; print(ExchangeAPI().fetch_equity_snapshot())"
logs a BAL_RAW line for inspection.
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from broker_binance import BinanceUSDM, BalanceAuthError, BalanceSyncError

__all__ = [
    "ExchangeAPI",
    "BalanceAuthError",
    "BalanceSyncError",
    "EquitySnapshot",
    "EquitySnapshotError",
]

logger = logging.getLogger(__name__)


POSITION_EPS = 1e-8


class EquitySnapshotError(RuntimeError):
    """Raised when snapshot cannot be obtained after retries."""


@dataclass(frozen=True)
class EquitySnapshot:
    ts_utc: datetime
    wallet_balance: float
    margin_balance: float
    unrealized_pnl: float
    available_balance: float
    source: str
    account_mode: str


def _decimal(value: Any) -> Decimal:
    if value in (None, "", b"", False):
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise EquitySnapshotError(f"Invalid numeric value: {value!r}") from exc


class ExchangeAPI:
    """Adapter around Binance USDⓈ-M futures; fetch_equity_snapshot feeds reporter/metrics/alerts."""



    def __init__(
        self,
        client: Optional[BinanceUSDM] = None,
        auto_connect: bool = False,
        *,
        testnet: Optional[bool] = None,
        source: Optional[str] = None,
    ) -> None:
        self.client = client or BinanceUSDM()
        if testnet is not None:
            setattr(self.client, "testnet", bool(testnet))
        self._explicit_source = source
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


    def fetch_equity_snapshot(
        self,
        *,
        retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> EquitySnapshot:
        """Single source of truth for account equity on USDⓈ-M futures.

        Uses Binance endpoints:
          - /fapi/v2/balance  → wallet and available collateral
          - /fapi/v2/account  → unrealized PnL and margin balance
        """
        if retries < 1:
            raise ValueError("retries must be >= 1")

        account_mode = "testnet" if getattr(self.client, "testnet", False) else "live"
        source = (
            self._explicit_source
            if self._explicit_source is not None
            else ("binance-usdm-testnet" if account_mode == "testnet" else "binance-usdm")
        )

        attempt = 0
        while attempt < retries:
            attempt += 1
            try:
                balance_payload = self._http_get("/fapi/v2/balance")
                account_payload = self._http_get("/fapi/v2/account")
                break
            except Exception as exc:  # pragma: no cover - fallback message only
                if attempt >= retries:
                    message = "Unable to fetch USDⓈ-M equity snapshot"
                    raise EquitySnapshotError(message) from exc
                time.sleep(backoff_seconds * attempt)
        else:  # pragma: no cover - loop should break or raise
            raise EquitySnapshotError("Unable to fetch USDⓈ-M equity snapshot")

        balances: Sequence[Any]
        if isinstance(balance_payload, Sequence) and not isinstance(balance_payload, (str, bytes)):
            balances = balance_payload
        else:
            raise EquitySnapshotError("Balance payload malformed")

        if not isinstance(account_payload, dict):
            raise EquitySnapshotError("Account payload malformed")
        account = account_payload

        wallet_balance = Decimal("0")
        available_balance = Decimal("0")
        for entry_any in balances:
            entry = entry_any or {}
            if entry.get("asset") != "USDT":
                continue
            wallet_balance += _decimal(entry.get("balance"))
            available_balance += _decimal(entry.get("availableBalance"))

        positions = account.get("positions")
        positions_seq: Sequence[Any]
        if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
            positions_seq = positions
        else:
            positions_seq = []
        unrealized_pnl = Decimal("0")
        for pos_any in positions_seq:
            pos = pos_any or {}
            unrealized_pnl += _decimal(pos.get("unrealizedProfit"))

        margin_balance = _decimal(account.get("totalMarginBalance"))

        ts_utc = datetime.now(timezone.utc)
        snapshot = EquitySnapshot(
            ts_utc=ts_utc,
            wallet_balance=float(wallet_balance),
            margin_balance=float(margin_balance),
            unrealized_pnl=float(unrealized_pnl),
            available_balance=float(available_balance),
            source=source,
            account_mode=account_mode,
        )

        payload = {
            "type": "BAL_RAW",
            "ts_utc": ts_utc.isoformat().replace("+00:00", "Z"),
            "wallet_balance": snapshot.wallet_balance,
            "margin_balance": snapshot.margin_balance,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "available_balance": snapshot.available_balance,
            "source": snapshot.source,
            "account_mode": snapshot.account_mode,
        }
        logger.info(json.dumps(payload, sort_keys=True))

        return snapshot

    def _http_get(self, path: str) -> Any:
        if path == "/fapi/v2/balance":
            return self._fetch_balances()
        if path == "/fapi/v2/account":
            return self._fetch_account()
        raise EquitySnapshotError(f"Unsupported endpoint: {path}")

    def get_equity_usdt(self) -> float:
        """DEPRECATED: use fetch_equity_snapshot(). Kept for backward compatibility.

        Returns snapshot.margin_balance.
        """
        snapshot = self.fetch_equity_snapshot()
        return snapshot.margin_balance


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

    def cancel_reduce_only_stop_orders(self, symbol: str) -> None:
        cancel_fn = getattr(self.client, "cancel_reduce_only_stop_orders", None)
        if callable(cancel_fn):
            cancel_fn(symbol)

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



    def _resolve_callable(self, names: Iterable[str]) -> Optional[Callable[[], Any]]:
        for name in names:
            attr = getattr(self.client, name, None)
            if callable(attr):
                return attr
        exchange = getattr(self.client, "exchange", None)
        if exchange is not None:
            for name in names:
                attr = getattr(exchange, name, None)
                if callable(attr):
                    return attr
        return None

    def _fetch_balances(self) -> List[Dict[str, Any]]:
        func = self._resolve_callable(("fapiPrivateGetBalance", "fapi_private_get_balance"))
        if func is None:
            raise EquitySnapshotError("Client does not expose a USDⓈ-M balance endpoint")
        result = func()
        if isinstance(result, Sequence):
            return list(result)  # type: ignore[list-item]
        raise EquitySnapshotError("Balance endpoint returned unexpected payload")

    def _fetch_account(self) -> Dict[str, Any]:
        func = self._resolve_callable(("fapiPrivateGetAccount", "fapi_private_get_account"))
        if func is None:
            raise EquitySnapshotError("Client does not expose a USDⓈ-M account endpoint")
        result = func()
        if isinstance(result, dict):
            return result
        raise EquitySnapshotError("Account endpoint returned unexpected payload")

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
