from __future__ import annotations

import csv
import json
import time
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable, Iterable
from decimal import Decimal


import logging
import math
import pandas as pd


from auto_trading_bot.exchange_api import EquitySnapshot


logger = logging.getLogger(__name__)

_DEBUG_VALUES = {"1", "true", "yes", "on"}


def _debug_enabled() -> bool:
    return os.getenv("OBS_DEBUG_ALERTS", "").strip().lower() in _DEBUG_VALUES


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


LAST_DD_LOG: Dict[str, float] = {}


def _log_report_metrics(payload: Dict[str, Any]) -> None:
    if not _debug_enabled():
        return
    try:
        logger.info("REPORTER_METRICS %s", json.dumps(payload, sort_keys=True))
    except Exception:
        logger.info("REPORTER_METRICS %s", payload)


__all__ = ["Reporter", "generate_report", "save_report"]

# RAW trades CSV schema version (append-only schema). Bump on column changes.
SCHEMA_VERSION = 5


class Reporter:
    """Consumes EquitySnapshot and maintains daily peak equity for drawdown metrics."""

    def __init__(
        self,
        base_dir: str,
        environment: str,
        clock: Optional[Callable[[], datetime]] = None,
        *,
        metrics: Optional[Any] = None,
        logger_name: str = __name__,
    ) -> None:
        self.base_dir = base_dir
        self.environment = environment
        self._clock = clock or datetime.now
        self._metrics = metrics
        self._logger = logging.getLogger(logger_name)
        self._daily_peak_equity: Dict[str, float] = {}
        self._last_dd_ratio: float = 0.0

    @classmethod
    def from_config(cls, *, metrics: Optional[Any] = None) -> Reporter:
        from config import DATA_BASE_DIR, TESTNET

        env = "testnet" if TESTNET else "live"
        return cls(DATA_BASE_DIR, env, metrics=metrics)

    def apply_equity_snapshot(
        self,
        snap: EquitySnapshot,
        *,
        now_utc: Optional[datetime] = None,
    ) -> float:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        else:
            now_utc = now_utc.astimezone(timezone.utc)

        day_key = now_utc.strftime("%Y-%m-%d")
        equity_dec = Decimal(str(snap.margin_balance))
        peak_existing = self._daily_peak_equity.get(day_key)
        peak_dec = equity_dec if peak_existing is None else max(Decimal(str(peak_existing)), equity_dec)
        self._daily_peak_equity[day_key] = float(peak_dec)

        eps = Decimal("1e-12")
        denom = peak_dec if peak_dec > Decimal("0") else eps
        dd_dec = (peak_dec - equity_dec) / denom
        if dd_dec < Decimal("0"):
            dd_dec = Decimal("0")
        if dd_dec > Decimal("1"):
            dd_dec = Decimal("1")

        dd_ratio = float(dd_dec)
        self._last_dd_ratio = dd_ratio

        payload = {
            "type": "DD_CALC",
            "ts_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "equity": float(equity_dec),
            "peak_equity": float(peak_dec),
            "dd_ratio": dd_ratio,
            "source": snap.source,
            "account_mode": snap.account_mode,
        }
        self._logger.info(json.dumps(payload, sort_keys=True))

        if self._metrics is not None:
            self._metrics.update_daily_drawdown(dd_ratio, equity=float(equity_dec), ts=now_utc)

        return dd_ratio

    @property
    def daily_peak_equity(self) -> Dict[str, float]:
        return dict(self._daily_peak_equity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _root_dir(self) -> str:
        path = os.path.join(self.base_dir, self.environment)
        os.makedirs(path, exist_ok=True)
        return path

    def _now_str(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        return self._clock().strftime(fmt)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------
    def _trade_fieldnames(self):
        return [
            # Schema / identity
            "schema_version",
            # Entry snapshot
            "timestamp",      # legacy alias
            "entry_ts_utc",
            "symbol",
            "side",
            "timeframe",
            "leverage",
            "entry_price",
            "qty",
            "reason",
            # Risk snapshot
            "entry_atr_abs",
            "atr_period",
            "atr_source",
            "stop_basis",
            "stop_k",
            "fallback_pct",
            "stop_distance",
            "risk_usdt_planned",
            # Lifecycle
            "status",
            "exit_timestamp",   # legacy alias
            "exit_ts_utc",
            "exit_price",
            "pnl_pct",
            "exit_reason",
            "exit_reason_code",
            # Ex-post
            "fees_quote_actual",
            "pnl_quote_expost",
            "R_atr_expost",
            "R_usd_expost",
            # Legacy compat
            "risk_usdt",
            "R_multiple",
        ]

    def log_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        reason: str = "signal",
        risk_usdt: float | None = None,
        *,
        timeframe: Optional[str] = None,
        leverage: Optional[float] = None,
        entry_atr_abs: Optional[float] = None,
        atr_period: Optional[int] = None,
        atr_source: Optional[str] = None,
        stop_basis: Optional[str] = None,
        stop_k: Optional[float] = None,
        fallback_pct: Optional[float] = None,
        stop_distance: Optional[float] = None,
        risk_usdt_planned: Optional[float] = None,
    ) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"trades_{date}.csv")
        exists = os.path.exists(filename)
        fieldnames = self._trade_fieldnames()

        row: Dict[str, Any] = {k: "" for k in fieldnames}
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": ts,
                "entry_ts_utc": ts,
                "symbol": symbol,
                "side": side,
                "timeframe": timeframe or "",
                "leverage": leverage if leverage is not None else "",
                "entry_price": entry_price,
                "qty": qty,
                "reason": reason,
                # snapshot
                "entry_atr_abs": entry_atr_abs if entry_atr_abs is not None else "",
                "atr_period": atr_period if atr_period is not None else "",
                "atr_source": atr_source or "",
                "stop_basis": (stop_basis or "").lower() if stop_basis else "",
                "stop_k": stop_k if stop_k is not None else "",
                "fallback_pct": fallback_pct if fallback_pct is not None else "",
                "stop_distance": stop_distance if stop_distance is not None else "",
                "risk_usdt_planned": risk_usdt_planned if risk_usdt_planned is not None else "",
                # lifecycle
                "status": "OPEN",
                "exit_timestamp": "",
                "exit_ts_utc": "",
                "exit_price": "",
                "pnl_pct": "",
                "exit_reason": "",
                "exit_reason_code": "",
                # ex-post (empty at entry)
                "fees_quote_actual": "",
                "pnl_quote_expost": "",
                "R_atr_expost": "",
                "R_usd_expost": "",
                # legacy
                "risk_usdt": "",
                "R_multiple": "",
            }
        )

        try:
            with open(filename, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not exists or os.path.getsize(filename) == 0:
                    writer.writeheader()
                if risk_usdt is not None:
                    try:
                        row["risk_usdt"] = float(risk_usdt)
                    except Exception:
                        row["risk_usdt"] = risk_usdt
                writer.writerow(row)
        except Exception as exc:
            print(f"[WARN] trade log failed: {exc}")

    def log_exit(
        self,
        symbol: str,
        side: str,
        exit_price: float,
        pnl_pct: float,
        reason: str = "exit",
        *,
        reason_code: str = "auto_exit",
        fees_quote_actual: Optional[float] = None,
    ) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"trades_{date}.csv")
        fieldnames = self._trade_fieldnames()

        try:
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "timestamp": ts,
                            "entry_ts_utc": "",
                            "symbol": symbol,
                            "side": side,
                            "status": "CLOSED",
                            "exit_timestamp": ts,
                            "exit_ts_utc": ts,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": reason,
                            "exit_reason_code": reason_code,
                        }
                    )
                return

            rows = []
            with open(filename, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    # ensure new schema keys exist
                    for field in fieldnames:
                        row.setdefault(field, "")
                    rows.append(row)

            found = False
            for row in reversed(rows):
                if row.get("symbol") == symbol and row.get("status", "").upper() == "OPEN":
                    # compute ex-post metrics
                    try:
                        entry_price = float(row.get("entry_price") or 0.0)
                        qty = float(row.get("qty") or 0.0)
                        pnl_dist = (float(exit_price) - entry_price) if (str(side).lower() == "long") else (entry_price - float(exit_price))
                        gross_pnl_quote = pnl_dist * qty if (qty > 0) else 0.0
                    except Exception:
                        pnl_dist, gross_pnl_quote = 0.0, 0.0

                    try:
                        stop_k = float(row.get("stop_k") or 0.0)
                    except Exception:
                        stop_k = 0.0
                    try:
                        entry_atr_abs = float(row.get("entry_atr_abs") or 0.0)
                    except Exception:
                        entry_atr_abs = 0.0
                    try:
                        risk_usdt_planned = float(row.get("risk_usdt_planned") or (row.get("risk_usdt") or 0.0) or 0.0)
                    except Exception:
                        risk_usdt_planned = 0.0
                    try:
                        fees = float(fees_quote_actual) if fees_quote_actual is not None else float(row.get("fees_quote_actual") or 0.0)
                    except Exception:
                        fees = 0.0

                    pnl_quote_expost = gross_pnl_quote - fees
                    denom_atr = stop_k * entry_atr_abs
                    r_atr = (pnl_dist / denom_atr) if denom_atr not in (0, 0.0) else ""
                    r_usd = (pnl_quote_expost / risk_usdt_planned) if risk_usdt_planned not in (0, 0.0) else ""

                    row.update(
                        {
                            "status": "CLOSED",
                            "exit_timestamp": ts,
                            "exit_ts_utc": ts,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": reason,
                            "exit_reason_code": reason_code,
                            "fees_quote_actual": fees if fees != 0 else (row.get("fees_quote_actual") or ""),
                            "pnl_quote_expost": pnl_quote_expost if pnl_quote_expost != 0 else (row.get("pnl_quote_expost") or ""),
                            "R_atr_expost": r_atr,
                            "R_usd_expost": r_usd,
                            # legacy convenience
                            "R_multiple": r_atr,
                        }
                    )
                    found = True
                    break

            if not found:
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": ts,
                        "entry_ts_utc": "",
                        "symbol": symbol,
                        "side": side,
                        "entry_price": "",
                        "qty": "",
                        "reason": "",
                        "status": "CLOSED",
                        "exit_timestamp": ts,
                        "exit_ts_utc": ts,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "exit_reason": reason,
                    }
                )

            with open(filename, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
        except Exception as exc:
            print(f"[WARN] exit log failed: {exc}")

    def log_signal_analysis(
        self,
        symbol: str,
        current_price: float,
        signal: Dict[str, Any],
        is_new_bar: Optional[bool] = None,
        funding_avoid: Optional[bool] = None,
        daily_loss_hit: Optional[bool] = None,
        equity_usdt: Optional[float] = None,
        daily_dd_pct: Optional[float] = None,
        decision: Optional[str] = None,
        skip_reason: str = "",
    ) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"signal_analysis_{date}.csv")
        exists = os.path.exists(filename)

        # Backward compatibility: legacy call packed values in dict
        if isinstance(is_new_bar, dict) and not isinstance(funding_avoid, (bool, type(None))):
            legacy = is_new_bar
            legacy_decision = funding_avoid
            legacy_skip = daily_loss_hit
            is_new_bar = legacy.get("is_new_bar")
            funding_avoid = legacy.get("funding_avoid")
            daily_loss_hit = legacy.get("daily_loss_hit")
            equity_usdt = legacy.get("equity_usdt")
            daily_dd_pct = legacy.get("daily_dd_pct")
            if decision is None:
                decision = legacy_decision
            if not skip_reason and legacy_skip is not None:
                skip_reason = legacy_skip

        try:
            equity_val = float(equity_usdt or 0.0)
        except Exception:
            equity_val = 0.0
        try:
            dd_val = float(daily_dd_pct or 0.0)
        except Exception:
            dd_val = 0.0

        decision = decision or ""
        signal = signal or {}
        fast_ma = signal.get("fast_ma", 0.0)
        slow_ma = signal.get("slow_ma", 0.0)
        ma_diff_pct = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        alignment = (signal.get("alignment") or {}) if isinstance(signal, dict) else {}
        candle_filter = (signal.get("candle_filter") or {}) if isinstance(signal, dict) else {}
        risk_multiplier = (signal.get("risk_multiplier") or {}) if isinstance(signal, dict) else {}

        try:
            with open(filename, "a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if not exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "symbol",
                            "price",
                            "fast_ma",
                            "slow_ma",
                            "ma_diff_pct",
                            "long_signal",
                            "short_signal",
                            "regime",
                            "long_aligned",
                            "short_aligned",
                            "long_cross",
                            "short_cross",
                            "candle_position_ratio",
                            "candle_safe_long",
                            "candle_safe_short",
                            "risk_multiplier_long",
                            "risk_multiplier_short",
                            "is_new_bar",
                            "funding_avoid",
                            "daily_loss_hit",
                            "equity_usdt",
                            "daily_dd_pct",
                            "decision",
                            "skip_reason",
                        ]
                    )
                writer.writerow(
                    [
                        ts,
                        symbol,
                        current_price,
                        fast_ma,
                        slow_ma,
                        round(ma_diff_pct, 3),
                        bool(signal.get("long")),
                        bool(signal.get("short")),
                        (signal.get("regime") if isinstance(signal, dict) else "UNKNOWN"),
                        bool(alignment.get("long_aligned")),
                        bool(alignment.get("short_aligned")),
                        bool(alignment.get("long_cross")),
                        bool(alignment.get("short_cross")),
                        candle_filter.get("candle_position_ratio", 0.0),
                        bool(candle_filter.get("candle_safe_long")),
                        bool(candle_filter.get("candle_safe_short")),
                        risk_multiplier.get("long", 1.0),
                        risk_multiplier.get("short", 1.0),
                        bool(is_new_bar),
                        bool(funding_avoid),
                        bool(daily_loss_hit),
                        equity_val,
                        round(dd_val, 3),
                        decision,
                        skip_reason or "",
                    ]
                )
        except Exception as exc:
            print(f"[WARN] signal log failed: {exc}")

    def log_filtered_signal(
        self,
        symbol: str,
        current_price: float,
        signal: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
        decision: str = "SKIP",
        skip_reason: str = "",
        **extra: Any,
    ) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"filtered_signals_{date}.csv")
        exists = os.path.exists(filename)

        signal = signal or {}
        analysis_payload: Dict[str, Any] = {}
        if isinstance(analysis, dict):
            analysis_payload.update(analysis)
        if extra:
            analysis_payload.update({k: v for k, v in extra.items() if k not in analysis_payload})

        fast_ma = signal.get("fast_ma", 0.0)
        slow_ma = signal.get("slow_ma", 0.0)
        ma_diff = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        candle_filter = signal.get("candle_filter", {}) or {}

        fieldnames = [
            "timestamp",
            "symbol",
            "price",
            "decision",
            "skip_reason",
            "fast_ma",
            "slow_ma",
            "ma_diff_pct",
            "long_signal",
            "short_signal",
            "candle_position_ratio",
            "candle_safe_long",
            "candle_safe_short",
            "context",
        ]

        try:
            with open(filename, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "price": float(current_price),
                        "decision": decision or "SKIP",
                        "skip_reason": skip_reason or analysis_payload.get("skip_reason", ""),
                        "fast_ma": fast_ma,
                        "slow_ma": slow_ma,
                        "ma_diff_pct": round(ma_diff, 3),
                        "long_signal": bool(signal.get("long", False)),
                        "short_signal": bool(signal.get("short", False)),
                        "candle_position_ratio": candle_filter.get("candle_position_ratio", 0.0),
                        "candle_safe_long": bool(candle_filter.get("candle_safe_long")),
                        "candle_safe_short": bool(candle_filter.get("candle_safe_short")),
                        "context": json.dumps(analysis_payload, default=str) if analysis_payload else "",
                    }
                )
        except Exception as exc:
            print(f"[WARN] filtered signal log failed: {exc}")

    def log_detailed_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        stop_price: float,
        risk_multiplier: float,
        atr_abs: float,
        sig: Optional[Dict[str, Any]] = None,
        equity_usdt: Optional[float] = None,
        reason: str = "signal",
        reasons_list: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"detailed_entries_{date}.csv")

        equity_alias = kwargs.pop("equity", None)
        signal_payload = sig if sig is not None else kwargs.pop("signal", None)
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if equity_usdt is not None and equity_alias is not None:
            try:
                if abs(float(equity_usdt) - float(equity_alias)) > 1e-6:
                    print(f"[WARN] log_detailed_entry equity mismatch ({equity_usdt} vs {equity_alias})")
            except Exception:
                pass
        equity_value = equity_usdt if equity_usdt is not None else equity_alias
        if equity_value is None:
            raise ValueError("equity_usdt or equity must be provided")
        try:
            equity_value = float(equity_value)
        except Exception:
            raise ValueError("equity must be numeric") from None

        exists = os.path.exists(filename)
        position_value = float(qty) * float(entry_price)
        risk_amount = float(qty) * abs(float(entry_price) - float(stop_price))
        risk_pct = (risk_amount / max(float(equity_value), 1e-9)) * 100.0
        stop_distance_pct = abs(float(entry_price) - float(stop_price)) / max(float(entry_price), 1e-9) * 100.0

        signal = signal_payload or {}
        fast_ma = signal.get("fast_ma", 0.0)
        slow_ma = signal.get("slow_ma", 0.0)
        ma_diff_pct = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        regime = signal.get("regime", "UNKNOWN")
        reasons_str = "|".join(reasons_list) if reasons_list else ""
        be_promotion_price = float(entry_price) + (1.5 * float(atr_abs)) if str(side).upper().startswith("LONG") else float(entry_price) - (1.5 * float(atr_abs))
        expected_trail_range = (
            f"${entry_price:.0f} ~ ${be_promotion_price + (2.5 * atr_abs):.0f}"
            if str(side).upper().startswith("LONG")
            else f"${be_promotion_price - (2.5 * atr_abs):.0f} ~ ${entry_price:.0f}"
        )
        entry_logic = (
            "Trend alignment with reduced risk"
            if float(risk_multiplier or 1.0) < 1.0
            else "Trend alignment with full risk"
        )

        try:
            with open(filename, "a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if not exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "symbol",
                            "side",
                            "entry_price",
                            "qty",
                            "position_value_usd",
                            "stop_price",
                            "stop_distance_pct",
                            "risk_amount_usd",
                            "risk_pct",
                            "risk_multiplier",
                            "be_promotion_price",
                            "expected_trail_range",
                            "fast_ma",
                            "slow_ma",
                            "ma_diff_pct",
                            "regime",
                            "atr_abs",
                            "equity_usdt",
                            "reason",
                            "entry_logic",
                            "reasons",
                        ]
                    )
                writer.writerow(
                    [
                        ts,
                        symbol,
                        side,
                        entry_price,
                        qty,
                        round(position_value, 2),
                        stop_price,
                        round(stop_distance_pct, 3),
                        round(risk_amount, 2),
                        round(risk_pct, 2),
                        float(risk_multiplier or 1.0),
                        be_promotion_price,
                        expected_trail_range,
                        fast_ma,
                        slow_ma,
                        round(ma_diff_pct, 3),
                        regime,
                        atr_abs,
                        equity_value,
                        reason,
                        entry_logic,
                        reasons_str,
                    ]
                )
        except Exception as exc:
            print(f"[WARN] detailed entry log failed: {exc}")


# ------------------------------------------------------------------
# Analytics/report generation (no plotting)
# ------------------------------------------------------------------
def _coerce_trades(trades: Any) -> pd.DataFrame:
    """Coerce input into a DataFrame with expected columns.

    Supported inputs:
    - pd.DataFrame
    - Iterable[Mapping]

    Tries to compute per-trade returns as decimal (e.g., 0.012 = 1.2%).
    Priority:
      1) If columns 'entry' and 'exit' exist, compute from prices and 'side'.
      2) Else if 'pnl_pct' exists, infer scale (percent vs decimal).

    Optionally uses 'entry_ts'/'exit_ts' or 'ts' for timestamps.
    """
    if isinstance(trades, pd.DataFrame):
        df = trades.copy()
    else:
        try:
            df = pd.DataFrame(list(trades))  # type: ignore[arg-type]
        except Exception:
            raise TypeError("trades must be a DataFrame or an iterable of mappings") from None

    # Normalize side to lowercase for grouping later (if present)
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()

    returns: Optional[pd.Series]
    returns = None

    # Case 1: compute from entry/exit and side
    if {"entry", "exit"}.issubset(df.columns):
        entry = pd.to_numeric(df["entry"], errors="coerce")
        exitp = pd.to_numeric(df["exit"], errors="coerce")
        side = df.get("side").astype(str).str.lower() if "side" in df.columns else "long"
        long_mask = side.eq("long")
        short_mask = side.eq("short")
        ret = pd.Series(index=df.index, dtype="float64")
        ret.loc[long_mask] = (exitp[long_mask] - entry[long_mask]) / entry[long_mask]
        ret.loc[short_mask] = (entry[short_mask] - exitp[short_mask]) / entry[short_mask]
        # Any remaining rows treated as long by default
        other_mask = ~(long_mask | short_mask)
        ret.loc[other_mask] = (exitp[other_mask] - entry[other_mask]) / entry[other_mask]
        returns = ret.fillna(0.0)

    # Case 2: fallback to pnl_pct (auto-scale)
    if returns is None and "pnl_pct" in df.columns:
        pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
        # Heuristic: values > 1 likely represent percent units; scale to decimal
        scale_div = 100.0 if pnl.abs().median() > 1.0 else 1.0
        returns = (pnl / scale_div).astype(float)

    if returns is None:
        # As a last resort create zeros; metrics will mostly be zero
        returns = pd.Series(0.0, index=df.index, dtype="float64")

    df["return"] = returns

    # Try to compose entry/exit timestamps for holding time
    # Prefer explicit entry_ts/exit_ts columns; fall back to single 'ts'.
    # If only one timestamp is present, hold time will be NaT and ignored in mean.
    if "entry_ts" in df.columns:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce", utc=True)
    elif "ts" in df.columns:
        df["entry_ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        df["entry_ts"] = pd.NaT
    if "exit_ts" in df.columns:
        df["exit_ts"] = pd.to_datetime(df["exit_ts"], errors="coerce", utc=True)
    else:
        df["exit_ts"] = pd.NaT

    return df


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Compute MDD as max peak-to-trough drawdown / peak (ratio, not percent).

    Builds an equity curve via cumulative product of (1 + r).
    Returns 0.0 for empty series.
    """
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    roll_max = equity.cummax()
    dd = (roll_max - equity) / roll_max
    return float(dd.max()) if not dd.empty else 0.0


def _streaks(returns: pd.Series) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak) based on sign of returns."""
    max_win = max_loss = cur_win = cur_loss = 0
    for r in returns.fillna(0.0):
        if r > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _group_metrics_by_side(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {
        "long_trades": 0.0,
        "long_win_rate": 0.0,
        "long_expectancy": 0.0,
        "long_profit_factor": 0.0,
        "short_trades": 0.0,
        "short_win_rate": 0.0,
        "short_expectancy": 0.0,
        "short_profit_factor": 0.0,
    }
    if "side" not in df.columns:
        return out
    for name, g in df.groupby(df["side"].astype(str).str.lower()):
        r = g["return"].astype(float)
        total = len(r)
        wins = (r > 0).sum()
        losses = (r <= 0).sum()
        win_rate = (wins / total) if total else 0.0
        avg_win = r[r > 0].mean() if wins > 0 else 0.0
        avg_loss = abs(r[r <= 0].mean()) if losses > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        gross_win = r[r > 0].sum()
        gross_loss = abs(r[r <= 0].sum())
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
        prefix = f"{name}_"  # 'long_' or 'short_'
        out[f"{prefix}trades"] = float(total)
        out[f"{prefix}win_rate"] = float(win_rate)
        out[f"{prefix}expectancy"] = float(expectancy)
        out[f"{prefix}profit_factor"] = float(profit_factor)
    return out


def generate_report(trades: Any, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Generate metrics without substituting Avg R.

    - avg_r_atr: mean of R_atr_expost (if present)
    - avg_r_usd: mean of R_usd_expost (if present)
    - win_rate: fraction of trades with positive pnl_quote_expost
    - expectancy_usd: win_rate*avg_win_usd - (1-win_rate)*avg_loss_usd
    """
    df = _coerce_trades(trades)

    r_atr_series = pd.to_numeric(df.get("R_atr_expost", pd.Series(dtype=float)), errors="coerce")
    r_usd_series = pd.to_numeric(df.get("R_usd_expost", pd.Series(dtype=float)), errors="coerce")
    pnl_usd = pd.to_numeric(df.get("pnl_quote_expost", pd.Series(dtype=float)), errors="coerce").fillna(pd.NA)

    # Return-based metrics (primary for win rate / expectancy)
    r = df["return"].astype(float).fillna(0.0)
    total = int(len(r))
    wins = int((r > 0).sum())
    losses = total - wins
    win_rate = (wins / total) if total else 0.0
    avg_win = float(r[r > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(abs(r[r <= 0].mean())) if losses > 0 else 0.0
    expectancy_ratio = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # USD-based metrics (optional; fallback to 0 if not provided)
    pnl_usd_valid = pnl_usd.dropna()
    usd_wins = pnl_usd_valid[pnl_usd_valid > 0]
    usd_losses = pnl_usd_valid[pnl_usd_valid <= 0]
    avg_win_usd = float(usd_wins.mean()) if not usd_wins.empty else 0.0
    avg_loss_usd = float(abs(usd_losses.mean())) if not usd_losses.empty else 0.0
    win_rate_usd = 0.0
    if not pnl_usd_valid.empty:
        win_rate_usd = float((pnl_usd_valid > 0).sum() / len(pnl_usd_valid))
    expectancy_usd = (win_rate_usd * avg_win_usd) - ((1 - win_rate_usd) * avg_loss_usd)

    avg_r_atr = float(r_atr_series.dropna().mean()) if not r_atr_series.dropna().empty else float("nan")
    avg_r_usd = float(r_usd_series.dropna().mean()) if not r_usd_series.dropna().empty else float("nan")

    # Legacy payoff/profit-factor using returns
    payoff_ratio = 0.0
    try:
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (math.inf if avg_win > 0 else 0.0)
    except Exception:
        payoff_ratio = 0.0

    gross_win = float(r[r > 0].sum())
    gross_loss = float(abs(r[r <= 0].sum()))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)

    sharpe = 0.0
    std = float(r.std(ddof=1)) if len(r) >= 2 else 0.0
    mean = float(r.mean()) if len(r) else 0.0
    if std > 0 and not math.isnan(std):
        sharpe = mean / std

    mdd = _max_drawdown_from_returns(r)
    max_win_streak, max_loss_streak = _streaks(r)

    avg_hold_time_sec = 0.0
    if "entry_ts" in df.columns and "exit_ts" in df.columns:
        mask = df["entry_ts"].notna() & df["exit_ts"].notna()
        if mask.any():
            dt = (df.loc[mask, "exit_ts"] - df.loc[mask, "entry_ts"]).dropna()
            if not dt.empty:
                avg_hold_time_sec = float(dt.dt.total_seconds().mean())

    stop_basis = (df.get("stop_basis") or pd.Series(dtype=str)).astype(str).str.lower()
    percent_count = int((stop_basis == "percent").sum()) if len(stop_basis) else 0

    data: Dict[str, Any] = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy_ratio,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "expectancy_usd": expectancy_usd,
        "avg_r_atr": avg_r_atr,
        "avg_r_usd": avg_r_usd,
        "payoff_ratio": payoff_ratio,
        "mdd": mdd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_hold_time_sec": avg_hold_time_sec,
        "fallback_percent_count": percent_count,
    }
    data.update(_group_metrics_by_side(df))

    _log_report_metrics({
        "avg_r_atr_30": _safe_number(data.get("avg_r_atr")),
        "window_trades": len(df),
        "profit_factor_30": _safe_number(data.get("profit_factor")),
        "expectancy_usd_30": _safe_number(data.get("expectancy_usd")),
    })

    return pd.DataFrame([data])


def save_report(df: pd.DataFrame, filepath: str) -> None:
    """Save report DataFrame to Excel (index=False)."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_excel(filepath, index=False)

# Binance USDⓈ-M daily drawdown:
#   wallet_balance := totalWalletBalance (cash)
#   unrealized_pnl := totalUnrealizedProfit (signed)
#   equity := margin_balance = wallet_balance + unrealized_pnl
# Daily DD ratio = max(0, (peak - equity) / peak), peak tracked intraday.

PEAK_STATE: Dict[str, Dict[str, float]] = {}


def _reset_peak(account: str) -> None:
    PEAK_STATE.pop(account, None)


def _current_utc_day() -> str:
    return datetime.utcnow().strftime('%Y-%m-%d')


def apply_equity_snapshot(account: str, equity: Optional[float], *, now: Optional[float] = None) -> Optional[float]:
    if equity is None or not math.isfinite(float(equity)) or equity <= 0:
        return None
    equity = float(equity)
    ts = now if now is not None else time.time()
    day = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
    state = PEAK_STATE.setdefault(account, {"day": day, "peak": equity})
    if state.get("day") != day or state.get("peak", 0.0) <= 0:
        state["day"] = day
        state["peak"] = equity
    if equity > state["peak"]:
        state["peak"] = equity
    peak = state["peak"]
    if peak <= 0:
        return None
    dd = max(0.0, (peak - equity) / peak)
    dd = min(dd, 1.0)
    return dd
