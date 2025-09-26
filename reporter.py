from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Callable, Iterable


import math
import pandas as pd


__all__ = ["Reporter", "generate_report", "save_report"]


class Reporter:
    """Handles structured CSV logging for trading activity and diagnostics."""

    def __init__(self, base_dir: str, environment: str, clock: Optional[Callable[[], datetime]] = None) -> None:
        self.base_dir = base_dir
        self.environment = environment
        self._clock = clock or datetime.now

    @classmethod
    def from_config(cls) -> "Reporter":
        from config import DATA_BASE_DIR, TESTNET

        env = "testnet" if TESTNET else "live"
        return cls(DATA_BASE_DIR, env)

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
    def log_trade(self, symbol: str, side: str, entry_price: float, qty: float, reason: str = "signal", risk_usdt: float | None = None) -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"trades_{date}.csv")
        exists = os.path.exists(filename)
        fieldnames = [
            "timestamp",
            "symbol",
            "side",
            "entry_price",
            "qty",
            "reason",
            "status",
            "exit_timestamp",
            "exit_price",
            "pnl_pct",
            "exit_reason",
            "risk_usdt",
            "R_multiple",
        ]

        row = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "qty": qty,
            "reason": reason,
            "status": "OPEN",
            "exit_timestamp": "",
            "exit_price": "",
            "pnl_pct": "",
            "exit_reason": "",
            "risk_usdt": "",
            "R_multiple": "",
        }

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

    def log_exit(self, symbol: str, side: str, exit_price: float, pnl_pct: float, reason: str = "exit") -> None:
        ts = self._now_str()
        date = self._now_str("%Y-%m-%d")
        filename = os.path.join(self._root_dir(), f"trades_{date}.csv")
        fieldnames = [
            "timestamp",
            "symbol",
            "side",
            "entry_price",
            "qty",
            "reason",
            "status",
            "exit_timestamp",
            "exit_price",
            "pnl_pct",
            "exit_reason",
            "risk_usdt",
            "R_multiple",
        ]

        try:
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": ts,
                            "symbol": symbol,
                            "side": side,
                            "status": "CLOSED",
                            "exit_timestamp": ts,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": reason,
                            "entry_price": "",
                            "qty": "",
                            "reason": "",
                        }
                    )
                return

            rows = []
            with open(filename, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for field in fieldnames:
                        row.setdefault(field, "")
                    rows.append(row)

            found = False
            for row in reversed(rows):
                if row.get("symbol") == symbol and row.get("status", "").upper() == "OPEN":
                    # Compute R-multiple if risk_usdt and qty/entry present
                    try:
                        entry_price = float(row.get("entry_price") or 0.0)
                        qty = float(row.get("qty") or 0.0)
                        risk_usdt = float(row.get("risk_usdt") or 0.0)
                        if qty > 0 and entry_price > 0:
                            pnl_usdt = (float(exit_price) - entry_price) * qty if (str(side).lower() == "long") else (entry_price - float(exit_price)) * qty
                        else:
                            pnl_usdt = 0.0
                        r_mult = (pnl_usdt / risk_usdt) if risk_usdt and risk_usdt != 0 else ""
                    except Exception:
                        r_mult = ""
                    row.update(
                        {
                            "status": "CLOSED",
                            "exit_timestamp": ts,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "exit_reason": reason,
                            "R_multiple": r_mult,
                        }
                    )
                    found = True
                    break

            if not found:
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": "",
                        "qty": "",
                        "reason": "",
                        "status": "CLOSED",
                        "exit_timestamp": ts,
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
    """Generate a single-row DataFrame of performance metrics.

    Metrics:
    - total_trades: count
    - win_rate: wins / total
    - payoff_ratio: avg(win) / abs(avg(loss))
    - expectancy: (win_rate*avg(win)) - (loss_rate*avg(loss))
    - mdd: max drawdown ratio based on equity curve from returns
    - profit_factor: gross profit / gross loss
    - sharpe: mean(returns)/std(returns) (no annualization)
    - max_win_streak / max_loss_streak: consecutive positives/negatives
    - avg_hold_time_sec: mean(exit_ts - entry_ts) in seconds (ignored if timestamps missing)
    - long/short prefixed metrics via groupby(side)

    Notes:
    - Returns are computed from entry/exit if available, otherwise from pnl_pct (auto-scaled).
    - All rates are expressed as ratios (e.g., 0.52 for 52%).
    """
    df = _coerce_trades(trades)
    r = df["return"].astype(float).fillna(0.0)

    total = int(len(r))
    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())
    win_rate = (wins / total) if total else 0.0
    loss_rate = 1 - win_rate if total else 0.0
    avg_win = float(r[r > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(abs(r[r <= 0].mean())) if losses > 0 else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (math.inf if avg_win > 0 else 0.0)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

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

    # Average holding time in seconds (only when both timestamps are present)
    hold_secs = None
    if "entry_ts" in df.columns and "exit_ts" in df.columns:
        mask = df["entry_ts"].notna() & df["exit_ts"].notna()
        if mask.any():
            dt = (df.loc[mask, "exit_ts"] - df.loc[mask, "entry_ts"]).dropna()
            if not dt.empty:
                hold_secs = float(dt.dt.total_seconds().mean())
    avg_hold_time_sec = hold_secs if hold_secs is not None else 0.0

    # Group-by side metrics
    by_side = _group_metrics_by_side(df)

    # Compose final single-row DataFrame
    # Avg R-multiple if provided on input; else fallback to expectancy
    avg_r_multiple = 0.0
    if "R_multiple" in df.columns:
        try:
            avg_r_multiple = float(pd.to_numeric(df["R_multiple"], errors="coerce").dropna().mean())
        except Exception:
            avg_r_multiple = 0.0

    data: Dict[str, Any] = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "avg_r": (avg_r_multiple if avg_r_multiple != 0.0 else expectancy),
        "mdd": mdd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_hold_time_sec": avg_hold_time_sec,
    }
    data.update(by_side)

    return pd.DataFrame([data])


def save_report(df: pd.DataFrame, filepath: str) -> None:
    """Save report DataFrame to Excel (index=False)."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_excel(filepath, index=False)
