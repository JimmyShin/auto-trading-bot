from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any

import glob
import csv
import requests


class SlackNotifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = (webhook_url or "").strip()

    @classmethod
    def from_env(cls) -> "SlackNotifier":
        return cls(os.getenv("SLACK_WEBHOOK_URL"))

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, text: str) -> bool:
        if not self.enabled():
            return False
        try:
            payload = {"text": str(text)}
            resp = requests.post(self.webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception as e:
            try:
                print(f"[WARN] Slack notify failed: {e}")
            except Exception:
                pass
            return False


_SLACK = None  # lazy singleton


def _get_notifier() -> SlackNotifier:
    global _SLACK
    if _SLACK is None:
        _SLACK = SlackNotifier.from_env()
    return _SLACK


def slack_notify_safely(message: str) -> bool:
    n = _get_notifier()
    return n.send(message)


def _collect_closed_trades(base_dir: str, env: str) -> List[Dict[str, Any]]:
    """Collect closed trades across data/<env>/trades_*.csv.

    Returns rows with side, entry_price, exit_price, qty, R_atr_expost, R_usd_expost,
    pnl_quote_expost, stop_basis, exit_ts_utc. Ignores incomplete rows.
    """
    root = os.path.join(base_dir, env)
    patterns = [os.path.join(root, "trades_*.csv")]
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    out: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8", newline="") as fh:
                r = csv.DictReader(fh)
                for row in r:
                    if (row.get("status") or "").upper() != "CLOSED":
                        continue
                    try:
                        entry = float(row.get("entry_price") or 0.0)
                        exitp = float(row.get("exit_price") or 0.0)
                        qty = float(row.get("qty") or 0.0)
                    except Exception:
                        continue
                    if entry <= 0 or exitp <= 0:
                        continue
                    side = (row.get("side") or "").strip().lower()
                    def _fnum(x):
                        try:
                            return float(x)
                        except Exception:
                            return None
                    out.append({
                        "symbol": row.get("symbol") or "",
                        "side": side,
                        "entry_price": entry,
                        "exit_price": exitp,
                        "qty": qty,
                        "entry_atr_abs": _fnum(row.get("entry_atr_abs")),
                        "stop_k": _fnum(row.get("stop_k")),
                        "stop_basis": (row.get("stop_basis") or "").strip().lower(),
                        "R_atr_expost": _fnum(row.get("R_atr_expost")),
                        "R_usd_expost": _fnum(row.get("R_usd_expost")),
                        "pnl_quote_expost": _fnum(row.get("pnl_quote_expost")),
                        "exit_ts_utc": row.get("exit_ts_utc") or row.get("exit_timestamp") or "",
                    })
        except Exception:
            continue
    return out


def _rolling_metrics_from_logs(base_dir: str, env: str, window: int = 30, rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    try:
        if rows is None:
            rows = _collect_closed_trades(base_dir, env)
        if not rows:
            return {
                "win_rate_pct": 0.0,
                "avg_r_atr": None,
                "avg_r_usd": None,
                "expectancy_usd": 0.0,
                "fallback_percent_count": 0,
                "N": 0,
            }
        import pandas as pd
        from reporter import generate_report
        df = pd.DataFrame(rows)
        try:
            if "exit_ts_utc" in df.columns:
                df = df.sort_values(by="exit_ts_utc")
        except Exception:
            pass
        dfw = df.tail(window)
        rep = generate_report(dfw).iloc[0]
        win_rate_pct = float(rep.get("win_rate", 0.0)) * 100.0
        avg_r_atr = rep.get("avg_r_atr")
        avg_r_usd = rep.get("avg_r_usd")
        expectancy_usd = rep.get("expectancy_usd", 0.0)
        fallback_count = int(rep.get("fallback_percent_count", 0))
        def _nan_to_none(v):
            try:
                return None if (isinstance(v, float) and v != v) else v
            except Exception:
                return v
        return {
            "win_rate_pct": float(win_rate_pct),
            "avg_r_atr": _nan_to_none(avg_r_atr),
            "avg_r_usd": _nan_to_none(avg_r_usd),
            "expectancy_usd": float(expectancy_usd) if expectancy_usd is not None else 0.0,
            "fallback_percent_count": fallback_count,
            "N": int(len(dfw)),
        }
    except Exception:
        return {
            "win_rate_pct": 0.0,
            "avg_r_atr": None,
            "avg_r_usd": None,
            "expectancy_usd": 0.0,
            "fallback_percent_count": 0,
            "N": 0,
        }


def _compute_latest_r_atr(symbol: str, side: str, entry_price: float, exit_price: float, rows: List[Dict[str, Any]]) -> Optional[float]:
    """Compute immediate R_atr using latest trade snapshot for the symbol."""
    if not rows:
        return None
    symbol = (symbol or "").strip()
    side_norm = (side or "").strip().lower()
    symbol_norm = symbol.upper()
    for row in reversed(rows):
        if (row.get("symbol") or "").upper() != symbol_norm:
            continue
        if (row.get("side") or "").lower() != side_norm:
            continue
        entry_atr = row.get("entry_atr_abs")
        stop_k = row.get("stop_k")
        if entry_atr in (None, 0, 0.0) or stop_k in (None, 0, 0.0):
            continue
        denom = float(stop_k) * float(entry_atr)
        if denom == 0:
            continue
        if side_norm == "long":
            pnl_distance = float(exit_price) - float(entry_price)
        else:
            pnl_distance = float(entry_price) - float(exit_price)
        return pnl_distance / denom
    return None


def slack_notify_exit(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl_usdt: float,
    pnl_pct: float,
    equity_usdt: float,
) -> bool:
    """Send a Slack message about a position exit.

    Uses rolling (N=30) metrics. Avg R shows ATR-based only; no fallbacks.
    """
    try:
        import config as cfg
        env = "testnet" if getattr(cfg, "TESTNET", True) else "live"
        base_dir = getattr(cfg, "DATA_BASE_DIR", "data")
    except Exception:
        env = "testnet"
        base_dir = "data"

    rows = _collect_closed_trades(base_dir, env)
    metrics = _rolling_metrics_from_logs(base_dir, env, window=30, rows=rows)
    win_rate_pct = metrics.get("win_rate_pct", 0.0)
    avg_r_atr = metrics.get("avg_r_atr", None)
    expectancy_usd = metrics.get("expectancy_usd", 0.0)
    fallback_count = int(metrics.get("fallback_percent_count", 0))
    N = int(metrics.get("N", 0))

    side_disp = (side or "").upper()
    computed_r = _compute_latest_r_atr(symbol, side, entry_price, exit_price, rows)
    use_direct_r = (avg_r_atr is None or (isinstance(avg_r_atr, float) and avg_r_atr != avg_r_atr)) or N <= 1
    if use_direct_r and computed_r is not None:
        avg_r_value = computed_r
    else:
        avg_r_value = avg_r_atr if avg_r_atr is not None else computed_r
    avg_r_text = "N/A"
    if avg_r_value is not None and not (isinstance(avg_r_value, float) and avg_r_value != avg_r_value):
        avg_r_text = f"{float(avg_r_value):.3f}"
    dd_text = "N/A"  # daily drawdown not provided here
    msg = (
        f":white_check_mark: EXIT {symbol} {side_disp}\n"
        f"Entry: {entry_price:.4f} -> Exit: {exit_price:.4f}\n"
        f"PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%)\n"
        f"Win rate(30): {win_rate_pct:.2f}% | Avg R: {avg_r_text} | Exp(USD 30): {float(expectancy_usd):.2f}\n"
        f"Equity: ${equity_usdt:,.2f} | DD: {dd_text}\n"
        f"fallback_trades_30: {fallback_count}/{N}"
    )
    return slack_notify_safely(msg)
