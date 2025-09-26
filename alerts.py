from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any, Callable

import threading
import time
from collections import deque

import glob
import csv
import requests

from metrics import get_metrics_manager


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
    metrics_mgr = get_metrics_manager()
    if metrics_mgr:
        try:
            metrics_mgr.set_avg_r(float(avg_r_value) if avg_r_value is not None else float("nan"))
        except Exception:
            pass
    return slack_notify_safely(msg)


class AlertScheduler(threading.Thread):
    """Background scheduler for observability alerts."""

    def __init__(
        self,
        metrics_manager,
        config: Dict[str, Any],
        slack_sender: Callable[[str], bool] = slack_notify_safely,
        *,
        interval_sec: float = 60.0,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(name="alert-scheduler", daemon=True)
        self.metrics = metrics_manager
        self.config = config or {}
        self.slack = slack_sender
        self.interval = interval_sec
        self.now_fn = now_fn or time.time
        self.cooldowns: Dict[str, float] = {}
        self.last_heartbeat_sent = 0.0
        self.last_heartbeat_trades = 0.0
        self.error_history: deque = deque()
        self.activity_history: deque = deque()
        self.drift_history: deque = deque()
        self._prev_error_totals: Dict[str, float] = {}

    # Thread loop ---------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - exercised in integration
        while True:
            self.run_step()
            time.sleep(self.interval)

    # Public helper for tests ---------------------------------------------
    def run_step(self) -> None:
        now = self.now_fn()
        snapshot = self.metrics.get_snapshot() if self.metrics else {}
        self._record_histories(now, snapshot)
        self._send_heartbeat_if_needed(now, snapshot)
        self._check_heartbeat_missing(now, snapshot)
        self._check_clock_drift(now)
        self._check_error_burst(now)
        self._check_no_trade(now)

    # Internal helpers ----------------------------------------------------
    def _record_histories(self, now: float, snap: Dict[str, Any]) -> None:
        signal_totals = snap.get("signal_totals", {}) or {}
        total_signals = float(sum(signal_totals.values()))
        trade_total = float(snap.get("trade_total", 0.0))
        self.activity_history.append((now, total_signals, trade_total))
        window = float(self.config.get("no_trade_window_sec", 7200))
        while self.activity_history and now - self.activity_history[0][0] > window:
            self.activity_history.popleft()

        current_errors = {k: float(v) for k, v in (snap.get("order_errors", {}) or {}).items()}
        delta_errors: Dict[str, float] = {}
        for reason, total in current_errors.items():
            previous = float(self._prev_error_totals.get(reason, 0.0))
            delta = max(0.0, total - previous)
            if delta:
                delta_errors[reason] = delta
        self.error_history.append((now, delta_errors))
        self._prev_error_totals = current_errors
        error_window = float(self.config.get("error_burst_window_sec", 300))
        while self.error_history and now - self.error_history[0][0] > error_window:
            self.error_history.popleft()

        drift_map = snap.get("time_drift", {}) or {}
        if "exchange" in drift_map:
            self.drift_history.append((now, float(drift_map["exchange"])))
        drift_window = 180.0
        while self.drift_history and now - self.drift_history[0][0] > drift_window:
            self.drift_history.popleft()

        # Update cached heartbeat time if snapshot provided
        # No-op: heartbeat timestamp used in downstream checks

    def _send_heartbeat_if_needed(self, now: float, snap: Dict[str, Any]) -> None:
        interval = float(self.config.get("heartbeat_interval_sec", 43200))
        if interval <= 0:
            return
        if now - self.last_heartbeat_sent < interval:
            return
        account = self.config.get("account_label", "live")
        run_id = (os.getenv("GITHUB_SHA") or os.getenv("RUN_ID") or "local")[:7]
        equity = float(snap.get("equity", 0.0))
        avg_r = snap.get("avg_r", float("nan"))
        avg_r_text = "N/A" if isinstance(avg_r, float) and avg_r != avg_r else f"{avg_r:.3f}"
        trade_total = float(snap.get("trade_total", 0.0))
        trade_delta = trade_total - self.last_heartbeat_trades
        heartbeat_ts = float(snap.get("heartbeat_ts", now))
        message = (
            f"HB ✅ run:{run_id} account:{account} eq:{equity:.2f} avgR(ATR30):{avg_r_text} "
            f"tradesΔ:{int(trade_delta)} ts:{int(heartbeat_ts)}"
        )
        if self.slack(message):
            self.last_heartbeat_sent = now
            self.last_heartbeat_trades = trade_total

    def _check_heartbeat_missing(self, now: float, snap: Dict[str, Any]) -> None:
        missing_sec = float(self.config.get("heartbeat_missing_sec", 64800))
        heartbeat_ts = float(snap.get("heartbeat_ts", 0.0))
        if not heartbeat_ts:
            return
        if now - heartbeat_ts <= missing_sec:
            return
        if self._should_alert("heartbeat_missing", now):
            self.slack("⚠️ Heartbeat missing: no updates in the last period. Check bot loop/metrics.")

    def _check_clock_drift(self, now: float) -> None:
        if not self.drift_history:
            return
        threshold = float(self.config.get("clock_drift_warn_ms", 5000.0))
        window = 180.0
        recent = [entry for entry in self.drift_history if now - entry[0] <= window]
        if len(recent) < max(3, int(window // max(self.interval, 1))):
            return
        if all(abs(val) >= threshold for _, val in recent):
            latest = recent[-1][1]
            if self._should_alert("clock_drift", now):
                direction = "+" if latest >= 0 else ""
                self.slack(f"⚠️ Clock drift: {direction}{int(latest)}ms (>{int(threshold)}). Check NTP/exchange sync.")

    def _check_error_burst(self, now: float) -> None:
        if not self.error_history:
            return
        threshold = float(self.config.get("error_burst_threshold", 10))
        window = float(self.config.get("error_burst_window_sec", 300))
        totals: Dict[str, float] = {}
        total_errors = 0.0
        for ts, delta_map in self.error_history:
            if now - ts > window:
                continue
            for reason, value in delta_map.items():
                totals[reason] = totals.get(reason, 0.0) + value
                total_errors += value
        if total_errors >= threshold and self._should_alert("error_burst", now):
            top_reason = max(totals.items(), key=lambda item: item[1])[0] if totals else "unknown"
            self.slack(
                f"⚠️ Error burst: {int(total_errors)} errors/{int(window/60)}m ({top_reason}). Investigate failures."
            )

    def _check_no_trade(self, now: float) -> None:
        if not self.activity_history:
            return
        window = float(self.config.get("no_trade_window_sec", 7200))
        min_signals = float(self.config.get("no_trade_signals_min", 5))
        signals_start, trades_start = None, None
        for ts, sig_total, trade_total in self.activity_history:
            if now - ts <= window:
                signals_start = sig_total if signals_start is None else signals_start
                trades_start = trade_total if trades_start is None else trades_start
                break
        if signals_start is None or trades_start is None:
            return
        _, signals_end, trades_end = self.activity_history[-1]
        signal_delta = signals_end - signals_start
        trade_delta = trades_end - trades_start
        if signal_delta >= min_signals and trade_delta <= 0 and self._should_alert("no_trade", now):
            self.slack(
                f"⚠️ Signals: {int(signal_delta)} in {int(window/3600)}h but no trades. Review filters/routing."
            )

    def _should_alert(self, key: str, now: float) -> bool:
        cooldown = float(self.config.get("alert_cooldown_sec", 600))
        last = self.cooldowns.get(key)
        if last is not None and now - last < cooldown:
            return False
        self.cooldowns[key] = now
        return True


def start_alert_scheduler(
    config: Dict[str, Any],
    metrics_manager=None,
    *,
    slack_sender: Callable[[str], bool] = slack_notify_safely,
    interval_sec: float = 60.0,
) -> AlertScheduler:
    scheduler = AlertScheduler(metrics_manager or get_metrics_manager(), config, slack_sender, interval_sec=interval_sec)
    scheduler.start()
    return scheduler
