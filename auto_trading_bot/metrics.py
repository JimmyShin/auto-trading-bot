from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

import config
from .exchange_api import ExchangeAPI
from .state_store import StateStore

__all__ = [
    "start_metrics_server",
    "get_metrics_manager",
    "dump_current_metrics",
    "MetricsManager",
    "Metrics",
    "compute_daily_dd_ratio",
]


_logger = logging.getLogger(__name__)
_DEBUG_VALUES = {"1", "true", "yes", "on"}


def _debug_enabled() -> bool:
    return os.getenv("OBS_DEBUG_ALERTS", "").strip().lower() in _DEBUG_VALUES


def _safe_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _log_metrics_update(name: str, labels: Dict[str, Any], value: Any) -> None:
    if not _debug_enabled():
        return
    payload = {
        "name": name,
        "labels": labels,
        "value": _safe_value(value),
        "ts": time.time(),
    }
    try:
        _logger.info("METRICS_UPDATE %s", json.dumps(payload, sort_keys=True))
    except Exception:
        _logger.info("METRICS_UPDATE %s", payload)


_bot_equity = Gauge("bot_equity", "Current account equity (quote)", ["account"])
_bot_equity_quote = Gauge("bot_equity_quote", "Current account equity (quote currency)", ["account"])
_bot_daily_drawdown = Gauge("bot_daily_drawdown", "Daily drawdown ratio", ["account"])
_bot_daily_drawdown_ratio = Gauge("bot_daily_drawdown_ratio", "Daily drawdown ratio", ["account"])
_bot_avg_r_atr_30 = Gauge("bot_avg_r_atr_30", "Rolling average ATR-based R over 30 trades", ["account"])
_bot_trades_delta = Gauge("bot_trades_delta", "Trades delta within window", ["account"])
_bot_signals_delta = Gauge("bot_signals_delta", "Signals delta within window", ["account"])
_bot_window_trades = Gauge("bot_window_trades", "Closed trades in reporting window", ["account"])
_bot_time_drift_ms = Gauge("bot_time_drift_ms", "Clock drift in milliseconds", ["source"])
_bot_heartbeat_ts = Gauge("bot_heartbeat_ts", "Last heartbeat epoch seconds")
_bot_trade_count_total = Counter("bot_trade_count_total", "Cumulative trade count", ["account"])
_bot_signal_emitted_total = Counter(
    "bot_signal_emitted_total",
    "Signals emitted",
    ["symbol", "timeframe"],
)
_bot_order_errors_total = Counter(
    "bot_order_errors_total",
    "Order/auth/nonce errors",
    ["account", "reason"],
)
_bot_loop_latency_ms = Histogram(
    "bot_loop_latency_ms",
    "Main loop latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)


def _set_optional(metric: Gauge, value: Optional[float]) -> None:
    if value is None:
        return
    metric.set(float(value))


_STATE: Optional[StateStore] = None
_fetch_snapshot = None


def get_state() -> StateStore:
    global _STATE
    if _STATE is None:
        _STATE = StateStore(Path(config.STATE_STORE_PATH))
    return _STATE


def set_state_for_tests(state: StateStore) -> None:
    global _STATE
    _STATE = state


def set_snapshot_fetcher(fetcher):
    global _fetch_snapshot
    _fetch_snapshot = fetcher


def compute_dd_from_snapshot(snapshot, state: Optional[StateStore] = None) -> float:
    store = state or get_state()
    eq = max(0.0, float(snapshot.get("margin_balance", 0.0)))
    ts = snapshot.get("ts_utc")
    day = ts.date() if isinstance(ts, datetime) else datetime.now(timezone.utc).date()
    peak = store.update_daily_peak_equity(eq, day=day)
    if peak <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (eq / peak)))


def compute_daily_dd_ratio(*, state: Optional[StateStore] = None, snap: Optional[Dict[str, Any]] = None, exchange: Optional[ExchangeAPI] = None) -> float:
    store = state or get_state()
    snapshot = snap
    if snapshot is None:
        fetcher = _fetch_snapshot or (exchange or ExchangeAPI())
        exchange_snap = fetcher.fetch_equity_snapshot() if hasattr(fetcher, "fetch_equity_snapshot") else fetcher()
        snapshot = {
            "margin_balance": exchange_snap.margin_balance,
            "ts_utc": exchange_snap.ts_utc,
        }
    return compute_dd_from_snapshot(snapshot, state=store)


class MetricsManager:
    """Coordinates Prometheus metrics and provides state snapshots."""

    def __init__(self, account: str) -> None:
        self.account = account
        self._equity_metric = _bot_equity.labels(account=account)
        self._equity_quote_metric = _bot_equity_quote.labels(account=account)
        self._dd_metric = _bot_daily_drawdown.labels(account=account)
        self._dd_ratio_metric = _bot_daily_drawdown_ratio.labels(account=account)
        self._avg_r_metric = _bot_avg_r_atr_30.labels(account=account)
        self._trades_delta_metric = _bot_trades_delta.labels(account=account)
        self._signals_delta_metric = _bot_signals_delta.labels(account=account)
        self._window_trades_metric = _bot_window_trades.labels(account=account)
        self._trade_counter = _bot_trade_count_total.labels(account=account)
        self._heartbeat_metric = _bot_heartbeat_ts
        self._loop_hist = _bot_loop_latency_ms
        self._lock = threading.Lock()
        self._time_drift_metrics: Dict[str, Gauge] = {}
        self._state: Dict[str, Any] = {
            "equity": None,
            "daily_dd": None,
            "avg_r": None,
            "trade_total": 0.0,
            "signal_totals": {},
            "order_errors": {},
            "heartbeat_ts": None,
            "time_drift": {},
            "trades_delta": None,
            "signals_delta": None,
            "window_trades": None,
        }
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._store = StateStore(Path(config.STATE_STORE_PATH))
        self._exchange = ExchangeAPI()

    # --- Metrics setters -------------------------------------------------
    def set_equity(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._equity_metric.set(stored)
            self._equity_quote_metric.set(stored)
        with self._lock:
            self._state["equity"] = stored
        _log_metrics_update("bot_equity", {"account": self.account}, stored)
        _log_metrics_update("bot_equity_quote", {"account": self.account}, stored)

    def set_daily_drawdown(self, ratio: Optional[float]) -> None:
        if ratio is None or not math.isfinite(float(ratio)):
            stored = None
        else:
            stored = max(0.0, float(ratio))
            self._dd_metric.set(stored)
            self._dd_ratio_metric.set(stored)
        with self._lock:
            self._state["daily_dd"] = stored
        _log_metrics_update("bot_daily_drawdown", {"account": self.account}, stored)
        _log_metrics_update("bot_daily_drawdown_ratio", {"account": self.account}, stored)

    def set_avg_r(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._avg_r_metric.set(stored)
        with self._lock:
            self._state["avg_r"] = stored
        _log_metrics_update("bot_avg_r_atr_30", {"account": self.account}, stored)

    def set_trades_delta(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._trades_delta_metric.set(stored)
        with self._lock:
            self._state["trades_delta"] = stored
        _log_metrics_update("bot_trades_delta", {"account": self.account}, stored)

    def set_signals_delta(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._signals_delta_metric.set(stored)
        with self._lock:
            self._state["signals_delta"] = stored
        _log_metrics_update("bot_signals_delta", {"account": self.account}, stored)

    def set_window_trades(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._window_trades_metric.set(stored)
        with self._lock:
            self._state["window_trades"] = stored
        _log_metrics_update("bot_window_trades", {"account": self.account}, stored)

    def set_time_drift(self, source: str, drift_ms: float) -> None:
        metric = self._time_drift_metrics.get(source)
        if metric is None:
            metric = _bot_time_drift_ms.labels(source=source)
            self._time_drift_metrics[source] = metric
        metric.set(drift_ms)
        with self._lock:
            drift_map = dict(self._state.get("time_drift", {}))
            drift_map[source] = float(drift_ms)
            self._state["time_drift"] = drift_map
        _log_metrics_update("bot_time_drift_ms", {"source": source}, drift_ms)

    def inc_trade_count(self, amount: float = 1.0) -> None:
        self._trade_counter.inc(amount)
        _log_metrics_update("bot_trade_count_total", {"account": self.account}, amount)
        with self._lock:
            self._state["trade_total"] = float(self._state.get("trade_total", 0.0)) + float(amount)

    def inc_signal(self, symbol: str, timeframe: str, amount: float = 1.0) -> None:
        _bot_signal_emitted_total.labels(symbol=symbol, timeframe=timeframe).inc(amount)
        with self._lock:
            totals = dict(self._state.get("signal_totals", {}))
            key = (symbol, timeframe)
            totals[key] = float(totals.get(key, 0.0)) + float(amount)
            self._state["signal_totals"] = totals
            total_for_key = totals[key]
        _log_metrics_update("bot_signal_emitted_total", {"symbol": symbol, "timeframe": timeframe}, total_for_key)

    def inc_order_error(self, reason: str, amount: float = 1.0) -> None:
        _bot_order_errors_total.labels(account=self.account, reason=reason).inc(amount)
        with self._lock:
            totals = dict(self._state.get("order_errors", {}))
            totals[reason] = float(totals.get(reason, 0.0)) + float(amount)
            self._state["order_errors"] = totals
            total_for_reason = totals[reason]
        _log_metrics_update("bot_order_errors_total", {"account": self.account, "reason": reason}, total_for_reason)

    def observe_loop_latency(self, latency_ms: float) -> None:
        self._loop_hist.observe(latency_ms)

    # --- Heartbeat --------------------------------------------------------
    def set_heartbeat(self, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        self._heartbeat_metric.set(timestamp)
        with self._lock:
            self._state["heartbeat_ts"] = float(timestamp)
        _log_metrics_update("bot_heartbeat_ts", {}, timestamp)

    def start_heartbeat_thread(self, interval: float = 60.0) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        def _worker() -> None:
            while True:
                self.set_heartbeat()
                time.sleep(interval)

        thread = threading.Thread(target=_worker, name="metrics-heartbeat", daemon=True)
        thread.start()
        self._heartbeat_thread = thread

    # --- Access -----------------------------------------------------------
    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            state_copy = {
                "equity": self._state.get("equity"),
                "daily_dd": self._state.get("daily_dd"),
                "avg_r": self._state.get("avg_r"),
                "trade_total": self._state.get("trade_total"),
                "signal_totals": dict(self._state.get("signal_totals", {})),
                "order_errors": dict(self._state.get("order_errors", {})),
                "heartbeat_ts": self._state.get("heartbeat_ts"),
                "time_drift": dict(self._state.get("time_drift", {})),
            }
        return state_copy


_MANAGER: Optional[MetricsManager] = None
_SERVER_STARTED = False


def start_metrics_server(port: int = 9108, account: str = "live", *, start_heartbeat_thread: bool = True) -> MetricsManager:
    """Start Prometheus metrics server and return metrics manager."""

    global _MANAGER, _SERVER_STARTED
    if _MANAGER is None:
        _MANAGER = MetricsManager(account)
    if not _SERVER_STARTED:
        start_http_server(port)
        _SERVER_STARTED = True
    if start_heartbeat_thread:
        _MANAGER.start_heartbeat_thread()
    return _MANAGER


def get_metrics_manager() -> Optional[MetricsManager]:
    return _MANAGER


def dump_current_metrics() -> Dict[str, Any]:
    mgr = get_metrics_manager()
    if mgr is None:
        return {}
    snapshot = mgr.get_snapshot()
    snapshot["account"] = mgr.account
    return snapshot


class Metrics:
    """Stores normalized daily drawdown ratio (0..1) and exposes read helpers."""

    def __init__(self) -> None:
        self._dd_ratio: float = 0.0
        self._dd_ts: Optional[datetime] = None
        self._dd_equity: Optional[float] = None

    def update_daily_drawdown(self, ratio: float, *, equity: float, ts: datetime) -> None:
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("ts must be timezone-aware")
        if ts.utcoffset() != timedelta(0):
            raise ValueError("ts must be in UTC")

        ratio_clamped = max(0.0, min(1.0, float(ratio)))
        if self._dd_ts is not None and ts <= self._dd_ts:
            return

        self._dd_ratio = ratio_clamped
        self._dd_equity = float(equity)
        self._dd_ts = ts

    def bot_daily_drawdown(self) -> float:
        return self._dd_ratio

    def last_drawdown_timestamp(self) -> Optional[datetime]:
        return self._dd_ts

    def last_drawdown_equity(self) -> Optional[float]:
        return self._dd_equity
