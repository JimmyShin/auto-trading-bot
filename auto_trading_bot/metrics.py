from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

__all__ = [
    "start_metrics_server",
    "get_metrics_manager",
    "dump_current_metrics",
    "MetricsManager",
    "Metrics",
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
_bot_daily_drawdown = Gauge("bot_daily_drawdown", "Daily drawdown ratio", ["account"])
_bot_avg_r_atr_30 = Gauge("bot_avg_r_atr_30", "Rolling average ATR-based R over 30 trades", ["account"])
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


class MetricsManager:
    """Coordinates Prometheus metrics and provides state snapshots."""

    def __init__(self, account: str) -> None:
        self.account = account
        self._equity_metric = _bot_equity.labels(account=account)
        self._dd_metric = _bot_daily_drawdown.labels(account=account)
        self._avg_r_metric = _bot_avg_r_atr_30.labels(account=account)
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
        }
        self._heartbeat_thread: Optional[threading.Thread] = None

    # --- Metrics setters -------------------------------------------------
    def set_equity(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._equity_metric.set(stored)
        with self._lock:
            self._state["equity"] = stored
        _log_metrics_update("bot_equity", {"account": self.account}, stored)

    def set_daily_drawdown(self, ratio: Optional[float]) -> None:
        if ratio is None or not math.isfinite(float(ratio)):
            stored = None
        else:
            stored = max(0.0, float(ratio))
            self._dd_metric.set(stored)
        with self._lock:
            self._state["daily_dd"] = stored
        _log_metrics_update("bot_daily_drawdown", {"account": self.account}, stored)

    def set_avg_r(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(float(value)):
            stored = None
        else:
            stored = float(value)
            self._avg_r_metric.set(stored)
        with self._lock:
            self._state["avg_r"] = stored
        _log_metrics_update("bot_avg_r_atr_30", {"account": self.account}, stored)

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
