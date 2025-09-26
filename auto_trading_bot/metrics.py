from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

__all__ = [
    "start_metrics_server",
    "get_metrics_manager",
    "MetricsManager",
]


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
            "equity": 0.0,
            "daily_dd": 0.0,
            "avg_r": float("nan"),
            "trade_total": 0,
            "signal_totals": {},
            "order_errors": {},
            "heartbeat_ts": 0.0,
            "time_drift": {},
        }
        self._heartbeat_thread: Optional[threading.Thread] = None

    # --- Metrics setters -------------------------------------------------
    def set_equity(self, value: float) -> None:
        self._equity_metric.set(value)
        with self._lock:
            self._state["equity"] = value

    def set_daily_drawdown(self, ratio: float) -> None:
        ratio = max(0.0, float(ratio))
        self._dd_metric.set(ratio)
        with self._lock:
            self._state["daily_dd"] = ratio

    def set_avg_r(self, value: float) -> None:
        self._avg_r_metric.set(value)
        with self._lock:
            self._state["avg_r"] = value

    def set_time_drift(self, source: str, drift_ms: float) -> None:
        metric = self._time_drift_metrics.get(source)
        if metric is None:
            metric = _bot_time_drift_ms.labels(source=source)
            self._time_drift_metrics[source] = metric
        metric.set(drift_ms)
        with self._lock:
            drift_map = dict(self._state.get("time_drift", {}))
            drift_map[source] = drift_ms
            self._state["time_drift"] = drift_map

    def inc_trade_count(self, amount: float = 1.0) -> None:
        self._trade_counter.inc(amount)
        with self._lock:
            self._state["trade_total"] = float(self._state.get("trade_total", 0)) + amount

    def inc_signal(self, symbol: str, timeframe: str, amount: float = 1.0) -> None:
        _bot_signal_emitted_total.labels(symbol=symbol, timeframe=timeframe).inc(amount)
        with self._lock:
            totals = dict(self._state.get("signal_totals", {}))
            key = (symbol, timeframe)
            totals[key] = float(totals.get(key, 0.0)) + amount
            self._state["signal_totals"] = totals

    def inc_order_error(self, reason: str, amount: float = 1.0) -> None:
        _bot_order_errors_total.labels(account=self.account, reason=reason).inc(amount)
        with self._lock:
            totals = dict(self._state.get("order_errors", {}))
            totals[reason] = float(totals.get(reason, 0.0)) + amount
            self._state["order_errors"] = totals

    def observe_loop_latency(self, latency_ms: float) -> None:
        self._loop_hist.observe(latency_ms)

    # --- Heartbeat --------------------------------------------------------
    def set_heartbeat(self, timestamp: Optional[float] = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        self._heartbeat_metric.set(timestamp)
        with self._lock:
            self._state["heartbeat_ts"] = float(timestamp)

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
                "equity": self._state.get("equity", 0.0),
                "daily_dd": self._state.get("daily_dd", 0.0),
                "avg_r": self._state.get("avg_r", float("nan")),
                "trade_total": float(self._state.get("trade_total", 0.0)),
                "signal_totals": dict(self._state.get("signal_totals", {})),
                "order_errors": dict(self._state.get("order_errors", {})),
                "heartbeat_ts": float(self._state.get("heartbeat_ts", 0.0)),
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

