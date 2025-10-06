from __future__ import annotations

import argparse

# Sample Slack payloads (one-liners):
# HB ✅ run:abc123 acct:testnet eq:12345.67 rATR30:NA tradesΔ:0 drift:12ms ts:1700000000
# ✅ EXIT BTC/USDT SHORT run:abc123 acct:testnet pnl:+42.00USDT R:2.10 eq_after:10123.00
# ⚠️ Clock drift: +7421ms (>5000). Check NTP/exchange time. run:abc123 acct:testnet
# Runbook:
#   - Enable debug: OBS_DEBUG_ALERTS=1
#   - Manual probes:
#       * python -m auto_trading_bot.alerts --debug-dump
#       * curl 127.0.0.1:${METRICS_PORT}/metrics | grep -E "bot_(equity|daily_drawdown|avg_r_atr_30|trade_count_total|signal_emitted_total|time_drift_ms|heartbeat_ts)"

# 🚨 AUTO_TESTNET_ON_DD tripped: dd:12.0% threshold:10.0%. Live entries paused. run:abc123 acct:testnet

import csv
import glob
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import requests

from auto_trading_bot.metrics import get_metrics_manager, compute_daily_dd_ratio, get_state
from auto_trading_bot import reporter
from auto_trading_bot.slack_notifier import SlackNotifier
from auto_trading_bot.slack_fmt import fmt_optional, fmt_currency, fmt_percent_ratio, fmt_int, fmt_float2, DASH
from auto_trading_bot.mode import TradingMode
import config
from auto_trading_bot.state_store import StateStore

logger = logging.getLogger(__name__)

_DEBUG_VALUES = {"1", "true", "yes", "on"}
_ALERT_SOURCE = "alerts"

def _debug_enabled() -> bool:
    return os.getenv("OBS_DEBUG_ALERTS", "").strip().lower() in _DEBUG_VALUES

def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def _log_alerts_event(event: str, payload: Dict[str, Any]) -> None:
    if not _debug_enabled():
        return
    body = {"source": _ALERT_SOURCE}
    body.update(payload)
    try:
        logger.info("%s %s", event, json.dumps(body, sort_keys=True, default=_safe_json_value))
    except Exception:
        logger.info("%s %s", event, body)

SENTINEL_EXIT_PRICES = {200.25, 200.2500, 0.0}
def _emergency_window_seconds() -> float:
    try:
        value = float(os.getenv("EMERGENCY_ALERT_WINDOW_SEC", "180"))
    except (TypeError, ValueError):
        value = 180.0
    return max(0.0, min(value, 1800.0))


EMERGENCY_MIN_WINDOW_SEC = _emergency_window_seconds()
RUN_ID_LENGTH = 7


_EXIT_EQUITY_WARNED = False


_SNAPSHOT_THREADS: Dict[str, str] = {}


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _aggregate_totals(values: Dict[Any, Any]) -> Optional[float]:
    if not values:
        return None
    total = 0.0
    seen = False
    for val in values.values():
        num = _safe_number(val)
        if num is None:
            continue
        total += num
        seen = True
    return total if seen else None


def _build_alert_payload(
    snapshot: Dict[str, Any],
    *,
    trades_delta: Optional[float] = None,
    signals_delta: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "acct": CONTEXT.account_label,
        "equity": _safe_number(snapshot.get("equity")),
        "daily_dd": _safe_number(snapshot.get("daily_dd")),
        "avg_r_atr_30": _safe_number(snapshot.get("avg_r")),
        "trades_delta": _safe_number(trades_delta),
        "signals_delta": _safe_number(signals_delta),
        "trade_count_total": _safe_number(snapshot.get("trade_total")),
        "signal_emitted_total": _aggregate_totals(snapshot.get("signal_totals", {})),
        "order_errors_total": _aggregate_totals(snapshot.get("order_errors", {})),
        "time_drift_ms": _safe_number((snapshot.get("time_drift") or {}).get("exchange")),
    }
    if extra:
        payload.update(extra)
    return payload


def _log_alert_payload(snapshot: Dict[str, Any], *, trades_delta: Optional[float] = None, signals_delta: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if not _debug_enabled():
        return
    payload = _build_alert_payload(snapshot, trades_delta=trades_delta, signals_delta=signals_delta, extra=extra)
    _log_alerts_event("ALERTS_INPUT", payload)


_SLACK: Optional[SlackNotifier] = None  # lazy singleton


def _get_notifier() -> SlackNotifier:
    global _SLACK
    if _SLACK is None:
        _SLACK = SlackNotifier()
    return _SLACK


def _send_with_blocks(sender: Callable[..., bool], text: str, blocks: Optional[List[Dict[str, Any]]] = None) -> bool:
    try:
        return sender(text, blocks=blocks)
    except TypeError:
        if blocks:
            return sender(text)
        raise


def slack_notify_safely(message: str, *, blocks: Optional[List[Dict[str, Any]]] = None) -> bool:
    notifier = _get_notifier()
    if not blocks:
        return notifier.send(message)
    if notifier.send(message, blocks=blocks):
        return True
    return notifier.send(message)


def send_to_slack(payload: Dict[str, Any], *, channel: str = "#trading-ops", thread_ts: Optional[str] = None, sender: Optional[Callable[..., bool]] = None) -> bool:
    sender = sender or slack_notify_safely
    message_text = (payload or {}).get("text") or ""
    blocks = (payload or {}).get("blocks")
    channel_prefix = f"[{channel}] " if channel else ""
    formatted = CONTEXT.format_text(f"{channel_prefix}{message_text}")
    if thread_ts:
        formatted = f"{formatted}\n(thread:{thread_ts})"
    success = sender(formatted, blocks=blocks)
    if success:
        logger.info("Slack notification sent", extra={"event": "snapshot_posted", "channel": channel})
    else:
        logger.warning("Slack notification failed", extra={"event": "snapshot_failed", "channel": channel})
    return success


def get_last_snapshot_thread(env: str) -> Optional[str]:
    return _SNAPSHOT_THREADS.get(env)


def send_daily_snapshot(env: str, summary: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    payload = reporter.build_snapshot_payload(env, summary or {}, metrics or {})
    thread_hint = metrics.get("thread_ts") or f"{CONTEXT.thread_key('SNAPSHOT')}:{int(time.time())}"
    if send_to_slack(payload, channel="#trading-ops"):
        _SNAPSHOT_THREADS[env] = thread_hint
        logger.info("Daily snapshot dispatched", extra={"event": "daily_snapshot", "env": env})
    else:
        logger.warning("Failed to dispatch daily snapshot", extra={"event": "daily_snapshot_failed", "env": env})


def send_weekly_summary(env: str, stats: Dict[str, Any]) -> None:
    payload = reporter.build_weekly_summary(env, stats or {})
    thread_ts = get_last_snapshot_thread(env)
    if send_to_slack(payload, channel="#trading-ops", thread_ts=thread_ts):
        logger.info("Weekly summary dispatched", extra={"event": "weekly_summary", "env": env})
    else:
        logger.warning("Failed to dispatch weekly summary", extra={"event": "weekly_summary_failed", "env": env})


@dataclass
class AlertContext:
    account_label: str
    server_env: str
    run_id: str
    slack_prefix: str = ""

    def format_text(self, text: str) -> str:
        parts: List[str] = []
        if self.slack_prefix:
            parts.append(self.slack_prefix.strip())
        if self.server_env and self.server_env.lower() != "prod":
            parts.append(f"[{self.server_env}]")
        prefix = " ".join(parts)
        if prefix:
            return f"{prefix} {text}"
        return text

    def thread_key(self, event_type: str) -> str:
        return f"{self.run_id}:{event_type}"


def _generate_run_id() -> str:
    for candidate in (os.getenv("OBS_RUN_ID"), os.getenv("RUN_ID"), os.getenv("GITHUB_SHA")):
        if candidate:
            candidate = candidate.strip()
            if candidate:
                return candidate[:RUN_ID_LENGTH]
    return format(int(time.time()), "x")[:RUN_ID_LENGTH]


def _load_context() -> AlertContext:
    try:
        import config as cfg

        obs = getattr(cfg, "OBSERVABILITY", {}) or {}
    except Exception:
        cfg = None
        obs = {}

    allowed_envs = {"prod", "staging", "dev", "local"}
    server_env_value = os.getenv("OBS_SERVER_ENV") or obs.get("server_env")
    if not server_env_value and cfg and hasattr(cfg, "OBS_SERVER_ENV"):
        server_env_value = getattr(cfg, "OBS_SERVER_ENV")
    server_env = (server_env_value or "local").strip().lower()
    if server_env not in allowed_envs:
        server_env = "local"

    env_account = os.getenv("OBS_ACCOUNT_LABEL")
    if env_account is not None:
        raw_account = env_account.strip()
    else:
        obs_account = obs.get("account_label") if obs else ""
        candidate = str(obs_account or (getattr(cfg, "OBS_ACCOUNT_LABEL", "") if cfg and hasattr(cfg, "OBS_ACCOUNT_LABEL") else "")).strip()
        if candidate and not (candidate.lower() == "live" and server_env != "prod"):
            raw_account = candidate
        else:
            raw_account = ""

    if raw_account:
        account = raw_account
    else:
        if cfg and hasattr(cfg, "TESTNET"):
            testnet_flag = bool(getattr(cfg, "TESTNET"))
        else:
            testnet_flag = os.getenv("TESTNET", "true").strip().lower() in {"1", "true", "yes", "on"}
        if testnet_flag:
            account = "testnet"
        elif server_env == "prod":
            account = "live"
        else:
            account = "testnet"

    env_slack_prefix = os.getenv("OBS_SLACK_PREFIX")
    if env_slack_prefix is not None:
        slack_prefix = env_slack_prefix.strip()
    else:
        slack_prefix = str(obs.get("slack_prefix") or (getattr(cfg, "OBS_SLACK_PREFIX", "") if cfg and hasattr(cfg, "OBS_SLACK_PREFIX") else "")).strip()

    run_id = _generate_run_id()
    return AlertContext(account_label=account, server_env=server_env, run_id=run_id, slack_prefix=slack_prefix)


CONTEXT = _load_context()


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _format_decimal(value: Optional[Any], digits: int = 2, fallback: str = "NA") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(val):
        return fallback
    return f"{val:.{digits}f}"


def _format_int(value: Optional[Any], fallback: str = "NA") -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(val):
        return fallback
    return str(int(round(val)))


def _format_percent(value: Optional[Any], digits: int = 1, fallback: str = "NA") -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(val):
        return fallback
    return f"{val * 100:.{digits}f}%"


def _format_signed(value: Optional[Any], digits: int = 2, suffix: str = "", fallback: str = "NA") -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(val):
        return fallback
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{digits}f}{suffix}"


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value.replace("Z", ""), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "NA"
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{sec}s")
    return ''.join(parts)


def _collect_closed_trades(base_dir: str, env: str) -> List[Dict[str, Any]]:
    """Collect closed trades across data/<env>/trades_*.csv."""
    root = os.path.join(base_dir, env)
    patterns = [os.path.join(root, "trades_*.csv")]
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    out: List[Dict[str, Any]] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if (row.get("status") or "").upper() != "CLOSED":
                        continue
                    try:
                        entry = float(row.get("entry_price") or 0.0)
                        exit_price = float(row.get("exit_price") or 0.0)
                        qty = float(row.get("qty") or 0.0)
                    except Exception:
                        continue
                    if entry <= 0 or exit_price <= 0:
                        continue
                    side = (row.get("side") or "").strip().lower()

                    def _safe_num(key: str) -> Optional[float]:
                        try:
                            return float(row.get(key) or "")
                        except Exception:
                            return None

                    out.append({
                        "symbol": row.get("symbol") or "",
                        "side": side,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "qty": qty,
                        "entry_ts_utc": row.get("entry_ts_utc") or row.get("entry_timestamp") or "",
                        "exit_ts_utc": row.get("exit_ts_utc") or row.get("exit_timestamp") or "",
                        "R_atr_expost": _safe_num("R_atr_expost"),
                        "R_usd_expost": _safe_num("R_usd_expost"),
                        "pnl_quote_expost": _safe_num("pnl_quote_expost"),
                        "fees_quote_actual": _safe_num("fees_quote_actual"),
                        "stop_k": _safe_num("stop_k"),
                        "entry_atr_abs": _safe_num("entry_atr_abs"),
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
        import pandas as pd  # type: ignore
        from auto_trading_bot.reporter import generate_report

        df = pd.DataFrame(rows)
        if len(df) > window:
            df = df.tail(window)
        report = generate_report(df)
        data = report.iloc[0].to_dict() if not report.empty else {}
        return {
            "win_rate_pct": float(data.get("win_rate", 0.0)) * 100.0,
            "avg_r_atr": data.get("avg_r_atr"),
            "avg_r_usd": data.get("avg_r_usd"),
            "expectancy_usd": float(data.get("expectancy_usd", 0.0)),
            "fallback_percent_count": int(data.get("fallback_percent_count", 0)),
            "N": len(df),
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


def _compute_latest_r_atr(symbol: str, side: str, entry_price: float, exit_price: float, rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    mult = 1.0 if (side or "").lower() == "long" else -1.0
    for row in reversed(list(rows)):
        if row.get("symbol") != symbol:
            continue
        try:
            stop_k = float(row.get("stop_k") or 0.0)
            atr_abs = float(row.get("entry_atr_abs") or 0.0)
        except Exception:
            continue
        denom = stop_k * atr_abs
        if denom <= 0:
            continue
        pnl_dist = (exit_price - entry_price) * mult
        return pnl_dist / denom
    return None


def _extract_trade_record(symbol: str, side: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    side = (side or "").lower()
    for row in reversed(rows):
        if row.get("symbol") == symbol and (row.get("side") or "").lower() == side:
            return row
    return None


def _derive_fill_id(record: Dict[str, Any]) -> Optional[str]:
    for key in ("exit_ts_utc", "exit_timestamp"):
        val = record.get(key)
        if val:
            return str(val)
    return None


def _compute_holding_seconds(record: Dict[str, Any]) -> Optional[float]:
    entry_ts = _parse_timestamp(str(record.get("entry_ts_utc") or ""))
    exit_ts = _parse_timestamp(str(record.get("exit_ts_utc") or ""))
    if entry_ts and exit_ts:
        return (exit_ts - entry_ts).total_seconds()
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
    *,
    fill_id: Optional[str] = None,
    fill_source: str = "exchange",
    fees_quote: Optional[float] = None,
    sender: Optional[Callable[..., bool]] = None,
    reason_code: str = "auto_exit",
) -> bool:
    """Send a Slack message about a confirmed exit."""
    global _EXIT_EQUITY_WARNED
    if exit_price in SENTINEL_EXIT_PRICES:
        logger.warning("Suppressing exit slack for sentinel exit price %s", exit_price)
        return False

    sender = sender or slack_notify_safely

    try:
        import config as cfg

        env = "testnet" if getattr(cfg, "TESTNET", True) else "live"
        base_dir = getattr(cfg, "DATA_BASE_DIR", "data")
    except Exception:
        env = "testnet"
        base_dir = "data"

    rows = _collect_closed_trades(base_dir, env)
    record = _extract_trade_record(symbol, side, rows) if rows else None
    if record is None:
        logger.warning("Suppressing exit slack for %s %s: no trade record", symbol, side)
        return False

    derived_fill_id = fill_id or _derive_fill_id(record)
    if not derived_fill_id:
        logger.warning("Suppressing exit slack for %s %s: missing fill id", symbol, side)
        return False

    trade_qty = record.get("qty") or qty
    try:
        trade_qty = float(trade_qty)
    except Exception:
        trade_qty = qty
    if not trade_qty or trade_qty <= 0:
        logger.warning("Suppressing exit slack for %s %s: invalid qty", symbol, side)
        return False

    r_trade = record.get("R_atr_expost")
    if r_trade is None or _is_nan(r_trade):
        r_trade = _compute_latest_r_atr(symbol, side, entry_price, exit_price, rows)

    avg_metrics = _rolling_metrics_from_logs(base_dir, env, rows=rows)
    holding_seconds = _compute_holding_seconds(record)
    fees_value = fees_quote if fees_quote is not None else record.get("fees_quote_actual")

    equity_value = _safe_number(equity_usdt)
    if not equity_value or equity_value <= 0:
        equity_text = DASH
        if not _EXIT_EQUITY_WARNED:
            logger.warning("Exit equity unavailable; displaying en dash (—) in Slack payload.")
            _EXIT_EQUITY_WARNED = True
    else:
        equity_text = f"{equity_value:.2f}"

    if _debug_enabled():
        payload = {
            "acct": CONTEXT.account_label,
            "equity": equity_value,
            "daily_dd": None,
            "avg_r_atr_30": _safe_number(avg_metrics.get("avg_r_atr")),
            "trades_delta": None,
            "signals_delta": None,
            "trade_count_total": None,
            "signal_emitted_total": None,
            "order_errors_total": None,
            "time_drift_ms": None,
            "event": "exit",
            "symbol": symbol,
            "side": side,
            "fill_id": derived_fill_id,
        }
        _log_alerts_event("ALERTS_INPUT", payload)

    message = (
        f"✅ EXIT {symbol} {side.upper()} run:{CONTEXT.run_id} acct:{CONTEXT.account_label} "
        f"pnl:{_format_signed(pnl_usdt, suffix='USDT')} R:{_format_decimal(r_trade, digits=2)} "
        f"eq_after:{equity_text}"
    )

    blocks: List[Dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"✅ EXIT {symbol} {side.upper()}", "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Entry*\n{_format_decimal(entry_price, 4)}"},
                {"type": "mrkdwn", "text": f"*Exit*\n{_format_decimal(exit_price, 4)}"},
                {"type": "mrkdwn", "text": f"*PnL_quote*\n{_format_signed(pnl_usdt, suffix=' USDT')}"},
                {"type": "mrkdwn", "text": f"*R*\n{_format_decimal(r_trade, 2)}"},
                {"type": "mrkdwn", "text": f"*Holding*\n{_format_duration(holding_seconds)}"},
                {"type": "mrkdwn", "text": f"*Fees*\n{_format_decimal(fees_value, 2)}"},
                {"type": "mrkdwn", "text": f"*Equity_after*\n{equity_text}"},
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"run:{CONTEXT.run_id} acct:{CONTEXT.account_label} "
                        f"thread:{CONTEXT.thread_key('EXIT')} source:{fill_source}"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": f"fill_id:{derived_fill_id}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"AvgR30:{_format_decimal(avg_metrics.get('avg_r_atr'), 2)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"reason_code:{reason_code}",
                },
            ],
        },
    ]

    metrics_mgr = get_metrics_manager()
    if metrics_mgr:
        try:
            metrics_mgr.set_avg_r(float(r_trade) if r_trade is not None else float("nan"))
        except Exception:
            pass

    formatted = CONTEXT.format_text(message)
    return _send_with_blocks(sender, formatted, blocks=blocks)


@dataclass
class EmergencyState:
    fail_count: int = 0
    first_fail_ts: Optional[float] = None
    last_alert_ts: Optional[float] = None
    last_alert_severity: Optional[float] = None


class AlertScheduler(threading.Thread):
    """Background scheduler for observability alerts."""

    def __init__(
        self,
        metrics_manager,
        config: Dict[str, Any],
        slack_sender: Callable[..., bool] = slack_notify_safely,
        *,
        interval_sec: float = 60.0,
        now_fn: Optional[Callable[[], float]] = None,
        guard_action: Optional[Callable[[float], None]] = None,
    ) -> None:
        super().__init__(name="alert-scheduler", daemon=True)
        self.metrics = metrics_manager
        self.config = config or {}
        self.slack = slack_sender
        self.interval = interval_sec
        self.now_fn = now_fn or time.time
        self.cooldowns: Dict[str, float] = {}
        self.emergencies: Dict[str, EmergencyState] = {}
        self.last_heartbeat_sent = 0.0
        self.last_heartbeat_trades: Optional[float] = None
        self.error_history: Deque[Tuple[float, Dict[str, float]]] = deque()
        self.activity_history: Deque[Tuple[float, float, Optional[float]]] = deque()
        self._prev_error_totals: Dict[str, float] = {}
        self._equity_warned = False
        self._guard_action = guard_action

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
        self._check_clock_drift(now, snapshot)
        self._check_error_burst(now)
        self._check_no_trade(now)
        self._check_auto_testnet(now, snapshot)

    # Internal helpers ----------------------------------------------------
    def _record_histories(self, now: float, snap: Dict[str, Any]) -> None:
        signal_totals = snap.get("signal_totals") or {}
        total_signals = float(sum(signal_totals.values())) if signal_totals else 0.0
        trade_total_raw = snap.get("trade_total")
        trade_total: Optional[float]
        if trade_total_raw is None:
            trade_total = None
        else:
            try:
                trade_total = float(trade_total_raw)
            except (TypeError, ValueError):
                trade_total = None
        self.activity_history.append((now, total_signals, trade_total))
        window = float(self.config.get("no_trade_window_sec", 7200))
        while self.activity_history and now - self.activity_history[0][0] > window:
            self.activity_history.popleft()

        current_errors = {k: float(v) for k, v in (snap.get("order_errors") or {}).items()}
        delta_errors: Dict[str, float] = {}
        for reason, total in current_errors.items():
            previous = float(self._prev_error_totals.get(reason, 0.0))
            delta = total - previous
            if delta > 0:
                delta_errors[reason] = delta
        self.error_history.append((now, delta_errors))
        self._prev_error_totals = current_errors
        error_window = float(self.config.get("error_burst_window_sec", 300))
        while self.error_history and now - self.error_history[0][0] > error_window:
            self.error_history.popleft()

    def _signal_trade_delta(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.activity_history:
            return None, None
        start_sig = self.activity_history[0][1]
        end_sig = self.activity_history[-1][1]
        signals_delta = end_sig - start_sig
        start_trade = self.activity_history[0][2]
        end_trade = self.activity_history[-1][2]
        trades_delta = None
        if start_trade is not None and end_trade is not None:
            trades_delta = end_trade - start_trade
        return signals_delta, trades_delta

    def _send_heartbeat_if_needed(self, now: float, snap: Dict[str, Any]) -> None:
        interval = float(self.config.get("heartbeat_interval_sec", 43200))
        if interval <= 0:
            return
        if now - self.last_heartbeat_sent < interval:
            return

        equity_value = _safe_number(snap.get("equity"))
        if not equity_value or equity_value <= 0:
            equity_text = DASH
            if not self._equity_warned:
                logger.warning("Heartbeat equity unavailable; awaiting metrics update before displaying equity.")
                self._equity_warned = True
        else:
            equity_text = fmt_currency(equity_value)
            self._equity_warned = False

        avg_r_value = _safe_number(snap.get("avg_r"))
        avg_r_text = fmt_optional(avg_r_value, fmt_float2)

        trade_total_raw = snap.get("trade_total")
        trade_total_val: Optional[float] = None
        trade_delta_value: Optional[float] = None
        trade_delta_text = DASH
        if trade_total_raw is not None:
            try:
                trade_total_val = float(trade_total_raw)
                baseline = self.last_heartbeat_trades if self.last_heartbeat_trades is not None else trade_total_val
                trade_delta_value = trade_total_val - baseline
                if trade_delta_value < 0:
                    trade_delta_value = 0.0
                trade_delta_text = fmt_int(trade_delta_value)
            except (TypeError, ValueError):
                trade_total_val = None

        drift_val = (snap.get("time_drift") or {}).get("exchange")
        drift_text = fmt_optional(_safe_number(drift_val), fmt_int)

        ts_value = snap.get("heartbeat_ts") or now
        ts_text = fmt_int(ts_value)

        signals_delta, trades_delta_window = self._signal_trade_delta()
        signals_val = _safe_number(signals_delta)
        trades_window_val = _safe_number(trades_delta_window)
        dd_val = _safe_number(snap.get("daily_dd"))

        _log_alert_payload(
            snap,
            trades_delta=trade_delta_value,
            signals_delta=signals_val,
            extra={"event": "heartbeat", "heartbeat_ts": _safe_number(ts_value)},
        )

        mode_label = config.get_trading_mode()
        body_lines = [
            "Equity",
            equity_text,
            "Daily DD",
            fmt_optional(dd_val, fmt_percent_ratio),
            "Avg R(30)",
            avg_r_text,
            "Trades Δ",
            trade_delta_text,
            "Signals Δ",
            fmt_optional(signals_val, fmt_int),
            "Window trades",
            fmt_optional(trades_window_val, fmt_int),
            f"run:{CONTEXT.run_id} acct:{mode_label} thread:{CONTEXT.thread_key('HB')}",
        ]
        message_body = "\n".join(body_lines)

        formatted = CONTEXT.format_text(f"HB ✅\n{message_body}")
        blocks: List[Dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message_body},
            }
        ]

        if _send_with_blocks(self.slack, formatted, blocks=blocks):
            self.last_heartbeat_sent = now
            if trade_total_val is not None:
                self.last_heartbeat_trades = trade_total_val
        elif trade_total_val is not None:
            self.last_heartbeat_trades = trade_total_val

    def reset_emergency_state(self) -> None:
        self.emergencies.clear()
    def _reset_emergency(self, event_type: str) -> None:
        key = f"{event_type}:{CONTEXT.account_label}"
        state = self.emergencies.get(key)
        if state:
            state.fail_count = 0
            state.first_fail_ts = None

    def _handle_emergency(
        self,
        event_type: str,
        *,
        severity: float,
        now: float,
        message: str,
        fields: Optional[List[Dict[str, Any]]] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = f"{event_type}:{CONTEXT.account_label}"
        state = self.emergencies.setdefault(key, EmergencyState())
        if state.fail_count == 0:
            state.first_fail_ts = now
        state.fail_count += 1

        if state.fail_count < 3:
            return
        if state.first_fail_ts is None or now - state.first_fail_ts < EMERGENCY_MIN_WINDOW_SEC:
            return

        cooldown = float(self.config.get("alert_cooldown_sec", 600))
        within_cooldown = state.last_alert_ts is not None and (now - state.last_alert_ts) < cooldown
        severity_escalated = state.last_alert_severity is not None and severity > state.last_alert_severity
        if within_cooldown and not severity_escalated:
            return

        if snapshot is None and self.metrics is not None:
            try:
                snapshot = self.metrics.get_snapshot()
            except Exception:
                snapshot = {}
        elif snapshot is None:
            snapshot = {}

        signals_delta, trades_delta_window = self._signal_trade_delta()
        _log_alert_payload(
            snapshot or {},
            trades_delta=_safe_number(trades_delta_window),
            signals_delta=_safe_number(signals_delta),
            extra={"event": event_type, "severity": severity},
        )

        blocks = None
        if fields:
            blocks = [
                {"type": "section", "fields": fields},
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"run:{CONTEXT.run_id} acct:{CONTEXT.account_label} "
                                f"thread:{CONTEXT.thread_key('EMERG')} type:{event_type}"
                            ),
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"severity:{_format_decimal(severity, 2)}",
                        },
                    ],
                },
            ]

        formatted = CONTEXT.format_text(message)
        if _send_with_blocks(self.slack, formatted, blocks=blocks):
            state.last_alert_ts = now
            state.last_alert_severity = severity
            state.fail_count = 0
            state.first_fail_ts = None
            self.cooldowns[key] = now
    def _check_clock_drift(self, now: float, snap: Dict[str, Any]) -> None:
        drift_val = (snap.get("time_drift") or {}).get("exchange")
        try:
            drift_float = float(drift_val)
        except (TypeError, ValueError):
            self._reset_emergency("clock_drift")
            return
        if not math.isfinite(drift_float):
            self._reset_emergency("clock_drift")
            return
        threshold = float(self.config.get("clock_drift_warn_ms", 5000.0))
        if abs(drift_float) < threshold:
            self._reset_emergency("clock_drift")
            return
        direction = "+" if drift_float >= 0 else ""
        message = (
            f"⚠️ Clock drift: {direction}{int(drift_float)}ms (>{int(threshold)}). "
            f"Check NTP/exchange time. run:{CONTEXT.run_id} acct:{CONTEXT.account_label}"
        )
        fields = [
            {"type": "mrkdwn", "text": f"*Drift*\n{direction}{int(drift_float)} ms"},
            {"type": "mrkdwn", "text": f"*Threshold*\n{int(threshold)} ms"},
        ]
        self._handle_emergency("clock_drift", severity=abs(drift_float), now=now, message=message, fields=fields, snapshot=snap)

    def _check_error_burst(self, now: float) -> None:
        if not self.error_history:
            self._reset_emergency("error_burst")
            return
        window = float(self.config.get("error_burst_window_sec", 300.0))
        threshold = float(self.config.get("error_burst_threshold", 10.0))
        totals: Dict[str, float] = {}
        total_errors = 0.0
        for ts, delta_map in self.error_history:
            if now - ts > window:
                continue
            for reason, value in delta_map.items():
                totals[reason] = totals.get(reason, 0.0) + value
                total_errors += value
        if total_errors < threshold:
            self._reset_emergency("error_burst")
            return
        top_reason = max(totals.items(), key=lambda item: item[1])[0] if totals else "unknown"
        message = (
            f"⚠️ Error burst: {int(total_errors)} errors/{int(window/60)}m ({top_reason}). "
            f"run:{CONTEXT.run_id} acct:{CONTEXT.account_label}"
        )
        fields = [
            {"type": "mrkdwn", "text": f"*Total*\n{int(total_errors)}"},
            {"type": "mrkdwn", "text": f"*Window*\n{int(window/60)} m"},
            {"type": "mrkdwn", "text": f"*Top reason*\n{top_reason}"},
        ]
        self._handle_emergency("error_burst", severity=total_errors, now=now, message=message, fields=fields, snapshot=self.metrics.get_snapshot() if self.metrics else {})

    def _check_no_trade(self, now: float) -> None:
        if len(self.activity_history) < 2:
            self._reset_emergency("no_trade")
            return
        window = float(self.config.get("no_trade_window_sec", 7200))
        min_signals = float(self.config.get("no_trade_signals_min", 5))
        signals_delta, trades_delta = self._signal_trade_delta()
        if signals_delta is None or signals_delta < min_signals:
            self._reset_emergency("no_trade")
            return
        if trades_delta is not None and trades_delta > 0:
            self._reset_emergency("no_trade")
            return
        hours = max(1, int(window / 3600))
        message = (
            f"⚠️ Signals: {int(signals_delta)} in {hours}h but 0 trades. "
            f"Check filters/routing. run:{CONTEXT.run_id} acct:{CONTEXT.account_label}"
        )
        fields = [
            {"type": "mrkdwn", "text": f"*Signals*\n{int(signals_delta)}"},
            {"type": "mrkdwn", "text": f"*Trades*\n{_format_int(trades_delta, fallback='0')}"},
            {"type": "mrkdwn", "text": f"*Window*\n{hours} h"},
        ]
        self._handle_emergency("no_trade", severity=signals_delta or 0.0, now=now, message=message, fields=fields, snapshot=self.metrics.get_snapshot() if self.metrics else {})

    def _check_auto_testnet(self, now: float, snap: Dict[str, Any]) -> None:
        state = get_state()
        dd_ratio = compute_daily_dd_ratio(state=state)
        if config.get_trading_mode() != TradingMode.LIVE.value:
            self._reset_emergency("auto_testnet_on_dd")
            return

        if dd_ratio < config.AUTO_TESTNET_ON_DD_THRESHOLD:
            self._reset_emergency("auto_testnet_on_dd")
            return

        if state.is_deduped_today("AUTO_TESTNET_ON_DD"):
            return

        message = (
            f"🚨 AUTO_TESTNET_ON_DD triggered → daily dd {fmt_percent_ratio(dd_ratio)} (threshold {fmt_percent_ratio(config.AUTO_TESTNET_ON_DD_THRESHOLD)}). "
            f"run:{CONTEXT.run_id} acct:{CONTEXT.account_label}"
        )
        if self._guard_action:
            try:
                self._guard_action(dd_ratio)
            except Exception as exc:
                logger.warning("guard_action failed: %s", exc)

        notifier = _get_notifier()
        notifier.send_markdown(message)
        state.mark_deduped_today("AUTO_TESTNET_ON_DD")


def start_alert_scheduler(
    config: Dict[str, Any],
    metrics_manager=None,
    *,
    slack_sender: Callable[..., bool] = slack_notify_safely,
    interval_sec: float = 60.0,
    guard_action: Optional[Callable[[float], None]] = None,
) -> AlertScheduler:
    scheduler = AlertScheduler(
        metrics_manager or get_metrics_manager(),
        config,
        slack_sender,
        interval_sec=interval_sec,
        guard_action=guard_action,
    )
    scheduler.start()
    return scheduler

def _debug_dump() -> None:
    mgr = get_metrics_manager()
    snapshot = mgr.get_snapshot() if mgr else {}
    _log_alert_payload(snapshot or {}, trades_delta=None, signals_delta=None, extra={"event": "debug-dump"})
    try:
        print(json.dumps(snapshot, indent=2, default=_safe_json_value))
    except Exception:
        print(snapshot)


RUNBOOK_HEADER = (
    "[RUNBOOK] If this alert fires, review BAL_RAW/DD_CALC logs and metrics.\n"
    "To simulate locally:\n"
    "curl -s http://localhost:9000/health\n"
    "# or run a dry-run check:\n"
    "python -m auto_trading_bot.cli check --now"
)


class Alerts:
    """Alerting and instrumentation layer handling heartbeat, guardrails, and inputs."""

    def __init__(
        self,
        metrics,
        *,
        logger_name: str = __name__,
        heartbeat_interval_sec: int = 60,
        dd_guard_threshold: float = 0.20,
        dedupe_window_sec: int = 300,
        guard_action: Optional[Callable[[float], None]] = None,
        source: str = "binance-usdm-testnet",
        account_mode: str = "testnet",
    ) -> None:
        self._metrics = metrics
        self._logger = logging.getLogger(logger_name)
        self._hb_interval = max(1, int(heartbeat_interval_sec))
        self._dd_threshold = max(0.0, float(dd_guard_threshold))
        self._dedupe_window = max(1, int(dedupe_window_sec))
        self._guard_action = guard_action
        self._source = source
        self._account_mode = account_mode

        self._last_heartbeat_ts: Optional[float] = None
        self._last_alert_by_key: Dict[str, float] = {}

    def _now_utc(self, now_utc: Optional[datetime]) -> datetime:
        if now_utc is None:
            return datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            return now_utc.replace(tzinfo=timezone.utc)
        return now_utc.astimezone(timezone.utc)

    def _dd_ratio(self) -> float:
        try:
            ratio = float(self._metrics.bot_daily_drawdown())
        except Exception:
            ratio = 0.0
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio

    def _should_emit(self, key: str, *, now_ts: float) -> bool:
        last = self._last_alert_by_key.get(key)
        if last is None or (now_ts - last) >= self._dedupe_window:
            self._last_alert_by_key[key] = now_ts
            return True
        return False

    def maybe_emit_heartbeat(self, *, now_utc: Optional[datetime] = None) -> bool:
        now = self._now_utc(now_utc)
        now_ts = now.timestamp()
        last = self._last_heartbeat_ts
        if last is not None and (now_ts - last) < self._hb_interval:
            return False

        dd_ratio = self._dd_ratio()
        payload = {
            "type": "HEARTBEAT",
            "ts_utc": now.isoformat().replace("+00:00", "Z"),
            "dd_ratio": dd_ratio,
            "account_mode": self._account_mode,
            "source": self._source,
        }
        self._logger.info(json.dumps(payload, sort_keys=True))
        self._last_heartbeat_ts = now_ts
        return True

    def evaluate(
        self,
        *,
        equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> None:
        now = self._now_utc(now_utc)
        dd_ratio = self._dd_ratio()

        payload_input = {
            "type": "ALERTS_INPUT",
            "ts_utc": now.isoformat().replace("+00:00", "Z"),
            "dd_ratio": dd_ratio,
            "equity": float(equity) if equity is not None else None,
            "peak_equity": float(peak_equity) if peak_equity is not None else None,
            "account_mode": self._account_mode,
            "source": self._source,
        }
        self._logger.info(json.dumps(payload_input, sort_keys=True, default=_safe_json_value))

        live_mode = self._account_mode == TradingMode.LIVE.value

        if dd_ratio < self._dd_threshold:
            return

        key = f"AUTO_TESTNET_ON_DD:{self._account_mode}:{round(self._dd_threshold, 4)}"
        now_ts = now.timestamp()
        if not self._should_emit(key, now_ts=now_ts):
            return

        payload_guard = {
            "type": "GUARDRAIL_TRIP",
            "ts_utc": now.isoformat().replace("+00:00", "Z"),
            "guard": "AUTO_TESTNET_ON_DD",
            "dd_ratio": dd_ratio,
            "threshold": float(self._dd_threshold),
            "account_mode": self._account_mode,
            "source": self._source,
        }
        self._logger.info(json.dumps(payload_guard, sort_keys=True, default=_safe_json_value))
        body = (
            f":rotating_light: AUTO_TESTNET_ON_DD triggered\n"
            f"dd={dd_ratio * 100:.1f}% (threshold={self._dd_threshold * 100:.1f}%) | "
            f"mode={self._account_mode} | source={self._source}"
        )
        message = f"{RUNBOOK_HEADER}\n\n{body}"
        self._logger.info(message)

        notifier = _get_notifier()
        should_notify = live_mode or getattr(notifier, "_dry", False)
        if should_notify:
            notifier.send_markdown(message)
        else:
            self._logger.info(
                "Skipping AUTO_TESTNET_ON_DD Slack send in %s mode (dry-run disabled)",
                self._account_mode,
            )

        if self._guard_action is not None:
            try:
                self._guard_action(dd_ratio)
            except Exception as exc:
                self._logger.warning("guard_action failed: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alerts debug utilities")
    parser.add_argument("--debug-dump", action="store_true", help="Log and print the current alert inputs")
    args = parser.parse_args()
    if args.debug_dump:
        _debug_dump()
    else:
        parser.print_help()
