from __future__ import annotations

import os
import time
import signal
import atexit
import logging
import math
import json
from logging import Logger
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from config import *  # TF, RISK_PCT, LEVERAGE, UNIVERSE, SAFE_RESTART, ATR_LEN, POLL_SEC, DATA_BASE_DIR, TESTNET
from auto_trading_bot.exchange_api import ExchangeAPI, BalanceAuthError, BalanceSyncError
from strategy import DonchianATREngine
from indicators import atr, is_near_funding
from auto_trading_bot.reporter import Reporter
from auto_trading_bot.alerts import Alerts, slack_notify_safely, slack_notify_exit, start_alert_scheduler
from auto_trading_bot.metrics import Metrics, start_metrics_server, get_metrics_manager
from auto_trading_bot.restart import consume_restart_intent, peek_restart_intent
RESTART_INTENT_SKIP_TTL_DEFAULT = int(os.getenv("RESTART_INTENT_SKIP_TTL", "600"))

from slack_notify import notify_emergency

try:
    from scripts.daily_report import generate_report as _gen_daily_report
except Exception:
    _gen_daily_report = None

logger = logging.getLogger("donchian_bot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    try:
        _fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        _fh.setFormatter(_fmt)
        logger.addHandler(_fh)
    except Exception:
        pass
    logger.addHandler(_ch)
metrics_store = Metrics()
reporter = Reporter.from_config(metrics=metrics_store)
alerts = Alerts(
    metrics_store,
    heartbeat_interval_sec=OBSERVABILITY.get("heartbeat_interval_sec", 60) if "OBSERVABILITY" in globals() else 60,
    dd_guard_threshold=OBSERVABILITY.get("dd_guard_threshold", 0.20) if "OBSERVABILITY" in globals() else 0.20,
    dedupe_window_sec=OBSERVABILITY.get("guard_dedupe_sec", 300) if "OBSERVABILITY" in globals() else 300,
    guard_action=lambda ratio: emergency_cleanup("drawdown_guard"),
    source="binance-usdm-testnet" if TESTNET else "binance-usdm",
    account_mode="testnet" if TESTNET else "live",
)
log_trade = reporter.log_trade
log_exit = reporter.log_exit
log_signal_analysis = reporter.log_signal_analysis
log_filtered_signal = reporter.log_filtered_signal
log_detailed_entry = reporter.log_detailed_entry


b_global = None
eng_global = None
lock = Lock()


def _emit_daily_report(now: datetime, env_label: str, base_dir: str) -> None:
    if not DAILY_REPORT_ENABLED:
        return
    if not _gen_daily_report:
        logger.warning(
            "Daily report generation skipped; generator not available",
            extra={'event': 'daily_report_missing_generator'},
        )
        return

    date_str = now.strftime('%Y-%m-%d')
    try:
        _out_path, summary = _gen_daily_report(date_str, env_label, base_dir)
    except Exception as report_exc:
        logger.warning(
            "Daily report generation failed: %s",
            report_exc,
            extra={'event': 'daily_report_failed'},
        )
        return

    try:
        trades = int(summary.get('trades', 0))
        wins = int(summary.get('wins', 0))
        losses = int(summary.get('losses', 0))
        win_rate = summary.get('win_rate_pct')
        gross_win = summary.get('gross_win_pct')
        gross_loss = summary.get('gross_loss_pct')
        pf = summary.get('profit_factor')
        slack_notify_safely(
            f"Daily {env_label} summary {date_str}: trades={trades} wins={wins} losses={losses} "
            f"win_rate={win_rate}% PF={pf} gross+={gross_win}% gross-={gross_loss}%"
        )
    except Exception:
        pass


def align_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return float(price)
    return math.floor(price / tick_size) * tick_size


def align_qty(qty: float, step_size: float) -> float:
    if step_size <= 0:
        return float(qty)
    return math.floor(qty / step_size) * step_size


def compute_tp_ladder(entry_px: float, side: str, base_qty: float) -> List[Dict[str, float]]:
    side = (side or "").lower()
    if base_qty <= 0 or entry_px <= 0:
        return []
    pct_offsets = [0.006, 0.012, 0.018]
    qty_splits = [0.40, 0.35, 0.25]
    levels: List[Dict[str, float]] = []
    for idx, (pct, split) in enumerate(zip(pct_offsets, qty_splits), start=1):
        qty = base_qty * split
        if side == "short":
            target = entry_px * (1 - pct)
        else:
            target = entry_px * (1 + pct)
        levels.append({"px": target, "qty": qty, "level": idx})
    return levels


def _log_json(logger_obj: Logger, level: int, payload: Dict[str, Any], *, extra_fields: Optional[Dict[str, Any]] = None) -> None:
    data = dict(payload)
    if extra_fields:
        data.update(extra_fields)
    logger_obj.log(level, json.dumps(data, sort_keys=True))


def _replace_stop_only(b: ExchangeAPI, symbol: str, position_side: str, stop_price: float, qty: float) -> None:
    if qty <= 0 or stop_price <= 0:
        return
    exit_side = 'sell' if (position_side or '').lower() == 'long' else 'buy'
    try:
        cancel_fn = getattr(b, 'cancel_reduce_only_stop_orders', None)
        if callable(cancel_fn):
            cancel_fn(symbol)
    except Exception:
        pass
    b.create_stop_market_safe(symbol, exit_side, float(stop_price), float(qty), reduce_only=True)


def place_tp_ladder(
    b: ExchangeAPI,
    symbol: str,
    side: str,
    base_qty: float,
    entry_px: float,
    levels: List[Dict[str, float]],
    precision: Optional[Dict[str, float]] = None,
) -> None:
    if base_qty <= 0 or entry_px <= 0 or not levels:
        return

    precision = precision or {}
    tick_size = float(precision.get('tick_size') or precision.get('tickSize') or 0)
    step_size = float(precision.get('step_size') or precision.get('stepSize') or 0)
    min_notional = float(precision.get('min_notional') or precision.get('minNotional') or 0)
    exit_side = 'sell' if (side or '').lower() != 'short' else 'buy'

    placed_any = False
    for idx, level in enumerate(levels, start=1):
        raw_px = float(level.get('px') or 0)
        raw_qty = float(level.get('qty') or 0)
        px = align_price(raw_px, tick_size)
        qty = align_qty(raw_qty, step_size)
        payload_base = {
            'symbol': symbol,
            'side': exit_side.upper(),
            'level': idx,
            'price': float(px),
            'qty': float(qty),
        }
        if qty <= 0:
            payload = {'type': 'TP_SKIP', 'reason': 'zero_qty', **payload_base}
            _log_json(logger, logging.INFO, payload)
            continue
        if px <= 0:
            payload = {'type': 'TP_SKIP', 'reason': 'precision', **payload_base}
            _log_json(logger, logging.INFO, payload)
            continue
        if min_notional > 0 and qty * px < min_notional:
            payload = {'type': 'TP_SKIP', 'reason': 'minNotional', **payload_base}
            _log_json(logger, logging.INFO, payload)
            continue
        payload = {'type': 'TP_PLACE', **payload_base}
        try:
            b.create_limit_order(
                symbol,
                exit_side,
                px,
                qty,
                reduce_only=True,
                time_in_force='GTC',
            )
            metrics_mgr = get_metrics_manager()
            if metrics_mgr is not None:
                try:
                    metrics_mgr.inc_tp_orders(symbol, exit_side)
                except Exception:
                    pass
            _log_json(logger, logging.INFO, payload)
            placed_any = True
        except Exception as exc:
            payload_err = {'type': 'TP_SKIP', 'reason': 'order_error', 'error': str(exc), **payload_base}
            _log_json(logger, logging.WARNING, payload_err)
    if not placed_any:
        _log_json(
            logger,
            logging.WARNING,
            {'type': 'TP_WARN', 'symbol': symbol, 'side': exit_side.upper(), 'reason': 'all_skipped'},
        )


def _ensure_protective_stop_on_restart(b: ExchangeAPI, eng: DonchianATREngine, symbol: str) -> bool:
    """Ensure a reduceOnly protective stop exists for an open position.

    Order:
    - last_trail_stop (if present in state)
    - BE (entry) if be_promoted
    - entry ± fallback percent
    - last price ± fallback percent
    """
    try:
        from config import EMERGENCY_STOP_FALLBACK_PCT
    except Exception:
        EMERGENCY_STOP_FALLBACK_PCT = 0.015
    try:
        pos = b.position_for(symbol)
        if not pos:
            return False
        amt = abs(float(pos.get('contracts') or pos.get('positionAmt') or 0))
        if amt <= 0:
            return False
        side_label = (pos.get('side') or '').lower()
        st = eng.state.get(symbol, {}) if eng else {}
        stop_price = st.get('last_trail_stop') if st else None
        entry_px = float((st.get('entry_price') if st else 0) or pos.get('entryPrice') or 0)
        be = bool(st.get('be_promoted', False)) if st else False
        if not stop_price and entry_px > 0:
            stop_price = entry_px if be else (
                entry_px * (1 - EMERGENCY_STOP_FALLBACK_PCT) if side_label == 'long'
                else entry_px * (1 + EMERGENCY_STOP_FALLBACK_PCT)
            )
        if not stop_price:
            try:
                tk = b.fetch_ticker(symbol)
                last = float(tk.get('last') or tk.get('close') or tk.get('bid') or tk.get('ask') or 0)
                if last > 0:
                    stop_price = last * (1 - EMERGENCY_STOP_FALLBACK_PCT) if side_label == 'long' else last * (1 + EMERGENCY_STOP_FALLBACK_PCT)
            except Exception:
                pass
        if not stop_price:
            return False
        _replace_stop_only(b, symbol, side_label or 'long', float(stop_price), amt)
        # Persist the last known protective stop for visibility after restart
        try:
            st = eng.state.get(symbol, {}) if eng else {}
            st['last_trail_stop'] = float(stop_price)
            eng.state[symbol] = st
            try:
                from strategy import save_state as _save
                _save(eng.state)
            except Exception:
                pass
        except Exception:
            pass
        _log_json(
            logger,
            logging.INFO,
            {
                'event': 'restart_stop_applied',
                'symbol': symbol,
                'stop_price': float(stop_price),
                'qty': float(amt),
            },
        )
        return True
    except BalanceAuthError as auth_err:
        _log_json(
            logger,
            logging.ERROR,
            {
                'event': 'restart_stop_failed',
                'symbol': symbol,
                'error': str(auth_err),
                'error_type': 'auth',
            },
        )
        raise
    except Exception as e:
        _log_json(
            logger,
            logging.WARNING,
            {
                'event': 'restart_stop_failed',
                'symbol': symbol,
                'error': str(e),
                'error_type': 'generic',
            },
        )
        return False


def startup_sync(b: ExchangeAPI, eng: DonchianATREngine) -> None:
    """Synchronize state and protective stops on boot.

    - Re-arm protective stops for any open positions (idempotent).
    - If state lacks entry_price but broker reports a position, persist entry_price, side, and in_position.
    - Persist last_trail_stop for visibility.
    """
    # Protective stops
    for symbol in UNIVERSE:
        try:
            _ensure_protective_stop_on_restart(b, eng, symbol)
        except Exception:
            # Best-effort; auth/sync failures are handled by caller
            pass
    # Entry price + side/in_position sync
    for symbol in UNIVERSE:
        try:
            pos = b.position_for(symbol)
            if not pos:
                continue
            amt = abs(float(pos.get('contracts') or pos.get('positionAmt') or 0))
            if amt <= 0:
                continue
            entry_px = float(pos.get('entryPrice') or 0)
            if entry_px > 0:
                st = eng.state.get(symbol, {}) if eng else {}
                if float(st.get('entry_price') or 0) <= 0:
                    st['entry_price'] = entry_px
                st['in_position'] = True
                if not st.get('side'):
                    side_label = (pos.get('side') or ('long' if (pos.get('contracts') or 0) > 0 else 'short')).lower()
                    st['side'] = side_label
                eng.state[symbol] = st
        except Exception:
            pass
    # Persist once after batch
    try:
        from strategy import save_state as _save
        _save(eng.state)
    except Exception:
        pass

def _rearm_protective_stops(exchange: Optional[ExchangeAPI], eng: Optional[DonchianATREngine]) -> None:
    if not exchange or not eng:
        return
    fatal_auth = False
    try:
        for symbol in UNIVERSE:
            if fatal_auth:
                break
            try:
                _ensure_protective_stop_on_restart(exchange, eng, symbol)
            except BalanceAuthError as auth_err:
                fatal_auth = True
                _log_json(
                    logger,
                    logging.ERROR,
                    {
                        'event': 'emergency_rearm_failed',
                        'symbol': symbol,
                        'error': str(auth_err),
                        'error_type': 'auth',
                    },
                )
            except Exception as exc:
                _log_json(
                    logger,
                    logging.WARNING,
                    {
                        'event': 'emergency_rearm_failed',
                        'symbol': symbol,
                        'error': str(exc),
                        'error_type': 'generic',
                    },
                )
        if not fatal_auth:
            _log_json(
                logger,
                logging.INFO,
                {'event': 'emergency_rearm_complete', 'symbols': list(UNIVERSE)},
            )
    except Exception as exc:
        _log_json(
            logger,
            logging.ERROR,
            {'event': 'emergency_rearm_error', 'error': str(exc)},
        )


class EmergencyManager:
    def __init__(
        self,
        exchange: ExchangeAPI,
        engine: DonchianATREngine,
        universe: list[str],
        notify_callback: Callable[[str], bool],
        *,
        auto_testnet_on_dd: bool,
        daily_dd_limit: float,
        emergency_policy: str,
        kill_switch: Dict[str, Any],
        order_retry_attempts: int = 3,
        order_retry_backoff_sec: float = 1.0,
        state_store: Optional[Any] = None,
        notify_func: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.exchange = exchange
        self.engine = engine
        self.universe = list(universe)
        self.notify = notify_func or notify_callback
        self.auto_testnet_on_dd = bool(auto_testnet_on_dd)
        self.daily_dd_limit = float(daily_dd_limit or 0.0)
        self.emergency_policy = (emergency_policy or "protect_only").lower()
        self.kill_switch_cfg = kill_switch or {}
        self.order_retry_attempts = max(0, int(order_retry_attempts))
        self.order_retry_backoff_sec = max(0.0, float(order_retry_backoff_sec))
        self.error_counts = {"auth": 0, "nonce": 0, "time_drift": 0}
        self.block_entries_until: Optional[datetime] = None
        self.auto_testnet_armed = False
        self.kill_switch_triggered = False
        self.metrics: Optional[Any] = None
        self.reporter: Optional[Reporter] = None
        self.alerts: Optional[Alerts] = None
        self._state_store = state_store
        self._guard_state_key: Optional[str] = None
        if self._state_store is not None:
            account_mode = getattr(exchange, "account_mode", None)
            if callable(account_mode):
                try:
                    account_mode = account_mode()
                except Exception:
                    account_mode = None
            self._guard_state_key = f"{(account_mode or 'live')}:emergency.guard_date_utc"

    def _next_utc_midnight(self, now: datetime) -> datetime:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        tomorrow = (now + timedelta(days=1)).date()
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)

    def positions_flat(self) -> bool:
        state = getattr(self.engine, "state", {})
        if isinstance(state, dict):
            for value in state.values():
                if isinstance(value, dict) and value.get("in_position"):
                    return False
        return True

    def should_block_entries(self, now: datetime) -> bool:
        if self.block_entries_until and now >= self.block_entries_until:
            self.block_entries_until = None
            self.auto_testnet_armed = False
        return self.block_entries_until is not None

    def handle_daily_drawdown(self, dd_ratio: float, now: datetime) -> bool:
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        now_utc = now_utc.astimezone(timezone.utc)
        self.should_block_entries(now_utc)

        current_date_iso = now_utc.date().isoformat()
        if self._state_store is not None and self._guard_state_key is not None:
            if self._state_store.get(self._guard_state_key) == current_date_iso:
                return False

        if not self.auto_testnet_on_dd or self.auto_testnet_armed:
            return False
        if dd_ratio >= self.daily_dd_limit and self.positions_flat():
            self.auto_testnet_armed = True
            self.block_entries_until = self._next_utc_midnight(now_utc)
            self.exchange.set_testnet(True)
            message = "[EMERGENCY] AUTO_TESTNET_ON_DD triggered → switched to testnet, live entries disabled until UTC reset."
            self.notify(message)
            if self.alerts is not None:
                self.alerts.evaluate(equity=None, peak_equity=None, now_utc=now_utc)
            if self._state_store is not None and self._guard_state_key is not None:
                self._state_store.set(self._guard_state_key, current_date_iso)
            return True
        return False

    def record_error(self, kind: str) -> None:
        if self.kill_switch_triggered:
            return
        key_map = {"auth": "auth", "nonce": "nonce", "time_drift": "time_drift"}
        mapped = key_map.get(kind)
        if mapped:
            self.error_counts[mapped] += 1

    def should_trigger_kill_switch(self) -> bool:
        if self.kill_switch_triggered:
            return True
        cfg = self.kill_switch_cfg or {}
        auth_limit = int(cfg.get("auth_failures", 5))
        nonce_limit = int(cfg.get("nonce_errors", cfg.get("max_retries", 3)))
        drift_limit = int(cfg.get("max_retries", 3))
        if self.error_counts["auth"] >= auth_limit:
            self.kill_switch_triggered = True
        if self.error_counts["nonce"] >= nonce_limit:
            self.kill_switch_triggered = True
        if self.error_counts["time_drift"] >= drift_limit:
            self.kill_switch_triggered = True
        return self.kill_switch_triggered

    def execute_cleanup(self, reason: str, now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(timezone.utc)
        reason_text = reason or "unspecified"
        attempt_message = f"[EMERGENCY] Policy '{self.emergency_policy}' triggered by {reason_text}; initiating cleanup."
        self.notify(attempt_message)
        _log_json(
            logger,
            logging.WARNING,
            {
                'event': 'emergency_cleanup_start',
                'policy': self.emergency_policy,
                'reason': reason_text,
            },
        )

        deadline = self._next_utc_midnight(now)

        if self.emergency_policy == "protect_only":
            self.block_entries_until = deadline
            _rearm_protective_stops(self.exchange, self.engine)
            completion_message = (
                f"[EMERGENCY] Policy 'protect_only' applied; entries blocked until "
                f"{deadline.isoformat().replace('+00:00', 'Z')}."
            )
            self.notify(completion_message)
            _log_json(
                logger,
                logging.INFO,
                {'event': 'protect_only_applied', 'block_until': deadline.isoformat()},
            )
            return

        require_profit = self.emergency_policy == "flatten_if_safe"
        summary = self._flatten_positions(require_profit=require_profit)
        closed = summary.get("closed", [])
        failed = summary.get("failed", [])
        skipped = summary.get("skipped", [])

        self.block_entries_until = deadline
        _rearm_protective_stops(self.exchange, self.engine)

        if self.emergency_policy == "flatten_all":
            detail = "unknown"
            if failed:
                detail_parts: list[str] = []
                for item in failed:
                    symbol = item.get("symbol", "?")
                    remaining = item.get("remaining_qty")
                    try:
                        remaining_val = float(remaining)
                        if math.isnan(remaining_val):
                            remaining_text = "nan"
                        else:
                            remaining_text = f"{remaining_val:.4f}"
                    except Exception:
                        remaining_text = str(remaining)
                    reason_note = item.get("reason") or "unknown"
                    detail_parts.append(f"{symbol} rem={remaining_text} ({reason_note})")
                detail = "; ".join(detail_parts) if detail_parts else "unknown"
                self.notify(f"[EMERGENCY] Flatten_all incomplete: {detail}. Manual intervention required.")
            logger.error("Flatten_all failed to flatten all positions: %s", detail)
            metrics_mgr = get_metrics_manager()
            if metrics_mgr is not None:
                try:
                    metrics_mgr.inc_flatten_failure("order_error")
                except Exception:
                    pass
            else:
                count = len(closed)
                self.notify(
                    f"[EMERGENCY] Policy 'flatten_all' executed successfully; {count} positions confirmed flat."
                )
                _log_json(
                    logger,
                    logging.INFO,
                    {'event': 'flatten_all_success', 'closed_count': count},
                )
        else:
            closed_count = len(closed)
            if closed_count:
                message = (
                    f"[EMERGENCY] Policy 'flatten_if_safe' closed {closed_count} profitable positions."
                )
            else:
                message = "[EMERGENCY] Policy 'flatten_if_safe' found no profitable positions to close."
            self.notify(message)
            if failed:
                logger.error("Flatten_if_safe encountered failures: %s", failed)

        if skipped:
            _log_json(
                logger,
                logging.INFO,
                {'event': 'flatten_skipped', 'count': len(skipped), 'symbols': skipped},
            )
            metrics_mgr = get_metrics_manager()
            if metrics_mgr is not None:
                try:
                    metrics_mgr.inc_flatten_partial("not_profitable")
                except Exception:
                    pass

    def _flatten_positions(self, require_profit: bool) -> Dict[str, Any]:
        outcomes: Dict[str, list[Dict[str, Any]]] = {"closed": [], "failed": [], "skipped": []}
        targets: list[str] = []
        meta: Dict[str, Dict[str, Any]] = {}
        for symbol in self.universe:
            try:
                pos = self.exchange.position_for(symbol)
            except Exception as exc:
                outcomes["skipped"].append({"symbol": symbol, "reason": f"position_fetch:{exc}"})
                continue
            if not pos:
                continue
            amt = self._position_amount(pos)
            if abs(amt) <= 1e-8:
                continue
            side = "long" if amt > 0 else "short"
            if require_profit and not self._is_profitable(symbol, side, pos):
                outcomes["skipped"].append({"symbol": symbol, "reason": "not_profitable"})
                continue
            targets.append(symbol)
            meta[symbol] = {"side": side, "initial_qty": abs(amt)}

        if not targets:
            return outcomes

        flatten_fn = getattr(self.exchange, "flatten_all", None)
        if not callable(flatten_fn):
            logger.error("ExchangeAPI has no flatten_all implementation; cannot execute emergency flatten.")
            for symbol in targets:
                meta_info = meta.get(symbol, {})
                outcomes["failed"].append(
                    {
                        "symbol": symbol,
                        "reason": "flatten_all_missing",
                        "remaining_qty": meta_info.get("initial_qty"),
                        "side": meta_info.get("side"),
                        "initial_qty": meta_info.get("initial_qty"),
                    }
                )
            metrics_mgr = get_metrics_manager()
            if metrics_mgr is not None:
                try:
                    metrics_mgr.inc_flatten_failure("missing_handler")
                except Exception:
                    pass
            return outcomes

        results = flatten_fn(
            targets,
            retries=self.order_retry_attempts,
            backoff_sec=self.order_retry_backoff_sec,
        ) or []

        metrics_mgr = get_metrics_manager()
        for result in results:
            if not isinstance(result, dict):
                continue
            symbol = result.get("symbol")
            if not symbol:
                continue
            meta_info = meta.get(symbol, {})
            result.setdefault("side", meta_info.get("side"))
            result.setdefault("initial_qty", meta_info.get("initial_qty"))
            if result.get("status") == "closed":
                outcomes["closed"].append(result)
                self._clear_engine_state(symbol)
            else:
                result.setdefault("remaining_qty", meta_info.get("initial_qty"))
                outcomes["failed"].append(result)
                if metrics_mgr is not None:
                    try:
                        metrics_mgr.inc_flatten_failure(result.get("reason") or "unknown")
                    except Exception:
                        pass

        reported = {entry.get("symbol") for entry in results if isinstance(entry, dict)}
        for symbol in targets:
            if symbol not in reported:
                meta_info = meta.get(symbol, {})
                outcomes["failed"].append(
                    {
                        "symbol": symbol,
                        "reason": "no_response",
                        "remaining_qty": meta_info.get("initial_qty"),
                        "side": meta_info.get("side"),
                        "initial_qty": meta_info.get("initial_qty"),
                    }
                )
                if metrics_mgr is not None:
                    try:
                        metrics_mgr.inc_flatten_partial("no_response")
                    except Exception:
                        pass

        return outcomes

    def _clear_engine_state(self, symbol: str) -> None:
        state_obj = getattr(self.engine, "state", None)
        if isinstance(state_obj, dict):
            entry = state_obj.get(symbol)
            if isinstance(entry, dict):
                entry["in_position"] = False
            else:
                state_obj[symbol] = {"in_position": False}
        if hasattr(self.engine, "clear_position_state"):
            try:
                self.engine.clear_position_state(symbol)
            except Exception:
                pass

    @staticmethod
    def _position_amount(position: Dict[str, Any]) -> float:
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

    def _is_profitable(self, symbol: str, side: str, pos: Dict[str, Any]) -> bool:
        try:
            profit = float(pos.get('unrealizedProfit'))
            if not math.isnan(profit):
                return profit > 0
        except Exception:
            pass
        try:
            entry_price = float(pos.get('entryPrice') or 0.0)
        except Exception:
            entry_price = 0.0
        if entry_price <= 0:
            return False
        mark = 0.0
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            mark = float(ticker.get('last') or ticker.get('close') or ticker.get('markPrice') or 0.0)
        except Exception:
            pass
        if mark <= 0:
            return False
        if side == 'long':
            return mark > entry_price
        return mark < entry_price


emergency_manager: Optional[EmergencyManager] = None


def emergency_cleanup(reason: str = "manual") -> None:
    manager = emergency_manager
    if manager:
        try:
            metrics_mgr = get_metrics_manager()
            if metrics_mgr is not None:
                snapshot = metrics_mgr.get_snapshot()
                eq = snapshot.get("equity")
                if eq is not None:
                    metrics_mgr.set_equity(eq)
                dd_val = snapshot.get("daily_dd")
                if dd_val is not None:
                    metrics_mgr.set_daily_drawdown(dd_val)
                avg_r_val = snapshot.get("avg_r")
                if avg_r_val is not None:
                    metrics_mgr.set_avg_r(avg_r_val)
                trades_delta_val = snapshot.get("trades_delta")
                if trades_delta_val is not None:
                    metrics_mgr.set_trades_delta(trades_delta_val)
                window_trades_val = snapshot.get("window_trades")
                if window_trades_val is not None:
                    metrics_mgr.set_window_trades(window_trades_val)
            manager.execute_cleanup(reason, datetime.now(timezone.utc))
        except Exception as exc:
            _log_json(
                logger,
                logging.WARNING,
                {'event': 'emergency_cleanup_failed', 'reason': reason, 'error': str(exc)},
            )
    else:
        if not _graceful_shutdown_requested:
            _rearm_protective_stops(b_global, eng_global)


def _should_skip_signal_emergency(signum: int) -> bool:
    if signum != signal.SIGTERM:
        return False
    intent = peek_restart_intent(ttl_seconds=RESTART_INTENT_SKIP_TTL_DEFAULT)
    if intent is None:
        return False
    logger.info(
        "Suppressing emergency cleanup on SIGTERM due to restart intent",
        extra={
            "event": "signal_skip_emergency",
            "mode": intent.mode,
            "ttl_sec": RESTART_INTENT_SKIP_TTL_DEFAULT,
        },
    )
    return True


def signal_handler(signum, frame):
    sender = f"signal:{signum}"
    _mark_shutdown_graceful(sender)
    try:
            if _should_skip_signal_emergency(signum):
                emergency_mgr = emergency_manager
                if emergency_mgr is not None:
                    emergency_mgr.emergency_policy = "protect_only"
                return
            emergency_cleanup("signal")
    finally:
        os._exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
_graceful_shutdown_requested = False


def _mark_shutdown_graceful(sender: str) -> None:
    global _graceful_shutdown_requested
    _graceful_shutdown_requested = True
    _log_json(
        logger,
        logging.INFO,
        {"event": "graceful_shutdown_requested", "sender": sender},
    )


def _atexit_cleanup() -> None:
    if _graceful_shutdown_requested:
        _log_json(logger, logging.INFO, {"event": "graceful_shutdown_skip", "reason": "graceful"})
        return
    emergency_cleanup("atexit")


atexit.register(_atexit_cleanup)


def fetch_df(b: ExchangeAPI, symbol: str, tf: str, lookback: int) -> pd.DataFrame:
    ohlcv = b.fetch_ohlcv(symbol, tf, limit=lookback)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main():
    startup_mode = 'SAFE' if SAFE_CONSOLE_BANNER else 'LIVE'
    logger.info(
        "Bot startup",
        extra={
            'event': 'startup_banner',
            'mode': startup_mode,
            'tf': TF,
            'risk_pct': RISK_PCT,
            'leverage': LEVERAGE,
            'universe': list(UNIVERSE),
        },
    )

    intent = consume_restart_intent()
    if intent is not None:
        _mark_shutdown_graceful("startup-consume")
        _log_json(
            logger,
            logging.INFO,
            {
                "event": "restart_intent_consumed",
                "mode": intent.mode,
                "created_at": intent.created_at.isoformat(),
            },
        )

    metrics_port = int(OBSERVABILITY.get("metrics_port", 9108))
    account_label = str(OBSERVABILITY.get("account_label", "live"))
    metrics_mgr = start_metrics_server(metrics_port, account_label)
    metrics_mgr.set_heartbeat()
    start_alert_scheduler(OBSERVABILITY, metrics_mgr)

    b = ExchangeAPI()
    b.connect()
    eng = DonchianATREngine()
    global b_global, eng_global
    b_global, eng_global = b, eng
    emergency_mgr = EmergencyManager(
        exchange=b,
        engine=eng,
        universe=UNIVERSE,
        notify_callback=notify_emergency,
        auto_testnet_on_dd=AUTO_TESTNET_ON_DD,
        daily_dd_limit=DAILY_DD_LIMIT,
        emergency_policy=EMERGENCY_POLICY,
        kill_switch=KILL_SWITCH,
    )
    if intent is not None and intent.mode == "graceful":
        emergency_mgr.emergency_policy = "protect_only"
        _log_json(
            logger,
            logging.INFO,
            {"event": "emergency_policy_override", "policy": "protect_only"},
        )
    global emergency_manager
    emergency_manager = emergency_mgr

    def _check_kill_switch() -> None:
        if emergency_mgr.should_trigger_kill_switch():
            notify_emergency("[CRITICAL] Kill-switch activated due to repeated failures -> bot terminated.")
            emergency_cleanup("kill-switch")
            raise SystemExit(1)

    # Log active config snapshot for traceability (secrets redacted)
    try:
        import json as _json
        import config as cfg
        cfg_snapshot = {
            "env": "testnet" if cfg.TESTNET else "live",
            "quote": cfg.QUOTE,
            "universe": list(cfg.UNIVERSE),
            "risk_pct": cfg.RISK_PCT,
            "leverage": cfg.LEVERAGE,
            "atr": {"len": cfg.ATR_LEN, "stop_k": cfg.ATR_STOP_K, "trail_k": cfg.ATR_TRAIL_K},
            "daily_loss_limit": cfg.DAILY_LOSS_LIMIT,
            "timeframe": cfg.TF,
            "lookback": cfg.LOOKBACK,
            "pyramiding": cfg.ENABLE_PYRAMIDING,
            "position_cap": {"min": cfg.POSITION_CAP_MIN, "max": cfg.POSITION_CAP_MAX, "multiple": cfg.POSITION_CAP_MULTIPLE},
            "data_dir": cfg.DATA_BASE_DIR,
        }
        logger.info("Active config snapshot: %s", _json.dumps(cfg_snapshot, ensure_ascii=False))
    except Exception:
        pass

    try:
        if alerts is not None:
            alerts.reset_emergency_state()
    except Exception:
        pass

    last_equity = 0.0
    try:
        snap = b.fetch_equity_snapshot()
        last_equity = snap.margin_balance
        metrics_mgr.set_equity(last_equity)
        reporter.apply_equity_snapshot(snap, now_utc=snap.ts_utc)
        alerts.evaluate(equity=snap.margin_balance, peak_equity=None, now_utc=snap.ts_utc)
        alerts.maybe_emit_heartbeat(now_utc=snap.ts_utc)
        logger.info("Startup equity snapshot margin=%.2f", last_equity)
    except BalanceAuthError as auth_err:
        logger.error('Startup balance auth error: %s', auth_err)
        slack_notify_safely(f":rotating_light: Startup auth failure: {auth_err}")
        raise
    except BalanceSyncError as sync_err:
        logger.error('Startup balance sync error: %s', sync_err)
        slack_notify_safely(f":rotating_light: Startup time sync failure: {sync_err}")
        raise
    except Exception as bal_err:
        logger.warning('Startup balance fetch failed; continuing with 0 equity: %s', bal_err)
        last_equity = 0.0

    # On startup, re-arm protective stops and sync state for any existing positions
    try:
        if intent is not None and intent.mode == "graceful":
            logger.info("Graceful restart: skipping protective rearm on startup", extra={'event': 'startup_skip_rearm'})
        else:
            startup_sync(b, eng)
        logger.info("Trading loop starting", extra={'event': 'loop_start'})
    except BalanceAuthError as auth_err:
        logger.error('Startup protective stop auth error: %s', auth_err)
        logger.error('Startup protective stop auth error: %s', auth_err, extra={'event': 'startup_protective_stop_error', 'error_type': 'auth'})
        try:
            slack_notify_safely(f":rotating_light: Protective stop restore failed: {auth_err}")
        except Exception:
            pass
        emergency_cleanup()
        return
    except Exception as e:
        _log_json(
            logger,
            logging.WARNING,
            {'event': 'startup_protect_failed', 'error': str(e)},
        )

    last_bar_ts = {sym: None for sym in UNIVERSE}
    last_heartbeat = 0.0

    while True:
        loop_start = time.perf_counter()
        lock.acquire()
        try:
            now_utc = datetime.now(timezone.utc)
            entries_blocked = emergency_mgr.should_block_entries(now_utc)
            eq = last_equity
            try:
                snap = b.fetch_equity_snapshot()
                reporter.apply_equity_snapshot(snap, now_utc=snap.ts_utc)
                eq = snap.margin_balance
                last_equity = eq
                eng.reset_daily_anchor(eq)
                alerts.evaluate(equity=eq, peak_equity=None, now_utc=snap.ts_utc)
                alerts.maybe_emit_heartbeat(now_utc=snap.ts_utc)
            except BalanceAuthError as auth_err:
                logger.error('Balance auth error: %s', auth_err)
                _log_json(
                    logger,
                    logging.ERROR,
                    {
                        'event': 'balance_fetch_failed',
                        'type': 'auth',
                        'error': str(auth_err),
                    },
                )
                emergency_mgr.record_error('auth')
                try:
                    slack_notify_safely(f":rotating_light: Runtime auth failure: {auth_err}")
                except Exception:
                    pass
                _check_kill_switch()
                emergency_cleanup("runtime-auth")
                raise SystemExit(1)
            except BalanceSyncError as sync_err:
                logger.error('Balance sync failure: %s', sync_err)
                _log_json(
                    logger,
                    logging.ERROR,
                    {
                        'event': 'balance_fetch_failed',
                        'type': 'time_drift',
                        'error': str(sync_err),
                    },
                )
                emergency_mgr.record_error('time_drift')
                try:
                    slack_notify_safely(f":rotating_light: Runtime time sync failure: {sync_err}")
                except Exception:
                    pass
                _check_kill_switch()
                emergency_cleanup("runtime-sync")
                raise SystemExit(2)
            except Exception as bal_err:
                logger.warning('Balance fetch error, using cached equity: %s', bal_err)
                _log_json(
                    logger,
                    logging.WARNING,
                    {
                        'event': 'balance_fetch_warning',
                        'equity_cached': last_equity,
                        'error': str(bal_err),
                    },
                )
            metrics_mgr.set_equity(eq)
            raw_client = getattr(b, 'raw', None)
            if raw_client is not None:
                drift = getattr(raw_client, 'time_diff', None)
                if drift is not None:
                    try:
                        metrics_mgr.set_time_drift('exchange', float(drift))
                    except Exception:
                        pass
            # Heartbeat every ~30s
            now = time.time()
            if now - last_heartbeat > 30:
                _log_json(
                    logger,
                    logging.INFO,
                    {
                        'event': 'heartbeat',
                        'equity': eq,
                        'ts_utc': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z'),
                    },
                )
                # Position snapshots
                try:
                    for _sym in UNIVERSE:
                        try:
                            _pos = b.position_for(_sym)
                            _amt = abs(float(_pos.get('contracts') or _pos.get('positionAmt') or 0)) if _pos else 0.0
                            if _pos and _amt > 0:
                                _side = (_pos.get('side') or ('long' if (_pos.get('contracts') or 0) > 0 else 'short')).lower()
                                _st = eng.state.get(_sym, {}) if eng else {}
                                # Prefer broker entryPrice for visibility; fall back to state
                                _entry_broker = 0.0
                                try:
                                    _entry_broker = float(_pos.get('entryPrice') or 0)
                                except Exception:
                                    _entry_broker = 0.0
                                _entry_state = float(_st.get('entry_price') or 0)
                                _entry = _entry_state if _entry_state > 0 else _entry_broker
                                # Last known protective/trailing stop from state (persisted when we place it)
                                _trail = _st.get('last_trail_stop')
                                _trail_val = float(_trail) if _trail is not None else 0.0
                                _log_json(
                                    logger,
                                    logging.INFO,
                                    {
                                        'event': 'position_snapshot',
                                        'symbol': _sym,
                                        'side': _side,
                                        'qty': round(_amt, 6),
                                        'entry': round(_entry, 4),
                                        'stop': round(_trail_val, 4),
                                    },
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
                last_heartbeat = now

            for symbol in UNIVERSE:
                try:
                    precision_info: Dict[str, Any] = {}
                    precision_fn = getattr(b, 'market_precision', None)
                    if callable(precision_fn):
                        try:
                            precision_info = precision_fn(symbol) or {}
                        except Exception:
                            precision_info = {}
                    if not precision_info:
                        raw_client = getattr(b, 'raw', None)
                        try:
                            if raw_client is not None and hasattr(raw_client, 'market'):
                                market = raw_client.market(symbol)
                                precision_info = market.get('precision', {}) or {}
                                precision_info.update({
                                    'min_notional': market.get('limits', {}).get('cost', {}).get('min'),
                                    'tick_size': market.get('limits', {}).get('price', {}).get('min'),
                                    'step_size': market.get('limits', {}).get('amount', {}).get('min'),
                                })
                        except Exception:
                            precision_info = {}

                    df = fetch_df(b, symbol, TF, LOOKBACK)
                    if len(df) < max(ATR_LEN, 25):
                        continue
                    atr_abs = float(atr(df, ATR_LEN) or 0)
                    close = float(df['close'].iloc[-1])
                    bar_ts = df['ts'].iloc[-1]
                    is_new_bar = (last_bar_ts[symbol] != bar_ts)
                    last_bar_ts[symbol] = bar_ts

                    from datetime import datetime as _dt
                    daily_loss_hit = eng.hit_daily_loss_limit(eq)
                    try:
                        eng.update_reset_tracking(symbol, df)
                    except Exception:
                        pass
                    can_reenter = eng.can_re_enter_after_stop(symbol)
                    funding_avoid = is_near_funding(_dt.utcnow(), minutes=FUNDING_AVOID_MIN)

                    plan = eng.make_entry_plan(
                        symbol=symbol,
                        df=df,
                        equity=eq,
                        price=close,
                        atr_abs=atr_abs,
                        is_new_bar=is_new_bar,
                        can_reenter=can_reenter,
                        funding_avoid=funding_avoid,
                        daily_loss_hit=daily_loss_hit,
                    )
                    # signal log
                    try:
                        sig = plan.get('signal', {})
                        anchor = float(((eng.state or {}).get('daily') or {}).get('anchor', eq) if hasattr(eng,'state') else eq)
                        dd_ratio = max(0.0, (anchor - eq) / max(anchor, 1e-9)) if anchor else 0.0
                        if emergency_mgr.handle_daily_drawdown(dd_ratio, now_utc):
                            entries_blocked = emergency_mgr.should_block_entries(now_utc)
                        metrics_mgr.set_daily_drawdown(dd_ratio)
                        dd_pct = dd_ratio * 100.0
                        entries_blocked = emergency_mgr.should_block_entries(now_utc)
                        log_signal_analysis(symbol, close, sig, is_new_bar, funding_avoid, daily_loss_hit, eq, dd_pct, plan.get('decision'), plan.get('skip_reason'))
                    except Exception as _e:
                        _log_json(
                            logger,
                            logging.WARNING,
                            {
                                'event': 'signal_log_error',
                                'symbol': symbol,
                                'error': str(_e),
                            },
                        )
                    if is_new_bar:
                        dec = plan.get('decision')
                        sk = plan.get('skip_reason')
                        fast_ma = (sig or {}).get('fast_ma', 0.0)
                        slow_ma = (sig or {}).get('slow_ma', 0.0)
                        ma_diff_pct = ((float(fast_ma)/float(slow_ma)-1)*100.0) if slow_ma else 0.0
                        cflt = (sig or {}).get('candle_filter', {}) or {}
                        pos_ratio = cflt.get('candle_position_ratio', 0.0)
                        rmult = (sig or {}).get('risk_multiplier', {}) or {}
                        rmL = rmult.get('long', 1.0)
                        rmS = rmult.get('short', 1.0)
                        _log_json(
                            logger,
                            logging.INFO,
                            {
                                'event': 'bar',
                                'symbol': symbol,
                                'ts': bar_ts.isoformat(),
                                'close': round(close, 4),
                                'atr': round(atr_abs, 4),
                                'fast_ma': round(fast_ma, 4),
                                'slow_ma': round(slow_ma, 4),
                                'ma_diff_pct': round(ma_diff_pct, 4),
                                'pos_ratio': round(pos_ratio, 4),
                                'rm_long': rmL,
                                'rm_short': rmS,
                                'decision': dec,
                                'skip_reason': sk,
                            },
                        )

                    # Manage trailing stop if in position
                    try:
                        pos = b.position_for(symbol)
                        has_pos = bool(pos) and abs(float(pos.get('contracts') or pos.get('positionAmt') or 0)) > 0
                    except Exception:
                        has_pos = False
                    if has_pos:
                        st = eng.update_after_move(symbol, atr_abs, close)
                        side = (st or {}).get('side')
                        entry_px = float((st or {}).get('entry_price') or 0)
                        be = bool((st or {}).get('be_promoted', False))
                        new_trail = eng.trail_stop_price(side, entry_px, close, atr_abs, be)
                        try:
                            amt = abs(float(pos.get('contracts') or pos.get('positionAmt') or 0))
                            if amt > 0:
                                _replace_stop_only(b, symbol, side, new_trail, amt)
                        except Exception:
                            pass

                        # Pyramiding: add when trigger appears and not yet executed at this level
                        try:
                            pyr_level = int((st or {}).get('pyramid_level', 0))
                            added_list = list((st or {}).get('pyramid_added', []))
                            if pyr_level > len(added_list):
                                lock_info = (st or {}).get('pyramid_locked_limit', {}) or {}
                                final_qty = float(lock_info.get('final_qty') or 0)
                                if final_qty > 0:
                                    try:
                                        cap_rem = float(b.remaining_addable_qty_under_risk_limit(symbol))
                                    except Exception:
                                        cap_rem = final_qty
                                    add_qty = min(final_qty, cap_rem)
                                    if add_qty > 0:
                                        side_order = 'buy' if side == 'long' else 'sell'
                                        try:
                                            add_order = b.create_market_order_safe(symbol, side_order, add_qty)
                                            eff_add = float(add_order.get('amount', add_qty) or add_qty)
                                            # persist
                                            added_list.append(eff_add)
                                            st['pyramid_added'] = added_list
                                            eng.state[symbol] = st
                                            try:
                                                from strategy import save_state as _save
                                                _save(eng.state)
                                            except Exception:
                                                pass
                                            # logs
                                            reason_tag = f"PYRAMID_L{pyr_level}"
                                            side_tag = f"{side.upper()}_ADD"
                                            log_trade(symbol, side_tag, close, eff_add, reason_tag)
                                            try:
                                                sig_for_add = plan.get('signal', {})
                                                log_detailed_entry(symbol, side_tag, close, eff_add, new_trail, 1.0, sig_for_add, atr_abs, eq, reason_tag, ['pyramid_trigger'])
                                            except Exception:
                                                pass
                                            _log_json(
                                                logger,
                                                logging.INFO,
                                                {
                                                    'event': 'pyramid_add',
                                                    'symbol': symbol,
                                                    'level': pyr_level,
                                                    'qty': round(eff_add, 6),
                                                    'price': round(close, 4),
                                                },
                                            )
                                            levels_add = compute_tp_ladder(close, side, eff_add)
                                            place_tp_ladder(
                                                b,
                                                symbol,
                                                side,
                                                eff_add,
                                                close,
                                                levels_add,
                                                precision_info,
                                            )
                                            # After adding, refresh protective stop for total amount
                                            try:
                                                pos2 = b.position_for(symbol)
                                                amt2 = abs(float(pos2.get('contracts') or pos2.get('positionAmt') or 0)) if pos2 else 0.0
                                                if amt2 > 0:
                                                    _replace_stop_only(b, symbol, side, new_trail, amt2)
                                            except Exception:
                                                pass
                                        except Exception as _pe:
                                            _log_json(
                                                logger,
                                                logging.WARNING,
                                                {
                                                    'event': 'pyramid_add_failed',
                                                    'symbol': symbol,
                                                    'error': str(_pe),
                                                },
                                            )
                        except Exception:
                            pass

                    decision = plan.get('decision')
                    if is_new_bar and decision in ('ENTER_LONG', 'ENTER_SHORT'):
                        try:
                            metrics_mgr.inc_signal(symbol, TF)
                        except Exception:
                            pass
                    if not entries_blocked and decision in ('ENTER_LONG','ENTER_SHORT'):
                        if has_pos:
                            continue
                        side_order = 'buy' if decision == 'ENTER_LONG' else 'sell'
                        qty = float(plan.get('qty') or 0)
                        stop_price = float(plan.get('stop_price') or 0)
                        if qty <= 0 or stop_price <= 0:
                            continue
                        try:
                            order = b.create_market_order_safe(symbol, side_order, qty)
                            eff_qty = float(order.get('amount', qty) or qty)
                            if side_order == 'buy':
                                _replace_stop_only(b, symbol, 'long', stop_price, eff_qty)
                                risk_usdt = abs(close - stop_price) * eff_qty
                                eng.update_symbol_state_on_entry(symbol, 'long', close, eff_qty, entry_stop_price=stop_price, risk_usdt=risk_usdt)
                                levels = compute_tp_ladder(close, 'long', eff_qty)
                                place_tp_ladder(
                                    b,
                                    symbol,
                                    'long',
                                    eff_qty,
                                    close,
                                    levels,
                                    precision_info,
                                )
                            else:
                                _replace_stop_only(b, symbol, 'short', stop_price, eff_qty)
                                risk_usdt = abs(stop_price - close) * eff_qty
                                eng.update_symbol_state_on_entry(symbol, 'short', close, eff_qty, entry_stop_price=stop_price, risk_usdt=risk_usdt)
                                levels = compute_tp_ladder(close, 'short', eff_qty)
                                place_tp_ladder(
                                    b,
                                    symbol,
                                    'short',
                                    eff_qty,
                                    close,
                                    levels,
                                    precision_info,
                                )
                            metrics_mgr = get_metrics_manager()
                            if metrics_mgr is not None:
                                try:
                                    metrics_mgr.observe_order_latency('entry_market', time.perf_counter() - loop_start)
                                except Exception:
                                    pass
                            _log_json(
                                logger,
                                logging.INFO,
                                {
                                    'event': 'entry',
                                    'symbol': symbol,
                                    'decision': decision,
                                    'qty': round(eff_qty, 6),
                                    'price': round(close, 4),
                                    'stop': round(stop_price, 4),
                                },
                            )
                        except Exception as e:
                            try:
                                metrics_mgr.inc_order_error('order')
                            except Exception:
                                pass
                            _log_json(
                                logger,
                                logging.WARNING,
                                {
                                    'event': 'entry_failed',
                                    'symbol': symbol,
                                    'error': str(e),
                                },
                            )

                    # Detect exit (position disappeared while state says in_position)
                    st_state = eng.state.get(symbol, {}) if eng else {}
                    state_in_pos = bool(st_state.get('in_position', False))
                    if (not has_pos) and state_in_pos:
                        try:
                            side_state = (st_state.get('side') or 'long').lower()
                            entry_px = float(st_state.get('entry_price') or 0)
                            # Determine exit price
                            exit_price_used = 0.0
                            try:
                                trades = b.fetch_my_trades(symbol, limit=5)
                                if trades:
                                    exit_price_used = float(trades[-1].get('price') or 0)
                            except Exception:
                                pass
                            if not exit_price_used or exit_price_used <= 0:
                                exit_price_used = close
                                reason_lab = 'estimated_exit'
                            else:
                                reason_lab = 'confirmed_exit'
                            # PnL pct
                            if entry_px > 0:
                                if side_state == 'long':
                                    pnl_pct = ((exit_price_used - entry_px) / entry_px) * 100.0
                                else:
                                    pnl_pct = ((entry_px - exit_price_used) / entry_px) * 100.0
                            else:
                                pnl_pct = 0.0
                            log_exit(symbol, side_state, exit_price_used, pnl_pct, reason_lab)
                            # Update state: record structural reset on stop or clear on profit/flat
                            try:
                                if pnl_pct < -0.5:
                                    eng.record_stop_loss_exit(symbol, side_state, exit_price_used)
                                    _log_json(
                                        logger,
                                        logging.INFO,
                                        {
                                            'event': 'exit',
                                            'symbol': symbol,
                                            'side': side_state,
                                            'pnl_pct': round(pnl_pct, 4),
                                            'category': 'stop_loss',
                                        },
                                    )
                                else:
                                    eng.clear_position_state(symbol)
                                    _log_json(
                                        logger,
                                        logging.INFO,
                                        {
                                            'event': 'exit',
                                            'symbol': symbol,
                                            'side': side_state,
                                            'pnl_pct': round(pnl_pct, 4),
                                            'category': 'position_closed',
                                        },
                                    )
                            except Exception as _se:
                                _log_json(
                                    logger,
                                    logging.WARNING,
                                    {
                                        'event': 'exit_state_update_failed',
                                        'symbol': symbol,
                                        'error': str(_se),
                                    },
                                )
                            # Slack notification for EXIT only
                            try:
                                # Use qty from state if available to estimate PnL in USDT
                                st_after = eng.state.get(symbol, {}) if eng else {}
                                qty = float(st_after.get('original_qty') or st_state.get('original_qty') or 0.0)
                                if qty > 0 and entry_px > 0 and exit_price_used > 0:
                                    if side_state == 'long':
                                        pnl_usdt = (exit_price_used - entry_px) * qty
                                    else:
                                        pnl_usdt = (entry_px - exit_price_used) * qty
                                else:
                                    pnl_usdt = 0.0
                                # Current equity for context
                                try:
                                    equity_now = float(b.get_equity_usdt())
                                except Exception:
                                    equity_now = 0.0
                                slack_notify_exit(symbol, side_state, entry_px, exit_price_used, qty, float(pnl_usdt), float(pnl_pct), equity_now)
                                try:
                                    metrics_mgr.inc_trade_count()
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            # Update daily report
                            _emit_daily_report(
                                datetime.now(),
                                'testnet' if ('TESTNET' in globals() and TESTNET) else 'live',
                                DATA_BASE_DIR if 'DATA_BASE_DIR' in globals() else 'data',
                            )
                        except Exception as e:
                            _log_json(
                                logger,
                                logging.WARNING,
                                {
                                    'event': 'exit_detection_failed',
                                    'symbol': symbol,
                                    'error': str(e),
                                },
                            )
                except Exception as se:
                    logger.exception("Loop error for %s: %s", symbol, se)
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; emergency cleanup...")
            emergency_cleanup()
            os._exit(0)
        except Exception as e:
            try:
                msg = str(e).lower()
                if 'nonce' in msg:
                    emergency_mgr.record_error('nonce')
                    metrics_mgr.inc_order_error('nonce')
                elif 'timestamp' in msg or 'time drift' in msg:
                    emergency_mgr.record_error('time_drift')
                    metrics_mgr.inc_order_error('time_drift')
                _check_kill_switch()
            except Exception:
                pass
            logger.exception("Unexpected error: %s", e)
            _log_json(
                logger,
                logging.INFO,
                {'event': 'loop_retry', 'delay_sec': 3},
            )
            time.sleep(3)
        finally:
            metrics_mgr.observe_loop_latency((time.perf_counter() - loop_start) * 1000.0)
            lock.release()


if __name__ == "__main__":
    main()

