import os
import time
import signal
import atexit
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from threading import Lock

import pandas as pd

from config import *  # TF, RISK_PCT, LEVERAGE, UNIVERSE, SAFE_RESTART, ATR_LEN, POLL_SEC, DATA_BASE_DIR, TESTNET
from broker_binance import BinanceUSDM, BalanceAuthError, BalanceSyncError
from strategy import DonchianATREngine
from indicators import atr, is_near_funding

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

b_global = None
eng_global = None
lock = Lock()


def _data_base_dir() -> str:
    base = DATA_BASE_DIR if 'DATA_BASE_DIR' in globals() else 'data'
    env = 'testnet' if ('TESTNET' in globals() and TESTNET) else 'live'
    path = os.path.join(base, env)
    os.makedirs(path, exist_ok=True)
    return path


def _now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)


def log_trade(symbol: str, side: str, entry_price: float, qty: float, reason: str = "signal"):
    try:
        import csv

        ts = _now_str()
        date = _now_str("%Y-%m-%d")
        base = _data_base_dir()
        fn = os.path.join(base, f"trades_{date}.csv")
        exists = os.path.exists(fn)
        fieldnames = [
            'timestamp', 'symbol', 'side', 'entry_price', 'qty', 'reason', 'status',
            'exit_timestamp', 'exit_price', 'pnl_pct', 'exit_reason'
        ]

        row = {
            'timestamp': ts,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'qty': qty,
            'reason': reason,
            'status': 'OPEN',
            'exit_timestamp': '',
            'exit_price': '',
            'pnl_pct': '',
            'exit_reason': ''
        }

        write_header = not exists or os.path.getsize(fn) == 0
        with open(fn, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[WARN] trade log failed: {e}")

def log_exit(symbol: str, side: str, exit_price: float, pnl_pct: float, reason: str = "exit"):
    try:
        import csv

        ts = _now_str()
        date = _now_str("%Y-%m-%d")
        base = _data_base_dir()
        fn = os.path.join(base, f"trades_{date}.csv")
        fieldnames = [
            'timestamp', 'symbol', 'side', 'entry_price', 'qty', 'reason', 'status',
            'exit_timestamp', 'exit_price', 'pnl_pct', 'exit_reason'
        ]

        if not os.path.exists(fn):
            with open(fn, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    'timestamp': ts,
                    'symbol': symbol,
                    'side': side,
                    'entry_price': '',
                    'qty': '',
                    'reason': '',
                    'status': 'CLOSED',
                    'exit_timestamp': ts,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': reason
                })
            return

        rows = []
        with open(fn, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for fld in fieldnames:
                    row.setdefault(fld, '')
                rows.append(row)

        found = False
        for row in reversed(rows):
            if row.get('symbol') == symbol and row.get('status', '').upper() == 'OPEN':
                row['status'] = 'CLOSED'
                row['exit_timestamp'] = ts
                row['exit_price'] = exit_price
                row['pnl_pct'] = pnl_pct
                row['exit_reason'] = reason
                found = True
                break

        if not found:
            rows.append({
                'timestamp': ts,
                'symbol': symbol,
                'side': side,
                'entry_price': '',
                'qty': '',
                'reason': '',
                'status': 'CLOSED',
                'exit_timestamp': ts,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': reason
            })

        with open(fn, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({fld: row.get(fld, '') for fld in fieldnames})
    except Exception as e:
        print(f"[WARN] exit log failed: {e}")

def log_signal_analysis(symbol,
                        current_price,
                        sig,
                        is_new_bar=None,
                        funding_avoid=None,
                        daily_loss_hit=None,
                        equity_usdt=None,
                        daily_dd_pct=None,
                        decision=None,
                        skip_reason=""):
    """Append a row to signal_analysis_YYYY-MM-DD.csv with key fields."""
    try:
        ts = _now_str()
        date = _now_str("%Y-%m-%d")
        base = _data_base_dir()
        fn = os.path.join(base, f"signal_analysis_{date}.csv")
        exists = os.path.exists(fn)
        if isinstance(is_new_bar, dict) and not isinstance(funding_avoid, (bool, type(None))):
            legacy_analysis = is_new_bar
            legacy_decision = funding_avoid
            legacy_skip = daily_loss_hit
            is_new_bar = legacy_analysis.get('is_new_bar')
            funding_avoid = legacy_analysis.get('funding_avoid')
            daily_loss_hit = legacy_analysis.get('daily_loss_hit')
            equity_usdt = legacy_analysis.get('equity_usdt')
            daily_dd_pct = legacy_analysis.get('daily_dd_pct')
            if decision is None:
                decision = legacy_decision
            if not skip_reason and legacy_skip is not None:
                skip_reason = legacy_skip
        try:
            equity_usdt = float(equity_usdt if equity_usdt is not None else 0.0)
        except Exception:
            equity_usdt = 0.0
        try:
            daily_dd_pct = float(daily_dd_pct if daily_dd_pct is not None else 0.0)
        except Exception:
            daily_dd_pct = 0.0
        decision = decision if decision is not None else ''

        fast_ma = sig.get('fast_ma', 0.0) if isinstance(sig, dict) else 0.0
        slow_ma = sig.get('slow_ma', 0.0) if isinstance(sig, dict) else 0.0
        ma_diff_pct = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        align = (sig or {}).get('alignment', {}) or {}
        cflt = (sig or {}).get('candle_filter', {}) or {}
        rmult = (sig or {}).get('risk_multiplier', {}) or {}

        import csv
        with open(fn, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    'timestamp','symbol','price',
                    'fast_ma','slow_ma','ma_diff_pct',
                    'long_signal','short_signal','regime',
                    'long_aligned','short_aligned','long_cross','short_cross',
                    'candle_position_ratio','candle_safe_long','candle_safe_short',
                    'risk_multiplier_long','risk_multiplier_short',
                    'is_new_bar','funding_avoid','daily_loss_hit',
                    'equity_usdt','daily_dd_pct','decision','skip_reason'
                ])
            w.writerow([
                ts, symbol, current_price,
                fast_ma, slow_ma, round(ma_diff_pct, 3),
                bool((sig or {}).get('long')), bool((sig or {}).get('short')), (sig or {}).get('regime','UNKNOWN'),
                bool(align.get('long_aligned')), bool(align.get('short_aligned')),
                bool(align.get('long_cross')), bool(align.get('short_cross')),
                cflt.get('candle_position_ratio', 0.0), bool(cflt.get('candle_safe_long')), bool(cflt.get('candle_safe_short')),
                rmult.get('long', 1.0), rmult.get('short', 1.0),
                bool(is_new_bar), bool(funding_avoid), bool(daily_loss_hit),
                float(equity_usdt), round(float(daily_dd_pct), 3), decision or '', skip_reason or ''
            ])
    except Exception as e:
        print(f"[WARN] signal log failed: {e}")


def log_filtered_signal(symbol,
                       current_price,
                       sig,
                       analysis=None,
                       decision="SKIP",
                       skip_reason="",
                       **extra):
    """Persist details about signals that were skipped by filters."""
    try:
        ts = _now_str()
        date = _now_str("%Y-%m-%d")
        base = _data_base_dir()
        fn = os.path.join(base, f"filtered_signals_{date}.csv")
        exists = os.path.exists(fn)

        sig = sig or {}
        analysis_payload = {}
        if isinstance(analysis, dict):
            analysis_payload.update({k: analysis[k] for k in analysis.keys()})
        if extra:
            analysis_payload.update({k: extra[k] for k in extra.keys() if k not in analysis_payload})

        fast_ma = sig.get('fast_ma', 0.0)
        slow_ma = sig.get('slow_ma', 0.0)
        ma_diff = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        candle_filter = sig.get('candle_filter', {}) or {}

        import csv
        with open(fn, 'a', encoding='utf-8', newline='') as f:
            fieldnames = [
                'timestamp','symbol','price','decision','skip_reason',
                'fast_ma','slow_ma','ma_diff_pct','long_signal','short_signal',
                'candle_position_ratio','candle_safe_long','candle_safe_short',
                'context'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            row = {
                'timestamp': ts,
                'symbol': symbol,
                'price': float(current_price),
                'decision': decision or 'SKIP',
                'skip_reason': skip_reason or analysis_payload.get('skip_reason', ''),
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'ma_diff_pct': round(ma_diff, 3),
                'long_signal': bool(sig.get('long', False)),
                'short_signal': bool(sig.get('short', False)),
                'candle_position_ratio': candle_filter.get('candle_position_ratio', 0.0),
                'candle_safe_long': bool(candle_filter.get('candle_safe_long')),
                'candle_safe_short': bool(candle_filter.get('candle_safe_short')),
                'context': json.dumps(analysis_payload, default=str) if analysis_payload else ''
            }
            writer.writerow(row)
    except Exception as e:
        print(f"[WARN] filtered signal log failed: {e}")

def log_detailed_entry(symbol,
                       side,
                       entry_price,
                       qty,
                       stop_price,
                       risk_multiplier,
                       sig,
                       atr_abs,
                       equity_usdt=None,
                       reason="signal",
                       reasons_list=None,
                       **kwargs):
    """Append a detailed entry row to detailed_entries_YYYY-MM-DD.csv."""
    try:
        ts = _now_str()
        date = _now_str("%Y-%m-%d")
        base = _data_base_dir()
        fn = os.path.join(base, f"detailed_entries_{date}.csv")
        equity_alias = kwargs.pop('equity', None)
        if kwargs:
            unexpected = ', '.join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if equity_usdt is not None and equity_alias is not None:
            try:
                if abs(float(equity_usdt) - float(equity_alias)) > 1e-6:
                    print(f"[WARN] log_detailed_entry equity mismatch ({equity_usdt} vs {equity_alias})")
            except Exception:
                pass
        equity_value = equity_usdt if equity_usdt is not None else equity_alias
        if equity_value is None:
            raise ValueError('equity_usdt or equity must be provided')
        try:
            equity_value = float(equity_value)
        except Exception:
            raise ValueError('equity must be numeric') from None
        equity_usdt = equity_value
        exists = os.path.exists(fn)

        position_value = float(qty) * float(entry_price)
        risk_amount = float(qty) * abs(float(entry_price) - float(stop_price))
        risk_pct = (risk_amount / max(float(equity_usdt), 1e-9)) * 100.0
        stop_distance_pct = abs(float(entry_price) - float(stop_price)) / max(float(entry_price), 1e-9) * 100.0
        if str(side).upper().startswith('LONG'):
            be_promotion_price = float(entry_price) + 1.5 * float(atr_abs)
            expected_trail_range = f"${entry_price:.0f} ~ ${be_promotion_price + (2.5 * atr_abs):.0f}"
        else:
            be_promotion_price = float(entry_price) - 1.5 * float(atr_abs)
            expected_trail_range = f"${be_promotion_price - (2.5 * atr_abs):.0f} ~ ${entry_price:.0f}"

        fast_ma = (sig or {}).get('fast_ma', 0.0)
        slow_ma = (sig or {}).get('slow_ma', 0.0)
        ma_diff_pct = ((float(fast_ma) / float(slow_ma) - 1) * 100.0) if slow_ma else 0.0
        regime = (sig or {}).get('regime','UNKNOWN')
        entry_logic = "Trend alignment with reduced risk" if float(risk_multiplier or 1.0) < 1.0 else "Trend alignment with full risk"
        reasons_str = "|".join(reasons_list) if reasons_list else ""

        import csv
        with open(fn, 'a', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    'timestamp','symbol','side','entry_price','qty','position_value_usd',
                    'stop_price','stop_distance_pct','risk_amount_usd','risk_pct','risk_multiplier',
                    'be_promotion_price','expected_trail_range',
                    'fast_ma','slow_ma','ma_diff_pct','regime','atr_abs','equity_usdt','reason','entry_logic','reasons'
                ])
            w.writerow([
                ts, symbol, side, entry_price, qty, round(position_value,2),
                stop_price, round(stop_distance_pct,3), round(risk_amount,2), round(risk_pct,2), float(risk_multiplier or 1.0),
                be_promotion_price, expected_trail_range,
                fast_ma, slow_ma, round(ma_diff_pct,3), regime, atr_abs, equity_usdt, reason, entry_logic, reasons_str
            ])
    except Exception as e:
        print(f"[WARN] detailed entry log failed: {e}")

def _ensure_protective_stop_on_restart(b: BinanceUSDM, eng: DonchianATREngine, symbol: str) -> bool:
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
        b.cancel_all(symbol)
        if side_label == 'long':
            b.create_stop_market_safe(symbol, 'sell', float(stop_price), amt, reduce_only=True)
        else:
            b.create_stop_market_safe(symbol, 'buy', float(stop_price), amt, reduce_only=True)
        print(f"[SAFE] {symbol} restart protective stop @ {float(stop_price):.4f}")
        return True
    except BalanceAuthError as auth_err:
        print(f"[FATAL] {symbol} restart stop failed: {auth_err}")
        raise
    except Exception as e:
        print(f"[WARN] {symbol} restart stop failed: {e}")
        return False


def policy_emergency_cleanup():
    """Policy-based emergency handling (safe minimal).

    Re-arm protective stops for any open positions to avoid forced closes
    when the app receives a termination signal.
    """
    global b_global, eng_global
    if not b_global or not eng_global:
        return
    fatal_auth = False
    fatal_msg = None
    try:
        for symbol in UNIVERSE:
            if fatal_auth:
                break
            try:
                _ensure_protective_stop_on_restart(b_global, eng_global, symbol)
            except BalanceAuthError as auth_err:
                fatal_auth = True
                fatal_msg = auth_err
                print(f"[FATAL] {symbol} emergency protect failed: {auth_err}")
            except Exception as e:
                print(f"[WARN] {symbol} emergency protect failed: {e}")
        if fatal_auth:
            print('[FATAL] Emergency cleanup aborted because Binance rejected API credentials.')
        else:
            print('[OK] Emergency handling completed')
    except Exception as e:
        print(f"[ERR] Emergency handling error: {e}")


# Legacy name for other call sites
emergency_cleanup = policy_emergency_cleanup


def signal_handler(signum, frame):
    emergency_cleanup()
    os._exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(emergency_cleanup)


def fetch_df(b: BinanceUSDM, symbol: str, tf: str, lookback: int) -> pd.DataFrame:
    ohlcv = b.fetch_ohlcv(symbol, tf, limit=lookback)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main():
    print(f"🚀 Donchian-ATR Bot (SAFE MODE) | TF={TF} | Risk={RISK_PCT*100:.0f}% | Lev={LEVERAGE}x")
    print(f"Symbols: {', '.join(UNIVERSE)}")
    print("="*40)

    b = BinanceUSDM()
    b.load_markets()
    eng = DonchianATREngine()
    global b_global, eng_global
    b_global, eng_global = b, eng

    last_equity = 0.0
    try:
        last_equity = float(b.get_equity_usdt())
        eng.reset_daily_anchor(last_equity)
    except BalanceAuthError as auth_err:
        logger.error('Startup balance auth error: %s', auth_err)
        print('[FATAL] Binance authentication failed on startup; check API key/IP permissions.')
        emergency_cleanup()
        return
    except BalanceSyncError as sync_err:
        logger.error('Startup balance sync error: %s', sync_err)
        print('[FATAL] Unable to sync with Binance server time on startup.')
        emergency_cleanup()
        return
    except Exception as bal_err:
        logger.warning('Startup balance fetch failed; continuing with 0 equity: %s', bal_err)

    # On startup, re-arm protective stops for any existing positions
    try:
        for symbol in UNIVERSE:
            _ensure_protective_stop_on_restart(b, eng, symbol)
        print("Trading loop starting...")
    except BalanceAuthError as auth_err:
        logger.error('Startup protective stop auth error: %s', auth_err)
        print('[FATAL] Binance rejected credentials while restoring protective stops.')
        emergency_cleanup()
        return
    except Exception as e:
        print(f"[WARN] startup protect failed: {e}")

    last_bar_ts = {sym: None for sym in UNIVERSE}
    last_heartbeat = 0.0

    while True:
        lock.acquire()
        try:
            eq = last_equity
            try:
                fresh_eq = float(b.get_equity_usdt())
                eq = fresh_eq
                last_equity = fresh_eq
                eng.reset_daily_anchor(fresh_eq)
            except BalanceAuthError as auth_err:
                logger.error('Balance auth error: %s', auth_err)
                print('[FATAL] Binance authentication failed; stopping bot for safety.')
                emergency_cleanup()
                raise SystemExit(1)
            except BalanceSyncError as sync_err:
                logger.error('Balance sync failure: %s', sync_err)
                print('[FATAL] Unable to stay behind Binance server time; stopping bot.')
                emergency_cleanup()
                raise SystemExit(2)
            except Exception as bal_err:
                logger.warning('Balance fetch error, using cached equity: %s', bal_err)
                if last_equity > 0:
                    print(f'[WARN] Using cached equity {last_equity:.2f} after balance error.')
                else:
                    print(f'[WARN] Balance fetch failed before first successful read: {bal_err}')
            # Heartbeat every ~30s
            now = time.time()
            if now - last_heartbeat > 30:
                print(f"[HB] {datetime.utcnow().strftime('%H:%M:%S')} equity={eq:.2f}")
                # Position snapshots
                try:
                    for _sym in UNIVERSE:
                        try:
                            _pos = b.position_for(_sym)
                            _amt = abs(float(_pos.get('contracts') or _pos.get('positionAmt') or 0)) if _pos else 0.0
                            if _pos and _amt > 0:
                                _side = (_pos.get('side') or ('long' if (_pos.get('contracts') or 0) > 0 else 'short')).lower()
                                _st = eng.state.get(_sym, {}) if eng else {}
                                _entry = float(_st.get('entry_price') or 0)
                                _trail = _st.get('last_trail_stop')
                                print(f"[POS] {_sym} {_side} qty={_amt:.6f} entry={_entry:.4f} stop={(float(_trail) if _trail else 0):.4f}")
                        except Exception:
                            pass
                except Exception:
                    pass
                last_heartbeat = now

            for symbol in UNIVERSE:
                try:
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
                        dd_pct = ((anchor - eq) / max(anchor, 1e-9)) * 100.0
                        log_signal_analysis(symbol, close, sig, is_new_bar, funding_avoid, daily_loss_hit, eq, dd_pct, plan.get('decision'), plan.get('skip_reason'))
                    except Exception as _e:
                        print(f"[WARN] signal log error {symbol}: {_e}")
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
                        print(f"[BAR] {symbol} {bar_ts.strftime('%Y-%m-%d %H:%M')} close={close:.4f} atr={atr_abs:.2f} fast={fast_ma:.2f} slow={slow_ma:.2f} ma={ma_diff_pct:+.2f}% pos={pos_ratio:.2f} rmL={rmL} rmS={rmS} decision={dec} skip={sk}")

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
                            b.cancel_all(symbol)
                            amt = abs(float(pos.get('contracts') or pos.get('positionAmt') or 0))
                            if side == 'long':
                                b.create_stop_market_safe(symbol, 'sell', new_trail, amt, reduce_only=True)
                            else:
                                b.create_stop_market_safe(symbol, 'buy', new_trail, amt, reduce_only=True)
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
                                            print(f"[ADD] {symbol} level={pyr_level} qty={eff_add:.6f} px={close:.4f}")
                                            # After adding, refresh protective stop for total amount
                                            try:
                                                pos2 = b.position_for(symbol)
                                                amt2 = abs(float(pos2.get('contracts') or pos2.get('positionAmt') or 0)) if pos2 else 0.0
                                                if amt2 > 0:
                                                    b.cancel_all(symbol)
                                                    if side == 'long':
                                                        b.create_stop_market_safe(symbol, 'sell', new_trail, amt2, reduce_only=True)
                                                    else:
                                                        b.create_stop_market_safe(symbol, 'buy', new_trail, amt2, reduce_only=True)
                                            except Exception:
                                                pass
                                        except Exception as _pe:
                                            print(f"[WARN] pyramiding add failed {symbol}: {_pe}")
                        except Exception:
                            pass

                    decision = plan.get('decision')
                    if decision in ('ENTER_LONG','ENTER_SHORT'):
                        if has_pos:
                            continue
                        side_order = 'buy' if decision == 'ENTER_LONG' else 'sell'
                        qty = float(plan.get('qty') or 0)
                        stop_price = float(plan.get('stop_price') or 0)
                        if qty <= 0 or stop_price <= 0:
                            continue
                        try:
                            b.cancel_all(symbol)
                            order = b.create_market_order_safe(symbol, side_order, qty)
                            eff_qty = float(order.get('amount', qty) or qty)
                            if side_order == 'buy':
                                b.create_stop_market(symbol, 'sell', stop_price, eff_qty, reduce_only=True)
                                eng.update_symbol_state_on_entry(symbol, 'long', close, eff_qty)
                                log_trade(symbol, 'LONG', close, eff_qty, 'MA_CROSS')
                                try:
                                    log_detailed_entry(symbol, 'LONG', close, eff_qty, stop_price, plan.get('risk_multiplier', 1.0), plan.get('signal', {}), atr_abs, eq, 'MA_ALIGNMENT', plan.get('reasons'))
                                except Exception:
                                    pass
                            else:
                                b.create_stop_market(symbol, 'buy', stop_price, eff_qty, reduce_only=True)
                                eng.update_symbol_state_on_entry(symbol, 'short', close, eff_qty)
                                log_trade(symbol, 'SHORT', close, eff_qty, 'MA_CROSS')
                                try:
                                    log_detailed_entry(symbol, 'SHORT', close, eff_qty, stop_price, plan.get('risk_multiplier', 1.0), plan.get('signal', {}), atr_abs, eq, 'MA_ALIGNMENT', plan.get('reasons'))
                                except Exception:
                                    pass
                            print(f"[ENTRY] {symbol} {decision} qty={eff_qty:.6f} px={close:.4f} stop={stop_price:.4f}")
                        except Exception as e:
                            print(f"[WARN] entry failed {symbol}: {e}")

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
                                trades = b.exchange.fetch_my_trades(symbol, limit=5)
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
                                    print(f"[EXIT] {symbol} stop-loss recorded {pnl_pct:+.2f}%")
                                else:
                                    eng.clear_position_state(symbol)
                                    print(f"[EXIT] {symbol} position closed {pnl_pct:+.2f}%")
                            except Exception as _se:
                                print(f"[WARN] state update after exit failed {symbol}: {_se}")
                            # Update daily report
                            try:
                                if _gen_daily_report:
                                    date_str = datetime.now().strftime('%Y-%m-%d')
                                    env_label = 'testnet' if ('TESTNET' in globals() and TESTNET) else 'live'
                                    base_dir = DATA_BASE_DIR if 'DATA_BASE_DIR' in globals() else 'data'
                                    _gen_daily_report(date_str, env_label, base_dir)
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"[WARN] exit detection failed {symbol}: {e}")
                except Exception as se:
                    logger.exception("Loop error for %s: %s", symbol, se)
            time.sleep(POLL_SEC)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received; emergency cleanup...")
            emergency_cleanup()
            os._exit(0)
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            print("3s retry...")
            time.sleep(3)
        finally:
            lock.release()


if __name__ == "__main__":
    main()

