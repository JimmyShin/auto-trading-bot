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
from exchange_api import ExchangeAPI, BalanceAuthError, BalanceSyncError
from strategy import DonchianATREngine
from indicators import atr, is_near_funding
from reporter import Reporter

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

reporter = Reporter.from_config()
log_trade = reporter.log_trade
log_exit = reporter.log_exit
log_signal_analysis = reporter.log_signal_analysis
log_filtered_signal = reporter.log_filtered_signal
log_detailed_entry = reporter.log_detailed_entry


b_global = None
eng_global = None
lock = Lock()


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


def fetch_df(b: ExchangeAPI, symbol: str, tf: str, lookback: int) -> pd.DataFrame:
    ohlcv = b.fetch_ohlcv(symbol, tf, limit=lookback)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def main():
    print(f"🚀 Donchian-ATR Bot (SAFE MODE) | TF={TF} | Risk={RISK_PCT*100:.0f}% | Lev={LEVERAGE}x")
    print(f"Symbols: {', '.join(UNIVERSE)}")
    print("="*40)

    b = ExchangeAPI()
    b.connect()
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

