import ccxt
import time
import requests
from typing import Dict, Any, List, Optional
from config import BINANCE_KEY, BINANCE_SECRET, TESTNET
import math

class BalanceSyncError(Exception):
    """Raised when balance retrieval fails because the client clock is ahead of the exchange."""
    pass

class BalanceAuthError(Exception):
    """Raised when Binance rejects the API credentials or permissions."""
    pass

class BinanceUSDM:
    def __init__(self):
        # ?쒓컙 ?숆린??愿???띿꽦 珥덇린??
        self.last_sync_time = 0
        self.time_diff = 0
        self.sync_interval = 1800  # 30遺꾨쭏???щ룞湲고솕 (???먯＜)
        self.sync_failed_count = 0
        self.max_sync_fails = 3  # 3???곗냽 ?ㅽ뙣 ??媛뺤젣 ?ъ떆??
        
        self.exchange = ccxt.binanceusdm({
            "apiKey": BINANCE_KEY,
            "secret": BINANCE_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future", "adjustForTimeDifference": True, "recvWindow": 60000, "timeDifference": 0},
            "timeout": 30000,
            "recvWindow": 60000,  # 1遺꾩쑝濡??뺣?
        })
        
        # ?쒕쾭 ?쒓컙 ?숆린??(Exchange 珥덇린????
        if not self.sync_server_time():
            print("CRITICAL initial time sync failed - manual check required")
            raise Exception("Time sync failed - safe trading not possible")
        
        # ?뚯뒪?몃꽬 ?쇱슦??
        if TESTNET:
            self.exchange.set_sandbox_mode(True)
        
        self._last_equity = None
        self._balance_failures = 0
        self._last_balance_warning = 0.0
    
    def sync_server_time(self, force_print=True):
        """바이낸스 서버 시간 동기화 (개선된 버전)"""
        max_attempts = 3
        futures_url = 'https://fapi.binance.com/fapi/v1/time'
        if TESTNET:
            futures_url = 'https://testnet.binancefuture.com/fapi/v1/time'
        spot_fallback_url = 'https://api.binance.com/api/v3/time'

        for attempt in range(max_attempts):
            try:
                request_time = time.time() * 1000
                try:
                    response = requests.get(futures_url, timeout=10)
                except Exception:
                    response = requests.get(spot_fallback_url, timeout=10)
                response_time = time.time() * 1000

                if response.status_code != 200:
                    raise Exception(f"서버 응답 에러: {response.status_code}")

                payload = response.json()
                server_time = payload.get('serverTime') or payload.get('server_time')
                if server_time is None:
                    raise Exception('serverTime missing in response')
                server_time = float(server_time)

                network_latency = (response_time - request_time) / 2
                adjusted_server_time = server_time + network_latency

                local_time = time.time() * 1000
                time_diff = adjusted_server_time - local_time
                safety_lead_ms = 600
                effective_diff = time_diff - safety_lead_ms
                effective_diff = min(effective_diff, -float(safety_lead_ms))

                if abs(time_diff) > 30000:
                    print(f"CRITICAL time diff: {time_diff:.0f}ms")
                    print("   Recommend checking system time or reboot")
                elif abs(time_diff) > 5000:
                    print(f"WARNING large time diff: {time_diff:.0f}ms")
                elif force_print:
                    print(f"Time sync: {time_diff:.0f}ms diff detected")

                self.exchange.options['adjustForTimeDifference'] = True
                try:
                    self.exchange.options['timeDifference'] = int(effective_diff)
                except Exception:
                    self.exchange.options['timeDifference'] = 0
                try:
                    self.exchange.timeDifference = int(effective_diff)
                except Exception:
                    pass

                self.last_sync_time = time.time()
                self.time_diff = effective_diff
                self.sync_failed_count = 0
                return True

            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"WARNING time sync attempt {attempt + 1}/{max_attempts} failed: {e}")
                    time.sleep(2)
                else:
                    print(f"ERROR time sync completely failed: {e}")
                    print("Proceeding with default settings but API errors likely")
                    self.sync_failed_count += 1
                    return False

        self.sync_failed_count += 1
        return False

    def check_and_resync_if_needed(self):
        """?꾩슂???먮룞 ?щ룞湲고솕 (媛뺥솕??濡쒖쭅)"""
        current_time = time.time()
        
        # 議곌굔 1: ?뺢린???숆린??(30遺꾨쭏??
        time_since_sync = current_time - self.last_sync_time
        needs_regular_sync = time_since_sync > self.sync_interval
        
        # 議곌굔 2: ?곗냽 ?ㅽ뙣 ??媛뺤젣 ?숆린??(10遺?媛꾧꺽)
        needs_forced_sync = (self.sync_failed_count >= self.max_sync_fails and 
                           time_since_sync > 600)  # 10遺?
        
        # 議곌굔 3: ?쒓컙 李⑥씠媛 ??寃쎌슦 (5珥??댁긽)
        needs_drift_sync = abs(self.time_diff) > 5000
        
        if needs_regular_sync:
            print("Regular time sync...")
            return self.sync_server_time(force_print=False)
        elif needs_forced_sync:
            print(f"Force sync after {self.sync_failed_count} failures...")
            return self.sync_server_time(force_print=True)
        elif needs_drift_sync:
            print(f"Time drift detected ({self.time_diff:.0f}ms), resyncing...")
            return self.sync_server_time(force_print=True)
        
        return True

    def sync_with_ccxt_builtin(self) -> bool:
        """Sync using ccxt's built-in load_time_difference()."""
        try:
            diff = self.exchange.load_time_difference()
            try:
                self.exchange.timeDifference = int(diff)
            except Exception:
                pass
            self.exchange.options['adjustForTimeDifference'] = True
            try:
                self.exchange.options['timeDifference'] = int(diff)
            except Exception:
                self.exchange.options['timeDifference'] = 0
            self.last_sync_time = time.time()
            self.time_diff = diff
            self.sync_failed_count = 0
            print(f"Time sync(ccxt): {diff:.0f}ms diff loaded")
            return True
        except Exception as e:
            print(f"[WARN] ccxt load_time_difference failed: {e}")
            return False

    async def close(self):
        pass  # ccxt sync client

    def load_markets(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.check_and_resync_if_needed()  # ?쒓컙 ?숆린???뺤씤
                return self.exchange.load_markets()
            except ccxt.InvalidNonce as e:
                if attempt < max_retries - 1:
                    print(f"?좑툘 ??꾩뒪?ы봽 ?먮윭 (?쒕룄 {attempt + 1}/{max_retries}), ?щ룞湲고솕...")
                    time.sleep(2)
                    self.sync_server_time(force_print=True)
                else:
                    print(f"??留덉폆 濡쒕뱶 理쒖쥌 ?ㅽ뙣: {e}")
                    raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"?좑툘 留덉폆 濡쒕뱶 ?ㅽ뙣 (?쒕룄 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    print(f"??留덉폆 濡쒕뱶 理쒖쥌 ?ㅽ뙣: {e}")
                    raise e

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 400):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.check_and_resync_if_needed()
                return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            except ccxt.InvalidNonce as e:
                if attempt < max_retries - 1:
                    print(f"?좑툘 ??꾩뒪?ы봽 ?먮윭, ?щ룞湲고솕 ???ъ떆??..")
                    self.sync_server_time(force_print=False)
                    time.sleep(1)
                else:
                    raise e

    def fetch_balance(self) -> Dict[str, Any]:
        max_retries = 3
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                self.check_and_resync_if_needed()
                balance = self.exchange.fetch_balance({'type': 'future'})
                self._balance_failures = 0
                return balance
            except ccxt.InvalidNonce as e:
                last_err = e
                if attempt < max_retries - 1:
                    print('[WARN] Balance nonce mismatch; resyncing and retrying...')
                    self.sync_server_time(force_print=True)
                    try:
                        self.sync_with_ccxt_builtin()
                    except Exception:
                        pass
                    time.sleep(1)
                    continue
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if ('-2015' in msg) or ('Invalid API-key' in msg):
                    raise BalanceAuthError(msg) from e
                if ('-1021' in msg) or ('Timestamp for this request was' in msg):
                    if attempt < max_retries - 1:
                        print('[WARN] Balance timestamp drift detected; resyncing and retrying...')
                        self.sync_server_time(force_print=True)
                        try:
                            self.sync_with_ccxt_builtin()
                        except Exception:
                            pass
                        time.sleep(1)
                        continue
                    break
                if attempt < max_retries - 1:
                    print(f'[WARN] Balance fetch failed (attempt {attempt + 1}/{max_retries}): {e}')
                    time.sleep(1)
                    continue
                break
        if last_err:
            msg = str(last_err)
            if ('-2015' in msg) or ('Invalid API-key' in msg):
                raise BalanceAuthError(msg) from last_err
            if ('-1021' in msg) or ('Timestamp for this request was' in msg):
                raise BalanceSyncError(msg) from last_err
            raise BalanceSyncError(msg) from last_err
        raise BalanceSyncError('fetch_balance failed')

    def get_equity_usdt(self) -> float:
        try:
            bal = self.fetch_balance()
        except BalanceAuthError:
            self._balance_failures += 1
            raise
        except BalanceSyncError as sync_err:
            self._balance_failures += 1
            if self._last_equity is not None and self._balance_failures <= 5:
                now = time.time()
                if now - self._last_balance_warning > 30:
                    print(f"[WARN] Using cached equity {self._last_equity:.2f} due to sync issue: {sync_err}")
                    self._last_balance_warning = now
                return self._last_equity
            raise
        except Exception as generic_err:
            self._balance_failures += 1
            if self._last_equity is not None and self._balance_failures <= 3:
                now = time.time()
                if now - self._last_balance_warning > 30:
                    print(f"[WARN] Using cached equity {self._last_equity:.2f} after balance error: {generic_err}")
                    self._last_balance_warning = now
                return self._last_equity
            raise

        self._balance_failures = 0
        self._last_balance_warning = 0.0
        if 'total' in bal and 'USDT' in bal['total']:
            equity = float(bal['total']['USDT'])
        else:
            equity = float(bal['USDT']['total'])
        self._last_equity = equity
        return equity


    def set_leverage(self, symbol: str, lev: int):
        try:
            self.exchange.fapiPrivate_post_leverage({
                "symbol": symbol.replace("/",""),
                "leverage": lev
            })
        except Exception:
            pass

    def set_isolated(self, symbol: str):
        try:
            self.exchange.fapiPrivate_post_margintype({
                "symbol": symbol.replace("/",""),
                "marginType": "ISOLATED"
            })
        except Exception:
            pass

    def set_testnet(self, enabled: bool) -> bool:
        """Toggle Binance futures testnet mode at runtime.

        Returns True on success, False otherwise.
        """
        try:
            self.exchange.set_sandbox_mode(bool(enabled))
            return True
        except Exception:
            return False

    def fetch_positions(self, symbols: List[str]):
        try:
            return self.exchange.fetch_positions(symbols)
        except Exception as e:
            msg = str(e)
            if ('-2015' in msg) or ('Invalid API-key' in msg):
                raise BalanceAuthError(msg) from e
            raise

    def position_for(self, symbol: str) -> Optional[dict]:
        try:
            ps = self.fetch_positions([symbol])
        except BalanceAuthError:
            raise

        symbol_variants = [
            symbol,                    # BTC/USDT
            symbol.replace('/', ''),   # BTCUSDT  
            symbol + ':USDT'          # BTC/USDT:USDT
        ]
        
        for p in ps:
            p_symbol = p.get('symbol', '')
            for variant in symbol_variants:
                if p_symbol == variant or p_symbol.split(':')[0] == variant:
                    amt = abs(float(p.get('contracts') or p.get('positionAmt') or 0))
                    if amt > 1e-8:  # 留ㅼ슦 ?묒? 媛믩룄 怨좊젮
                        return p
        return None
    
    def get_actual_position_size(self, symbol: str) -> float:
        """?ㅼ젣 ?ъ????ш린 諛섑솚 (signed)"""
        pos = self.position_for(symbol)
        if pos:
            return float(pos.get('contracts') or pos.get('positionAmt') or 0)
        return 0.0

    def cancel_all(self, symbol: str):
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception:
            pass

    # --- 二쇰Ц ?좏떥 ---
    def create_stop_entry(self, symbol: str, side: str, stop_price: float, amount: float):
        """
        ?뚰뙆 吏꾩엯?? STOP_MARKET
        side: 'buy'/'sell'
        """
        params = {
            "stopPrice": self.exchange.price_to_precision(symbol, stop_price),
            "workingType": "MARK_PRICE",
            "reduceOnly": False,
        }
        return self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)

    def create_stop_market(self, symbol: str, side: str, stop_price: float, amount: float, reduce_only=True):
        """
        ?먯젅/?몃젅?쇱슜: STOP_MARKET reduceOnly
        """
        params = {
            "stopPrice": self.exchange.price_to_precision(symbol, stop_price),
            "workingType": "MARK_PRICE",
            "reduceOnly": reduce_only,
        }
        return self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)

    def fetch_ticker(self, symbol: str):
        return self.exchange.fetch_ticker(symbol)

    # --- Helpers for risk-limit-aware sizing ---
    def _get_mark_price(self, symbol: str) -> float:
        try:
            sym = symbol.replace('/', '')
            info = self.exchange.fapiPublic_get_premiumindex({ 'symbol': sym })
            if isinstance(info, list) and info:
                info = info[0]
            mp = float(info.get('markPrice', 0))
            if mp > 0:
                return mp
        except Exception:
            pass
        try:
            t = self.fetch_ticker(symbol)
            return float(t.get('last') or t.get('bid') or t.get('ask') or 0)
        except Exception:
            return 0.0

    def _get_step_size(self, symbol: str) -> float:
        try:
            m = self.exchange.market(symbol)
            filters = m.get('info', {}).get('filters', [])
            lot = next((f for f in filters if f.get('filterType') == 'LOT_SIZE'), None)
            if lot and 'stepSize' in lot:
                return float(lot['stepSize'])
        except Exception:
            pass
        # fallback to precision if available
        try:
            p = self.exchange.market(symbol).get('precision', {}).get('amount')
            if isinstance(p, int):
                return 10 ** (-p)
        except Exception:
            pass
        return 0.0

    def _floor_to_step(self, qty: float, step: float) -> float:
        if step and step > 0:
            return math.floor(qty / step) * step
        return qty

    def _get_leverage_brackets(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Return leverage brackets list for the symbol, or None on failure.
        Each bracket has notionalFloor/notionalCap fields among others.
        """
        try:
            sym = symbol.replace('/', '')
            # Try snake_case name
            getter = getattr(self.exchange, 'fapiPrivate_get_leveragebracket', None)
            data = None
            if callable(getter):
                data = getter({ 'symbol': sym })
            else:
                # Try camelCase fallback
                getter2 = getattr(self.exchange, 'fapiPrivateGetLeverageBracket', None)
                if callable(getter2):
                    data = getter2({ 'symbol': sym })
            if data is None:
                return None
            # ccxt returns list with one element per symbol
            if isinstance(data, list) and data:
                item = data[0]
                brackets = item.get('brackets') or item.get('brackets'.lower()) or []
                # Normalize numeric fields
                for b in brackets:
                    for k in ('notionalFloor', 'notionalCap'):
                        if k in b:
                            b[k] = float(b[k])
                return brackets
        except Exception:
            return None
        return None

    def remaining_addable_qty_under_risk_limit(self, symbol: str) -> float:
        """Compute how many contracts can still be added without exceeding
        Binance USDM risk-limit (max notional for current bracket).

        Returns 0 if already at cap or on failure to determine cap.
        """
        try:
            mark = self._get_mark_price(symbol)
            if mark <= 0:
                return 0.0
            pos = self.position_for(symbol)
            current_contracts = float(pos.get('contracts') or pos.get('positionAmt') or 0) if pos else 0.0
            current_notional = abs(current_contracts) * mark

            brackets = self._get_leverage_brackets(symbol)
            if not brackets:
                return 0.0
            # find current bracket by current_notional floor<=x<cap
            bracket = None
            for b in brackets:
                floor = float(b.get('notionalFloor', 0) or 0)
                cap = float(b.get('notionalCap', 0) or 0)
                if current_notional >= floor and (cap == 0 or current_notional < cap):
                    bracket = b
                    break
            if bracket is None:
                # If not found, take the last as cap
                bracket = brackets[-1]
            cap = float(bracket.get('notionalCap', 0) or 0)
            if cap <= 0:
                return 0.0
            remaining_notional = max(0.0, cap - current_notional)
            remaining_qty = remaining_notional / mark
            step = self._get_step_size(symbol)
            floored = self._floor_to_step(remaining_qty, step)
            # ensure precision formatting does not round up
            try:
                floored = float(self.exchange.amount_to_precision(symbol, floored))
            except Exception:
                pass
            return max(0.0, floored)
        except Exception:
            return 0.0

    # --- Safe market helper (handles PERCENT_PRICE errors on illiquid testnet) ---
    def create_market_order_safe(self, symbol: str, side: str, amount: float, attempts: int = 3):
        """
        Try market order; if PERCENT_PRICE/PERCENT_PRICE_BY_SIDE filter trips (common on testnet
        or thin books), progressively halve the amount and retry up to `attempts`.
        Returns last order on success. Raises last exception on failure.
        """
        last_err = None
        qty = float(amount)
        for i in range(max(1, attempts)):
            try:
                return self.exchange.create_market_order(symbol, side, qty)
            except Exception as e:
                msg = str(e)
                last_err = e
                # Handle risk-limit exceed (-2027). Try shrinking to remaining capacity once.
                if "-2027" in msg or "maximum allowable position" in msg.lower():
                    cap_qty = self.remaining_addable_qty_under_risk_limit(symbol)
                    if cap_qty > 0 and cap_qty < qty:
                        try:
                            return self.exchange.create_market_order(symbol, side, cap_qty)
                        except Exception as e3:
                            last_err = e3
                    # else bubble error
                if ("PERCENT_PRICE" in msg or "PERCENT_PRICE_BY_SIDE" in msg or "best price" in msg) and qty > 0:
                    # try limit IOC within percent bounds
                    try:
                        return self._create_limit_ioc_with_bounds(symbol, side, qty)
                    except Exception as e2:
                        # halve qty and retry
                        qty = qty / 2.0
                        continue
                raise
        # exhausted attempts
        raise last_err if last_err else Exception("create_market_order_safe failed")

    # --- Safe STOP_MARKET helper (avoid -2021 immediate trigger) ---
    def create_stop_market_safe(self, symbol: str, side: str, stop_price: float, amount: float, reduce_only: bool = True):
        try:
            sym = symbol.replace('/', '')
            info = self.exchange.fapiPublic_get_premiumindex({ 'symbol': sym })
            mark = float(info[0]['markPrice'] if isinstance(info, list) and info else info.get('markPrice', 0))
        except Exception:
            try:
                t = self.fetch_ticker(symbol)
                mark = float(t.get('last') or t.get('bid') or t.get('ask') or 0)
            except Exception:
                mark = 0.0
        sp = float(stop_price)
        nudge = 0.001
        if mark > 0:
            if side.lower() == 'sell':
                sp = min(sp, mark * (1 - nudge))
            else:
                sp = max(sp, mark * (1 + nudge))
        params = {
            "stopPrice": self.exchange.price_to_precision(symbol, sp),
            "workingType": "MARK_PRICE",
            "reduceOnly": reduce_only,
        }
        return self.exchange.create_order(symbol, "STOP_MARKET", side, amount, None, params)

    # --- Internals for PERCENT_PRICE bounds ---
    def _get_percent_bounds(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            m = self.exchange.market(symbol)
            filters = m.get('info', {}).get('filters', [])
            pp = next((f for f in filters if f.get('filterType') == 'PERCENT_PRICE'), None)
            if not pp:
                return None
            # Mark price is used for percent price filter
            sym = symbol.replace('/', '')
            info = self.exchange.fapiPublic_get_premiumindex({ 'symbol': sym })
            # Some ccxt versions return list; normalize
            if isinstance(info, list) and info:
                info = info[0]
            mark = float(info.get('markPrice'))
            up = float(pp.get('multiplierUp'))
            down = float(pp.get('multiplierDown'))
            return { 'mark': mark, 'max_buy': mark * up, 'min_sell': mark * down }
        except Exception:
            return None

    def _create_limit_ioc_with_bounds(self, symbol: str, side: str, amount: float):
        bounds = self._get_percent_bounds(symbol)
        t = self.fetch_ticker(symbol)
        bid = float(t.get('bid') or t.get('last') or 0)
        ask = float(t.get('ask') or t.get('last') or 0)
        price = None
        if bounds:
            if side == 'buy':
                # cannot cross above allowed max
                allowed = bounds['max_buy']
                if ask > 0:
                    price = min(ask, allowed)
                else:
                    price = allowed
            else:
                allowed = bounds['min_sell']
                if bid > 0:
                    price = max(bid, allowed)
                else:
                    price = allowed
        else:
            # fallback: best bid/ask
            price = ask if side == 'buy' else bid
        if price is None or price <= 0:
            # as a last resort, try market to bubble error
            return self.exchange.create_market_order(symbol, side, amount)
        px = float(self.exchange.price_to_precision(symbol, price))
        params = { 'timeInForce': 'IOC' }
        return self.exchange.create_order(symbol, 'LIMIT', side, amount, px, params)




