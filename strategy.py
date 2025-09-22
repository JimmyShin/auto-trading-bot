import json
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional

from config import (ATR_LEN, ATR_STOP_K, ATR_TRAIL_K,
                    RISK_PCT, DAILY_LOSS_LIMIT, STATE_FILE,
                    CANDLE_LONG_MAX_POS_RATIO, CANDLE_SHORT_MIN_POS_RATIO,
                    OVERSIZE_TOLERANCE)
from indicators import atr
from typing import Tuple

def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state: Dict):
    tmp_path = f"{STATE_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    # Atomic replace with Windows-friendly retry (handles transient locks)
    import time, shutil
    last_err = None
    for i in range(5):
        try:
            os.replace(tmp_path, STATE_FILE)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.1 * (i + 1))
        except Exception as e:
            last_err = e
            break
    # Fallback to move if replace kept failing
    try:
        shutil.move(tmp_path, STATE_FILE)
    except Exception:
        if last_err:
            raise last_err

class DonchianATREngine:
    """
    15분봉 MA크로스 전략
    - 5/20MA 크로스오버 신호
    - ATR 기반 사이징/손절/트레일
    """
    def __init__(self):
        self.state = load_state()  # {symbol: {...}}, {daily: {...}}

    def reset_daily_anchor(self, equity: float):
        today = datetime.utcnow().date().isoformat()
        if self.state.get("daily", {}).get("date") != today:
            self.state["daily"] = {"date": today, "anchor": float(equity)}
            save_state(self.state)

    def hit_daily_loss_limit(self, equity: float) -> bool:
        daily = self.state.get("daily")
        if not daily:
            return False
        anchor = float(daily.get("anchor", equity))
        dd = max(0.0, (anchor - equity) / max(anchor, 1e-9))
        return dd >= DAILY_LOSS_LIMIT

    def detect_signal(self, df: pd.DataFrame) -> Dict:
        """이동평균 정렬 + 창구 시그널 + 캔들 위치 리스크 조정"""
        from indicators import ma_crossover_signal
        signal = ma_crossover_signal(df, fast_period=5, slow_period=20)
        
        # 캔들 위치 분석
        candle_analysis = self.get_candle_position_analysis(df)
        candle_safe_long = self.is_safe_candle_position(df, "long")
        candle_safe_short = self.is_safe_candle_position(df, "short")
        
        # 신호 결정 (정렬 기반)
        base_long = signal["long"]  # 이제 정렬 기반
        base_short = signal["short"]  # 이제 정렬 기반
        
        # 리스크 승수 계산 (캔들 위치 기반)
        long_risk_multiplier = 1.0 if candle_safe_long else 0.5
        short_risk_multiplier = 1.0 if candle_safe_short else 0.5
        
        return {
            "long": base_long,
            "short": base_short,
            "fast_ma": signal["fast_ma"],
            "slow_ma": signal["slow_ma"],
            "regime": signal["regime"],
            "alignment": signal["alignment"],  # 정렬 상태 정보 추가
            "risk_multiplier": {
                "long": long_risk_multiplier,
                "short": short_risk_multiplier
            },
            "candle_filter": {
                "original_long": base_long,
                "original_short": base_short,
                "candle_safe_long": candle_safe_long,
                "candle_safe_short": candle_safe_short,
                "candle_position_ratio": candle_analysis["position_ratio"],
                "candle_range": candle_analysis["range"],
                "filter_reason": candle_analysis["filter_reason"]
            }
        }

    def make_entry_plan(
        self,
        symbol: str,
        df: pd.DataFrame,
        equity: float,
        price: float,
        atr_abs: float,
        is_new_bar: bool,
        can_reenter: bool,
        funding_avoid: bool,
        daily_loss_hit: bool,
    ) -> Dict:
        """Return a unified entry decision with rationale and sizing.

        Decision format:
        {
          'decision': 'ENTER_LONG'|'ENTER_SHORT'|'SKIP',
          'skip_reason': str|None,
          'side': 'long'|'short'|None,
          'qty': float,
          'stop_price': float,
          'risk_multiplier': float,
          'reasons': [str,...],
          'signal': {...},
        }
        """
        reasons = []
        out: Dict = {
            'decision': 'SKIP',
            'skip_reason': None,
            'side': None,
            'qty': 0.0,
            'stop_price': 0.0,
            'risk_multiplier': 1.0,
        }

        sig = self.detect_signal(df)
        out['signal'] = sig

        if daily_loss_hit:
            out['skip_reason'] = 'daily_loss_limit'
            reasons.append('daily_loss_limit')
            out['reasons'] = reasons
            return out
        if not is_new_bar and (sig.get('long') or sig.get('short')):
            out['skip_reason'] = 'not_new_bar'
            reasons.append('not_new_bar')
            out['reasons'] = reasons
            return out
        if not can_reenter:
            out['skip_reason'] = 'structural_reset_pending'
            reasons.append('structural_reset_pending')
            out['reasons'] = reasons
            return out
        if funding_avoid:
            out['skip_reason'] = 'funding_avoid'
            reasons.append('funding_avoid')
            out['reasons'] = reasons
            return out

        # Choose side by signal
        side: Optional[str] = None
        if sig.get('long'):
            side = 'long'
        elif sig.get('short'):
            side = 'short'
        else:
            out['skip_reason'] = 'no_signal'
            out['reasons'] = reasons + ['no_signal']
            return out

        candle_filter = sig.get('candle_filter', {})
        safe_long = bool(candle_filter.get('candle_safe_long'))
        safe_short = bool(candle_filter.get('candle_safe_short'))
        if side == 'long' and not safe_long:
            out['skip_reason'] = 'candle_position_filter'
            out['reasons'] = reasons + ['candle_position_filter']
            return out
        if side == 'short' and not safe_short:
            out['skip_reason'] = 'candle_position_filter'
            out['reasons'] = reasons + ['candle_position_filter']
            return out

        # Sizing with risk multiplier
        risk_mult = float(sig.get('risk_multiplier', {}).get('long' if side == 'long' else 'short', 1.0))
        out['risk_multiplier'] = risk_mult

        from config import LEVERAGE, RISK_PCT, ATR_STOP_K
        adjusted_risk_pct = RISK_PCT * risk_mult
        qty = self.calc_qty_by_risk_adjusted(equity, price, atr_abs, LEVERAGE, symbol, adjusted_risk_pct)
        stop_price = price - max(ATR_STOP_K * atr_abs, price * 0.004) if side == 'long' else price + max(ATR_STOP_K * atr_abs, price * 0.004)
        risk_amount = abs(qty * (price - stop_price))

        # Guards
        if qty <= 0:
            out['skip_reason'] = 'insufficient_qty'
            out['reasons'] = reasons + ['insufficient_qty']
            return out

        # Oversize guard: tolerance via config
        if risk_amount > equity * RISK_PCT * float(OVERSIZE_TOLERANCE):
            out['skip_reason'] = 'oversize_risk'
            out['reasons'] = reasons + ['oversize_risk']
            return out

        out['side'] = side
        out['qty'] = qty
        out['stop_price'] = stop_price
        out['decision'] = 'ENTER_LONG' if side == 'long' else 'ENTER_SHORT'
        out['reasons'] = reasons + ['trend_alignment', 'candle_safe', f"risk_mult_{risk_mult:.2f}"]
        return out
    
    def is_safe_candle_position(self, df: pd.DataFrame, direction: str) -> bool:
        """캔들 내 위치 기반 안전 진입 확인"""
        if len(df) < 1:
            return False
            
        candle = df.iloc[-1]
        current_price = float(candle['close'])
        candle_high = float(candle['high'])  
        candle_low = float(candle['low'])
        candle_range = candle_high - candle_low
        
        # 캔들 범위가 너무 작으면 통과 (안전)
        if candle_range == 0:
            return True
            
        # 현재가의 캔들 내 위치 계산 (0=저가, 1=고가)
        position_ratio = (current_price - candle_low) / candle_range
        
        if direction == "long":
            # 롱: 캔들 하위 70% 구간에서만 진입 허용 (상위 30% 금지)
            return position_ratio < float(CANDLE_LONG_MAX_POS_RATIO)
        else:  # short
            # 숏: 캔들 상위 70% 구간에서만 진입 허용 (하위 30% 금지)  
            return position_ratio > float(CANDLE_SHORT_MIN_POS_RATIO)
    
    def get_candle_position_analysis(self, df: pd.DataFrame) -> dict:
        """캔들 위치 분석 정보 반환"""
        if len(df) < 1:
            return {"position_ratio": 0, "range": 0, "filter_reason": "no_data"}
            
        candle = df.iloc[-1]
        current_price = float(candle['close'])
        candle_high = float(candle['high'])
        candle_low = float(candle['low'])
        candle_range = candle_high - candle_low
        
        if candle_range == 0:
            return {"position_ratio": 0.5, "range": 0, "filter_reason": "zero_range"}
            
        position_ratio = (current_price - candle_low) / candle_range
        
        # 필터링 이유 분석
        filter_reason = "safe"
        if position_ratio > 0.7:
            filter_reason = f"too_high_{position_ratio:.2f}"
        elif position_ratio < 0.3:
            filter_reason = f"too_low_{position_ratio:.2f}"
            
        return {
            "position_ratio": round(position_ratio, 3),
            "range": round(candle_range, 2),
            "filter_reason": filter_reason
        }


    def calc_qty_by_risk(self, equity_usdt: float, price: float, atr_abs: float, leverage: int, symbol: str = "") -> float:
        """기본 리스크 기반 수량 계산 (기존 로직 유지)"""
        return self.calc_qty_by_risk_adjusted(equity_usdt, price, atr_abs, leverage, symbol, RISK_PCT)
    
    def calc_qty_by_risk_adjusted(self, equity_usdt: float, price: float, atr_abs: float, leverage: int, symbol: str = "", risk_pct: float = None) -> float:
        """조정된 리스크 기반 수량 계산 (캔들 위치 승수 적용)"""
        if risk_pct is None:
            risk_pct = RISK_PCT
            
        risk_quote = equity_usdt * risk_pct
        stop_dist = max(ATR_STOP_K * atr_abs, price * 0.004)  # 최소 0.4% (휩쏘 방지)
        qty = (risk_quote / stop_dist)
        
        # 최소 주문 크기 보장 (바이낸스 선물 기준)
        min_qty = 0.001  # 기본 최소값
        min_notional = 20.0  # 최소 주문 금액 $20
        min_qty_by_notional = min_notional / price
        
        # 심볼별 최소 수량
        if 'ETH' in symbol:
            min_qty = 0.01   # ETH 최소 0.01
        elif 'SOL' in symbol:
            min_qty = 0.01   # SOL 최소 0.01
        elif 'BNB' in symbol:
            min_qty = 0.01   # BNB 최소 0.01
        elif 'AVAX' in symbol:
            min_qty = 0.1    # AVAX 최소 0.1
        elif 'ADA' in symbol:
            min_qty = 1.0    # ADA 최소 1.0
            
        final_min_qty = max(min_qty, min_qty_by_notional)
        
        if qty > 0 and qty < final_min_qty:
            print(f"WARNING calculated qty {qty:.6f} < min order {final_min_qty:.6f} -> adjusted")
            qty = final_min_qty
            
        # 포지션 명목가 상한 적용 (동적)
        from config import get_position_cap
        position_cap = get_position_cap(equity_usdt)
        max_qty_by_cap = position_cap / price
        
        if qty > max_qty_by_cap:
            print(f"WARNING position too large: ${qty * price:.0f} > ${position_cap:.0f} -> capped")
            qty = max_qty_by_cap
            
        return max(qty, 0.0)

    def trail_stop_price(self, side: str, entry_price: float, last_price: float, atr_abs: float, be_promoted: bool) -> float:
        """BE 승격 후 2*ATR 트레일 (기본)"""
        if side == "long":
            base = last_price - ATR_TRAIL_K * atr_abs
            be = entry_price  # 본전
            return max(base, be if be_promoted else -1e18)
        else:
            base = last_price + ATR_TRAIL_K * atr_abs
            be = entry_price
            return min(base, be if be_promoted else 1e18)

    def update_symbol_state_on_entry(self, symbol: str, side: str, entry_px: float, qty: float = 0):
        st = self.state.get(symbol, {})
        st.update({
            "in_position": True,
            "side": side,                       # "long"/"short"
            "entry_price": float(entry_px),
            "entry_time": datetime.utcnow().isoformat(),
            "be_promoted": False,               # 본전 승격
            "original_qty": float(qty),         # 최초 포지션 크기
            "pyramid_level": 0,                 # 피라미딩 단계
            "pyramid_added": [],                # 추가된 수량들 기록
        })
        self.state[symbol] = st
        save_state(self.state)

    def update_after_move(self, symbol: str, atr_abs: float, last_price: float):
        st = self.state.get(symbol)
        if not st or not st.get("in_position"): return None
        side = st["side"]
        entry = float(st["entry_price"])

        # 현재 수익률 계산 (R 단위)
        if side == "long":
            profit_r = (last_price - entry) / (atr_abs * 1.0)  # ATR_STOP_K 기준
            if last_price - entry >= atr_abs * 1.5:
                st["be_promoted"] = True
        else:
            profit_r = (entry - last_price) / (atr_abs * 1.0)
            if entry - last_price >= atr_abs * 1.5:
                st["be_promoted"] = True

        # 피라미딩 체크
        from config import ENABLE_PYRAMIDING, PYRAMID_LEVELS
        if ENABLE_PYRAMIDING:
            current_level = st.get("pyramid_level", 0)
            if current_level < len(PYRAMID_LEVELS):
                target_r, add_ratio = PYRAMID_LEVELS[current_level]
                if profit_r >= target_r:
                    st["pyramid_level"] = current_level + 1
                    st["pyramid_trigger_r"] = profit_r  # 피라미딩 트리거 R 기록
                    
                    # 잠긴 이익 기반 수량 제한 계산
                    locked_profit_limit = self.calc_locked_profit_pyramid_limit(symbol, atr_abs, last_price, add_ratio)
                    st["pyramid_locked_limit"] = locked_profit_limit
                    
                    print(f"🔺 {symbol} 피라미딩 트리거: +{profit_r:.1f}R 달성")

        self.state[symbol] = st
        save_state(self.state)
        return st
    
    def calc_locked_profit_pyramid_limit(self, symbol: str, atr_abs: float, current_price: float, add_ratio: float) -> float:
        """잠긴 이익 기반 피라미딩 수량 제한 계산"""
        st = self.state.get(symbol, {})
        side = st.get("side")
        entry_price = float(st.get("entry_price", current_price))
        original_qty = float(st.get("original_qty", 0))
        be_promoted = st.get("be_promoted", False)
        
        # 현재 트레일 스톱 가격 계산
        trail_stop = self.trail_stop_price(side, entry_price, current_price, atr_abs, be_promoted)
        
        # 잠긴 이익 계산 (트레일 스톱 기준)
        if side == "long":
            locked_profit_per_share = max(0, trail_stop - entry_price)
        else:
            locked_profit_per_share = max(0, entry_price - trail_stop)
        
        total_locked_profit = locked_profit_per_share * original_qty
        
        # 기존 방식: 원금 기준 추가 수량
        traditional_add_qty = original_qty * add_ratio
        
        # 잠긴 이익의 50%로 추가 수량 제한 (급반전 대비)
        if total_locked_profit > 0:
            max_giveback = total_locked_profit * 0.5  # 잠긴 이익의 50%만 리스크
            safe_add_qty = max_giveback / current_price
        else:
            safe_add_qty = 0  # 잠긴 이익이 없으면 추가 불가
        
        # 둘 중 작은 값 반환
        final_qty = min(traditional_add_qty, safe_add_qty)
        
        return {
            "locked_profit": total_locked_profit,
            "traditional_qty": traditional_add_qty,  
            "safe_qty": safe_add_qty,
            "final_qty": final_qty
        }

    def clear_position_state(self, symbol: str):
        st = self.state.get(symbol, {})
        st.update({"in_position": False})
        self.state[symbol] = st
        save_state(self.state)
    
    def record_stop_loss_exit(self, symbol: str, side: str, exit_price: float):
        """손절 청산 기록 - 구조적 리셋 조건을 위해"""
        st = self.state.get(symbol, {})
        st.update({
            "in_position": False,
            "last_stop_loss": {
                "timestamp": datetime.utcnow().isoformat(),
                "side": side,
                "exit_price": float(exit_price),
                "bars_since_stop": 0,  # 손절 후 경과 봉 수
                "reset_qualified": False  # 구조적 리셋 완료 여부
            }
        })
        self.state[symbol] = st
        save_state(self.state)
    
    def update_reset_tracking(self, symbol: str, df: pd.DataFrame):
        """구조적 리셋 조건 추적 업데이트"""
        st = self.state.get(symbol, {})
        last_stop = st.get("last_stop_loss")
        
        if not last_stop or last_stop.get("reset_qualified"):
            return  # 손절 기록 없거나 이미 리셋 완료
            
        # 경과 봉 수 증가
        last_stop["bars_since_stop"] += 1
        
        # MA 값 계산
        from indicators import sma
        fast_ma = sma(df, 5).iloc[-1]
        slow_ma = sma(df, 20).iloc[-1]
        
        # 리셋 조건 확인
        reset_conditions = self.check_reset_conditions(symbol, df, last_stop)
        
        if reset_conditions["qualified"]:
            last_stop["reset_qualified"] = True
            print(f"RESET: {symbol} structural reset completed - re-entry allowed")
            print(f"   conditions: {', '.join(reset_conditions['reasons'])}")
            
        st["last_stop_loss"] = last_stop
        self.state[symbol] = st
        save_state(self.state)
    
    def check_reset_conditions(self, symbol: str, df: pd.DataFrame, last_stop: dict) -> dict:
        """구조적 리셋 조건 검사"""
        from indicators import sma
        
        if len(df) < 25:
            return {"qualified": False, "reasons": []}
            
        fast_ma = sma(df, 5)
        slow_ma = sma(df, 20)
        
        # 현재 및 이전 값들
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        
        reasons = []
        side = last_stop.get("side", "")
        bars_since = last_stop.get("bars_since_stop", 0)
        
        # 조건 1: 최소 경과 시간 (2봉 이상)
        min_bars_passed = bars_since >= 2
        if min_bars_passed:
            reasons.append(f"min_bars({bars_since})")
            
        # 조건 2: 구조적 역시그널 냉각 확인
        if side == "long":
            # 롱 손절 후: 5MA가 20MA 아래로 내려간 적이 있는지
            cooling_occurred = self.check_cooling_period(df, "long", bars_since)
            new_cross_up = (prev_fast <= prev_slow) and (current_fast > current_slow)
        else:
            # 숏 손절 후: 5MA가 20MA 위로 올라간 적이 있는지  
            cooling_occurred = self.check_cooling_period(df, "short", bars_since)
            new_cross_up = (prev_fast >= prev_slow) and (current_fast < current_slow)
            
        if cooling_occurred:
            reasons.append("cooling_done")
        if new_cross_up:
            reasons.append("new_cross")
            
        # 조건 3: 20MA 리테스트 후 회복 (선택적)
        retest_ok = self.check_ma_retest(df, side)
        if retest_ok:
            reasons.append("retest_ok")
            
        # 종합 판정: 최소 조건 + (냉각 또는 새크로스) + 추가 조건
        qualified = (min_bars_passed and 
                    (cooling_occurred or new_cross_up) and
                    len(reasons) >= 2)  # 최소 2개 조건 만족
        
        return {"qualified": qualified, "reasons": reasons}
    
    def check_cooling_period(self, df: pd.DataFrame, original_side: str, bars_back: int) -> bool:
        """냉각기간 동안 역시그널이 있었는지 확인"""
        from indicators import sma
        
        if len(df) < bars_back + 5:
            return False
            
        # 최근 N봉 동안의 MA 관계 확인
        fast_ma = sma(df, 5)
        slow_ma = sma(df, 20)
        
        # 손절 후 최근 봉들에서 역관계가 있었는지
        for i in range(min(bars_back, 5)):  # 최대 5봉까지만 확인
            idx = -(i+1)
            fast_val = fast_ma.iloc[idx]
            slow_val = slow_ma.iloc[idx]
            
            if original_side == "long" and fast_val < slow_val:
                return True  # 롱 손절 후 5MA < 20MA 확인
            elif original_side == "short" and fast_val > slow_val:
                return True  # 숏 손절 후 5MA > 20MA 확인
                
        return False
    
    def check_ma_retest(self, df: pd.DataFrame, side: str) -> bool:
        """20MA 리테스트 후 회복 확인 (선택적 조건)"""
        if len(df) < 5:
            return True  # 데이터 부족시 통과
            
        # 최근 3봉에서 20MA 근처 터치 후 회복 패턴
        from indicators import sma
        ma20 = sma(df, 20)
        closes = df['close']
        
        for i in range(1, min(4, len(df))):
            close_val = closes.iloc[-i]
            ma_val = ma20.iloc[-i]
            distance_pct = abs(close_val - ma_val) / ma_val
            
            if distance_pct < 0.005:  # 0.5% 이내 접근 = 리테스트
                return True
                
        return True  # 리테스트 없어도 일단 허용
    
    def can_re_enter_after_stop(self, symbol: str) -> bool:
        """손절 후 재진입 가능 여부 확인"""
        st = self.state.get(symbol, {})
        last_stop = st.get("last_stop_loss")
        
        if not last_stop:
            return True  # 손절 기록 없으면 진입 허용
            
        return last_stop.get("reset_qualified", False)
    
    def sync_position_state(self, broker, symbol: str):
        """API 포지션과 state.json 강제 동기화"""
        try:
            # 실제 API 포지션 확인
            actual_size = broker.get_actual_position_size(symbol)
            has_actual_position = abs(actual_size) > 1e-8
            
            # state.json 상태 확인  
            st = self.state.get(symbol, {})
            state_has_position = st.get("in_position", False)
            
            # 동기화 필요 여부 판단
            if has_actual_position and not state_has_position:
                # API에 포지션 있는데 state엔 없음 → state 업데이트
                print(f"🔄 {symbol} 포지션 상태 동기화: API에서 포지션 발견")
                st.update({
                    "in_position": True,
                    "side": "long" if actual_size > 0 else "short",
                    "original_qty": abs(actual_size)
                })
                self.state[symbol] = st
                save_state(self.state)
                return True
                
            elif not has_actual_position and state_has_position:
                # API에 포지션 없는데 state엔 있음 → 실제 청산됨
                print(f"🔔 {symbol} 실제 청산 확인됨 (API 포지션 없음)")
                return False  # 청산 처리 필요
                
            return None  # 동기화 불필요
            
        except Exception as e:
            print(f"⚠️ {symbol} 포지션 동기화 실패: {e}")
            return None
