import json
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional

from config import (ATR_LEN, ATR_STOP_K,
                    RISK_PCT, STATE_FILE,
                    CANDLE_LONG_MAX_POS_RATIO, CANDLE_SHORT_MIN_POS_RATIO,
                    OVERSIZE_TOLERANCE)
from indicators import atr
from typing import Tuple

from risk import RiskManager

def load_state(path: Optional[str] = None) -> Dict:
    """Load persisted engine state from disk."""
    state_path = path or STATE_FILE
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state: Dict, path: Optional[str] = None):
    """Atomically persist engine state."""
    state_path = path or STATE_FILE
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    # Atomic replace with Windows-friendly retry (handles transient locks)
    import time, shutil
    last_err = None
    for i in range(5):
        try:
            os.replace(tmp_path, state_path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.1 * (i + 1))
        except Exception as e:
            last_err = e
            break
    # Fallback to move if replace kept failing
    try:
        shutil.move(tmp_path, state_path)
    except Exception:
        if last_err:
            raise last_err

class DonchianATREngine:
    """
    15분봉 MA크로스 전략
    - 5/20MA 크로스오버 신호
    - ATR 기반 사이징/손절/트레일
    """
    def __init__(self, state_path: Optional[str] = None, persist_state: bool = True, initial_state: Optional[Dict] = None, risk_manager: Optional[RiskManager] = None):
        self.state_path = state_path or STATE_FILE
        self._persist_state = persist_state
        if initial_state is not None:
            self.state = initial_state
        elif self._persist_state:
            self.state = load_state(self.state_path)  # {symbol: {...}}, {daily: {...}}
        else:
            self.state = {}
        self._risk = risk_manager or RiskManager(self.state)
        self._risk.bind_state(self.state)

    def _save_state(self):
        if self._persist_state:
            save_state(self.state, self.state_path)

    def reset_daily_anchor(self, equity: float):
        if self._risk.reset_daily_anchor(equity):
            self._save_state()

    def hit_daily_loss_limit(self, equity: float) -> bool:
        return self._risk.hit_daily_loss_limit(equity)

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
        """Return the base position size using configured risk parameters."""
        return self._risk.calc_qty_by_risk(equity_usdt, price, atr_abs, leverage, symbol)

    def calc_qty_by_risk_adjusted(self, equity_usdt: float, price: float, atr_abs: float, leverage: int, symbol: str = "", risk_pct: float = None) -> float:
        """Return the position size using an adjusted risk percentage."""
        return self._risk.calc_qty_by_risk_adjusted(equity_usdt, price, atr_abs, leverage, symbol, risk_pct)

    def trail_stop_price(self, side: str, entry_price: float, last_price: float, atr_abs: float, be_promoted: bool) -> float:
        """Return the ATR-based trailing stop for the active position."""
        return self._risk.trail_stop_price(side, entry_price, last_price, atr_abs, be_promoted)

    def update_symbol_state_on_entry(self, symbol: str, side: str, entry_px: float, qty: float = 0, entry_stop_price: float | None = None, risk_usdt: float = 0.0):
        state = self._risk.update_symbol_state_on_entry(symbol, side, entry_px, qty, entry_stop_price, risk_usdt)
        self._save_state()
        return state

    def update_after_move(self, symbol: str, atr_abs: float, last_price: float):
        state = self._risk.update_after_move(symbol, atr_abs, last_price)
        if state is not None:
            self._save_state()
        return state

    def calc_locked_profit_pyramid_limit(self, symbol: str, atr_abs: float, current_price: float, add_ratio: float) -> Dict[str, float]:
        """Expose the risk manager calculation for pyramid sizing limits."""
        return self._risk.calc_locked_profit_pyramid_limit(symbol, atr_abs, current_price, add_ratio)

    def clear_position_state(self, symbol: str):
        state = self._risk.clear_position_state(symbol)
        self._save_state()
        return state

    def record_stop_loss_exit(self, symbol: str, side: str, exit_price: float):
        state = self._risk.record_stop_loss_exit(symbol, side, exit_price)
        self._save_state()
        return state

    def update_reset_tracking(self, symbol: str, df: pd.DataFrame):
        updated = self._risk.update_reset_tracking(symbol, df)
        if updated:
            self._save_state()
        return updated

    def check_reset_conditions(self, symbol: str, df: pd.DataFrame, last_stop: dict) -> dict:
        return self._risk.check_reset_conditions(symbol, df, last_stop)

    def check_cooling_period(self, df: pd.DataFrame, original_side: str, bars_back: int) -> bool:
        return self._risk.check_cooling_period(df, original_side, bars_back)

    def check_ma_retest(self, df: pd.DataFrame, side: str) -> bool:
        return self._risk.check_ma_retest(df, side)

    def can_re_enter_after_stop(self, symbol: str) -> bool:
        return self._risk.can_re_enter_after_stop(symbol)

    def sync_position_state(self, broker, symbol: str):
        result = self._risk.sync_position_state(broker, symbol)
        if result:
            self._save_state()
        return result

