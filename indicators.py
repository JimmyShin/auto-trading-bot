import pandas as pd
from datetime import datetime

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n=14) -> float:
    h, l, c = df['high'], df['low'], df['close']
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def regime_label(df: pd.DataFrame) -> str:
    """EMA50/200 + 기울기 간단 점수화 (일봉 대체 라이트 버전)
       운영 안정화를 원하면 일봉 데이터로 교체 권장"""
    close = df['close']
    if len(close) < 220:  # 200EMA + 여유
        return "RANGE"
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    slope50 = (ema50.iloc[-1] / ema50.iloc[-20] - 1) * 100 if len(ema50) > 20 else 0.0

    score = 0
    if ema50.iloc[-1] > ema200.iloc[-1]: score += 1
    if ema50.iloc[-1] < ema200.iloc[-1]: score -= 1
    if slope50 > +0.5: score += 1
    if slope50 < -0.5: score -= 1

    if score >= 1: return "UP"
    if score <= -1: return "DOWN"
    return "RANGE"

def is_near_funding(now_utc: datetime, minutes=10) -> bool:
    """기본 펀딩 정산 시각(UTC 00/08/16) 전후 minutes분 회피"""
    hh = now_utc.hour
    mm = now_utc.minute
    targets = [(0,0),(8,0),(16,0)]
    return any(abs((hh*60+mm) - (H*60+M)) < minutes for H,M in targets)


def sma(df: pd.DataFrame, period: int) -> pd.Series:
    """단순 이동평균"""
    return df['close'].rolling(period).mean()

def ma_crossover_signal(df: pd.DataFrame, fast_period=5, slow_period=20):
    """이동평균 정렬 + 창구 시그널 (레짐 필터 적용)
    - 정렬 상태: fast MA > slow MA (롱), fast MA < slow MA (숏)
    - 24봉 창구: 정렬 상태에서 계속 진입 기회 제공
    - 레짐 필터: 완전 역추세만 차단
    """
    fast_ma = sma(df, fast_period)
    slow_ma = sma(df, slow_period)
    
    # 현재와 이전 값 비교
    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
    prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow
    
    # 기존 크로스오버 확인 (참고용)
    long_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
    short_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)
    
    # 새로운 정렬 기반 신호
    long_aligned = current_fast > current_slow
    short_aligned = current_fast < current_slow
    
    # 레짐 필터 적용
    regime = regime_label(df)
    
    # 완전 역추세만 방지, RANGE 적극 허용
    long_signal = long_aligned and (regime != "DOWN")  # DOWN만 차단
    short_signal = short_aligned and (regime != "UP")   # UP만 차단
    
    return {
        "long": long_signal,
        "short": short_signal,
        "fast_ma": current_fast,
        "slow_ma": current_slow,
        "regime": regime,
        "alignment": {
            "long_aligned": long_aligned,
            "short_aligned": short_aligned,
            "long_cross": long_cross,
            "short_cross": short_cross
        }
    }
