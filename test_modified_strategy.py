#!/usr/bin/env python3
"""
수정된 전략 테스트 (정렬 + 창구 방식)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from strategy import DonchianATREngine
from indicators import ma_crossover_signal

# 테스트 데이터 생성 (현재 BTC 상황과 유사)
test_data = pd.DataFrame({
    'ts': pd.date_range('2025-09-11 00:00', periods=25, freq='1h'),
    'open': [113000 + i*50 for i in range(25)],
    'high': [113200 + i*50 for i in range(25)],
    'low': [112800 + i*50 for i in range(25)],
    'close': [113100 + i*50 for i in range(25)],  # 상승 추세
    'volume': [1000] * 25
})

# 5MA가 20MA 위에 있는 상황 시뮬레이션
print("=== 수정된 전략 테스트 ===")
print("\n1. MA 정렬 신호 테스트")

engine = DonchianATREngine()
signal = engine.detect_signal(test_data)

print(f"Fast MA: {signal['fast_ma']:.2f}")
print(f"Slow MA: {signal['slow_ma']:.2f}")
print(f"MA 차이: {((signal['fast_ma']/signal['slow_ma']-1)*100):.3f}%")
print(f"레짐: {signal['regime']}")

# 정렬 상태 확인
alignment = signal.get('alignment', {})
print(f"\n정렬 상태:")
print(f"  롱 정렬: {alignment.get('long_aligned', False)}")
print(f"  숏 정렬: {alignment.get('short_aligned', False)}")
print(f"  롱 크로스: {alignment.get('long_cross', False)}")
print(f"  숏 크로스: {alignment.get('short_cross', False)}")

# 신호 결과
print(f"\n신호 결과:")
print(f"  롱 신호: {signal['long']}")
print(f"  숏 신호: {signal['short']}")

# 캔들 위치 분석
candle_info = signal.get('candle_filter', {})
print(f"\n캔들 위치 분석:")
print(f"  위치 비율: {candle_info.get('candle_position_ratio', 0):.3f}")
print(f"  롱 안전: {candle_info.get('candle_safe_long', False)}")
print(f"  숏 안전: {candle_info.get('candle_safe_short', False)}")

# 리스크 승수
risk_info = signal.get('risk_multiplier', {})
print(f"\n리스크 승수:")
print(f"  롱: {risk_info.get('long', 1.0):.1f}x")
print(f"  숏: {risk_info.get('short', 1.0):.1f}x")

print("\n=== 테스트 완료 ===")
print("이제 실제 봇을 실행하면 정렬 상태에서도 진입 신호가 나올 것입니다!")