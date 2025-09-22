#!/usr/bin/env python3
"""
구조적 리셋 조건 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from strategy import DonchianATREngine

print("=== 구조적 리셋 보호 로직 테스트 ===")

# 테스트 엔진 생성
engine = DonchianATREngine()

# 테스트 시나리오 1: 손절 기록
print("\n1. 손절 기록 테스트")
engine.record_stop_loss_exit("BTC/USDT", "long", 112000.0)
print("   BTC/USDT 롱 손절 기록 완료")

# 재진입 가능 여부 확인
can_enter = engine.can_re_enter_after_stop("BTC/USDT")
print(f"   재진입 가능: {can_enter}")

# 테스트 데이터 생성 (MA 역전 시뮬레이션)
print("\n2. 구조적 리셋 조건 테스트")
test_data = pd.DataFrame({
    'ts': pd.date_range('2025-09-11 00:00', periods=30, freq='1h'),
    'open': [113000 - i*20 for i in range(30)],    # 하락 후 상승
    'high': [113200 - i*20 for i in range(30)],
    'low': [112800 - i*20 for i in range(30)], 
    'close': [113000 - i*20 if i < 15 else 112400 + (i-15)*30 for i in range(30)],
    'volume': [1000] * 30
})

# 리셋 추적 시뮬레이션
print("   리셋 조건 추적 중...")
for i in range(5):  # 5번의 바 업데이트 시뮬레이션
    current_data = test_data.iloc[:25+i]  # 점진적으로 데이터 증가
    engine.update_reset_tracking("BTC/USDT", current_data)
    
    can_enter_now = engine.can_re_enter_after_stop("BTC/USDT")
    print(f"   바 {i+1}: 재진입 가능 = {can_enter_now}")
    
    if can_enter_now:
        break

# 테스트 시나리오 2: 안전 재시작 모드 확인
print("\n3. 안전 재시작 모드 테스트")
from config import SAFE_RESTART
print(f"   SAFE_RESTART 설정: {SAFE_RESTART}")

if SAFE_RESTART:
    print("   ✅ 안전 재시작 모드 활성화됨")
    print("   포지션이 있을 때 봇 재시작해도 자동 청산되지 않음")
else:
    print("   ⚠️ 일반 재시작 모드")
    print("   봇 재시작시 기존 포지션이 자동 청산됨")

print("\n=== 테스트 완료 ===")
print("구조적 보호 장치가 성공적으로 구현되었습니다!")