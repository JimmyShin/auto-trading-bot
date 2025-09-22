#!/usr/bin/env python3
"""
상세한 진입 로깅 시스템 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import log_detailed_entry

# 테스트 신호 데이터
test_sig = {
    'fast_ma': 113700.0,
    'slow_ma': 113000.0,
    'regime': 'UP'
}

print("=== 상세 진입 로깅 테스트 ===")

# 테스트 1: 롱 진입 (풀 리스크)
print("\n1. 롱 진입 (풀 리스크) 테스트")
log_detailed_entry(
    symbol="BTC/USDT",
    side="LONG", 
    entry_price=114000.0,
    qty=0.12,
    stop_price=112800.0,
    risk_multiplier=1.0,
    sig=test_sig,
    atr_abs=800.0,
    equity=7000.0,
    reason="MA_ALIGNMENT"
)
print("✅ 롱 진입 로그 완료")

# 테스트 2: 숏 진입 (리스크 절반)
print("\n2. 숏 진입 (리스크 절반) 테스트") 
log_detailed_entry(
    symbol="ETH/USDT",
    side="SHORT",
    entry_price=4300.0,
    qty=1.5,
    stop_price=4350.0,
    risk_multiplier=0.5,
    sig={'fast_ma': 4290.0, 'slow_ma': 4310.0, 'regime': 'DOWN'},
    atr_abs=45.0,
    equity=7000.0,
    reason="MA_ALIGNMENT"
)
print("✅ 숏 진입 로그 완료")

# 테스트 3: 피라미딩 진입
print("\n3. 피라미딩 진입 테스트")
log_detailed_entry(
    symbol="SOL/USDT",
    side="LONG_ADD",
    entry_price=225.0,
    qty=30.0,
    stop_price=0,  # 피라미딩은 기존 스톱 사용
    risk_multiplier=0.5,
    sig={'fast_ma': 224.0, 'slow_ma': 222.0, 'regime': 'UP'},
    atr_abs=5.5,
    equity=7000.0,
    reason="PYRAMID_L1"
)
print("✅ 피라미딩 로그 완료")

print("\n=== 테스트 완료 ===")
print("detailed_entries_*.csv 파일을 확인해보세요!")

# 생성된 파일 확인
import glob
csv_files = glob.glob("detailed_entries_*.csv")
if csv_files:
    print(f"생성된 파일: {csv_files[0]}")
    
    # 파일 내용 일부 출력
    try:
        with open(csv_files[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("\n파일 내용 (처음 2줄):")
            for i, line in enumerate(lines[:2]):
                print(f"  {i+1}: {line.strip()}")
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
else:
    print("CSV 파일이 생성되지 않았습니다.")