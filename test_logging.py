#!/usr/bin/env python3
"""
신호 분석 로깅 시스템 테스트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import log_signal_analysis, log_trade, log_filtered_signal

# 테스트 데이터
test_signal_data = {
    'long': False,
    'short': False,
    'fast_ma': 95850.5,
    'slow_ma': 95200.2,
    'regime': 'RANGE',
    'candle_filter': {
        'original_long': True,
        'original_short': False,
        'position_safe': False,
        'candle_position_ratio': 0.85,
        'candle_range': 1200.5,
        'filter_reason': 'too_high_0.85'
    }
}

test_analysis_data = {
    'is_new_bar': True,
    'funding_avoid': False,
    'daily_loss_hit': False,
    'equity_usdt': 3622.01,
    'daily_dd_pct': 2.5
}

print("Signal analysis logging system test started...")

# 테스트 1: 캔들 위치 필터로 인한 스킵
print("\n1. Candle position filter skip test")
log_signal_analysis("BTC/USDT", 96000.0, test_signal_data, test_analysis_data, "SKIP", "candle_position_filter")

# Test 2: No signal
print("2. No signal test")
no_signal_data = test_signal_data.copy()
no_signal_data['candle_filter']['original_long'] = False
log_signal_analysis("ETH/USDT", 3900.0, no_signal_data, test_analysis_data, "SKIP", "no_signal")

# Test 3: Daily loss limit hit
print("3. Daily loss limit test")
daily_loss_data = test_analysis_data.copy()
daily_loss_data['daily_loss_hit'] = True
daily_loss_data['daily_dd_pct'] = 6.2
log_signal_analysis("SOL/USDT", 180.0, test_signal_data, daily_loss_data, "SKIP", "daily_loss_limit")

# Test 4: Successful long entry
print("4. Successful long entry test")
long_signal_data = test_signal_data.copy()
long_signal_data['long'] = True
long_signal_data['candle_filter']['position_safe'] = True
long_signal_data['candle_filter']['candle_position_ratio'] = 0.3
log_signal_analysis("BNB/USDT", 650.0, long_signal_data, test_analysis_data, "ENTER_LONG", None)

print("\nTest completed! Check CSV files.")
print("Generated files:")
import glob
csv_files = glob.glob("signal_analysis_*.csv")
for f in csv_files:
    print(f"  - {f}")