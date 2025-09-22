import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from broker_binance import BinanceUSDM

if __name__ == "__main__":
    b = BinanceUSDM()
    print("effective_time_diff_ms=", int(b.time_diff))
