import os
from dotenv import load_dotenv

load_dotenv()

# API / environment
BINANCE_KEY = os.getenv("BINANCE_KEY", "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
TESTNET = os.getenv("TESTNET", "True").lower() == "true"

# Restart behavior
SAFE_RESTART = os.getenv("SAFE_RESTART", "true").lower() == "true"

# Universe / quote
QUOTE = os.getenv("QUOTE", "USDT")
UNIVERSE = [s.strip() for s in os.getenv("UNIVERSE", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,AVAX/USDT").split(",")]

# Risk / leverage
RISK_PCT = 0.025  # 2.5%
LEVERAGE = 10     # 10x

# Position cap (notional)
def get_position_cap(equity: float) -> float:
    dynamic_cap = equity * 2.0
    min_cap = 10000
    max_cap = 25000
    return min(max(dynamic_cap, min_cap), max_cap)

# Pyramiding
ENABLE_PYRAMIDING = True
PYRAMID_LEVELS = [
    (2.0, 0.3),  # +2.0R add 30%
    (3.0, 0.2),  # +3.0R add 20%
]
PYRAMID_COOLDOWN_BARS = int(os.getenv("PYRAMID_COOLDOWN_BARS", "1"))

# ATR / stops
ATR_LEN = 14
ATR_STOP_K = 1.2
ATR_TRAIL_K = 2.5

# Risk controls
DAILY_LOSS_LIMIT = 0.06  # 6% daily loss
FUNDING_AVOID_MIN = 5    # avoid entries/adds 5 minutes pre-funding

# Runtime
POLL_SEC = 30
TF = "1h"
LOOKBACK = 400
STATE_FILE = "state.json"

# Strategy tuning (adjust during testing)
# Candle-position filters: long allowed below this fraction from high; short allowed above this from low
CANDLE_LONG_MAX_POS_RATIO = 0.75
CANDLE_SHORT_MIN_POS_RATIO = 0.25

# Oversize tolerance (multiplier vs RISK_PCT)
OVERSIZE_TOLERANCE = 1.30

# Data directory base (logs/CSVs go under data/<env>/)
DATA_BASE_DIR = os.getenv("DATA_DIR", "data")

# Auto-switch to testnet after daily loss limit is hit (when flat)
AUTO_TESTNET_ON_DD = True

# Emergency handling
EMERGENCY_POLICY = os.getenv("EMERGENCY_POLICY", "protect_only").lower()
# Options: 'protect_only' | 'flatten_if_safe' | 'flatten_all'
EMERGENCY_MIN_PNL_TO_CLOSE_PCT = float(os.getenv("EMERGENCY_MIN_PNL_TO_CLOSE_PCT", "0"))
EMERGENCY_STOP_FALLBACK_PCT = float(os.getenv("EMERGENCY_STOP_FALLBACK_PCT", "0.015"))  # 1.5%
