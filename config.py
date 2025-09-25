import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _coerce_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("universe must be a list or comma-separated string")


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
POSITION_CAP_MULTIPLE = float(os.getenv("POSITION_CAP_MULTIPLE", "2.0"))
POSITION_CAP_MIN = float(os.getenv("POSITION_CAP_MIN", "10000"))
POSITION_CAP_MAX = float(os.getenv("POSITION_CAP_MAX", "25000"))

def get_position_cap(equity: float) -> float:
    dynamic_cap = equity * POSITION_CAP_MULTIPLE
    return min(max(dynamic_cap, POSITION_CAP_MIN), POSITION_CAP_MAX)

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
CONFIG_JSON_PATH = Path(os.getenv("BOT_CONFIG_JSON", "config.json"))

if CONFIG_JSON_PATH.exists():
    try:
        _json_config = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {CONFIG_JSON_PATH}: {exc}") from exc

    if 'binance_key' in _json_config:
        BINANCE_KEY = str(_json_config['binance_key'])
    if 'binance_secret' in _json_config:
        BINANCE_SECRET = str(_json_config['binance_secret'])
    if 'testnet' in _json_config:
        TESTNET = _coerce_bool(_json_config['testnet'])
    if 'safe_restart' in _json_config:
        SAFE_RESTART = _coerce_bool(_json_config['safe_restart'])
    if 'quote' in _json_config:
        QUOTE = str(_json_config['quote'])
    if 'universe' in _json_config:
        UNIVERSE = _coerce_list(_json_config['universe'])
    if 'risk_pct' in _json_config:
        RISK_PCT = float(_json_config['risk_pct'])
    if 'leverage' in _json_config:
        LEVERAGE = float(_json_config['leverage'])
    if 'enable_pyramiding' in _json_config:
        ENABLE_PYRAMIDING = _coerce_bool(_json_config['enable_pyramiding'])
    if 'pyramid_levels' in _json_config:
        levels = []
        for entry in _json_config['pyramid_levels'] or []:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                levels.append((float(entry[0]), float(entry[1])))
        if levels:
            PYRAMID_LEVELS = levels
    if 'pyramid_cooldown_bars' in _json_config:
        PYRAMID_COOLDOWN_BARS = int(_json_config['pyramid_cooldown_bars'])
    if 'atr_len' in _json_config:
        ATR_LEN = int(_json_config['atr_len'])
    if 'atr_stop_k' in _json_config:
        ATR_STOP_K = float(_json_config['atr_stop_k'])
    if 'atr_trail_k' in _json_config:
        ATR_TRAIL_K = float(_json_config['atr_trail_k'])
    if 'daily_loss_limit' in _json_config:
        DAILY_LOSS_LIMIT = float(_json_config['daily_loss_limit'])
    if 'funding_avoid_min' in _json_config:
        FUNDING_AVOID_MIN = int(_json_config['funding_avoid_min'])
    if 'poll_sec' in _json_config:
        POLL_SEC = int(_json_config['poll_sec'])
    if 'timeframe' in _json_config:
        TF = str(_json_config['timeframe'])
    if 'lookback' in _json_config:
        LOOKBACK = int(_json_config['lookback'])
    if 'state_file' in _json_config:
        STATE_FILE = str(_json_config['state_file'])
    if 'candle_long_max_pos_ratio' in _json_config:
        CANDLE_LONG_MAX_POS_RATIO = float(_json_config['candle_long_max_pos_ratio'])
    if 'candle_short_min_pos_ratio' in _json_config:
        CANDLE_SHORT_MIN_POS_RATIO = float(_json_config['candle_short_min_pos_ratio'])
    if 'oversize_tolerance' in _json_config:
        OVERSIZE_TOLERANCE = float(_json_config['oversize_tolerance'])
    if 'data_dir' in _json_config:
        DATA_BASE_DIR = str(_json_config['data_dir'])
    if 'auto_testnet_on_dd' in _json_config:
        AUTO_TESTNET_ON_DD = _coerce_bool(_json_config['auto_testnet_on_dd'])
    if 'emergency_policy' in _json_config:
        EMERGENCY_POLICY = str(_json_config['emergency_policy']).lower()
    if 'emergency_min_pnl_to_close_pct' in _json_config:
        EMERGENCY_MIN_PNL_TO_CLOSE_PCT = float(_json_config['emergency_min_pnl_to_close_pct'])
    if 'emergency_stop_fallback_pct' in _json_config:
        EMERGENCY_STOP_FALLBACK_PCT = float(_json_config['emergency_stop_fallback_pct'])
    if 'position_cap' in _json_config:
        pc = _json_config['position_cap'] or {}
        if 'multiple' in pc:
            POSITION_CAP_MULTIPLE = float(pc['multiple'])
        if 'min' in pc:
            POSITION_CAP_MIN = float(pc['min'])
        if 'max' in pc:
            POSITION_CAP_MAX = float(pc['max'])

    del _json_config

