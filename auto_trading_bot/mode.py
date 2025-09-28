from enum import Enum


class TradingMode(str, Enum):
    LIVE = "live"
    TESTNET = "testnet"
