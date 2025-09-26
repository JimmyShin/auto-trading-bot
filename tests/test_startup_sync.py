import math

import pytest

from strategy import DonchianATREngine
from auto_trading_bot.main import startup_sync, _ensure_protective_stop_on_restart
import config as cfg


class FakeBroker:
    def __init__(self, side: str = "short", entry_price: float = 100.0, qty: float = 1.0):
        self._side = side
        self._entry = entry_price
        self._qty = -abs(qty) if side == "short" else abs(qty)
        self._last_stop_calls = {}

    def position_for(self, symbol: str):
        # Report a position only for BTC/USDT; others return empty
        if symbol != "BTC/USDT":
            return {}
        return {
            "positionAmt": self._qty,
            "entryPrice": self._entry,
            "side": self._side,
        }

    def fetch_ticker(self, symbol: str):
        return {"last": self._entry}

    def cancel_all(self, symbol: str):
        return None

    def create_stop_market_safe(self, symbol: str, side: str, stop_price: float, amount: float, reduce_only: bool = True):
        self._last_stop_calls[symbol] = {
            "side": side,
            "stop_price": float(stop_price),
            "amount": float(amount),
            "reduce_only": reduce_only,
        }
        return {"status": "ok"}


def test_startup_sync_sets_entry_and_stop(tmp_path, monkeypatch):
    # Ensure config UNIVERSE includes BTC/USDT
    assert "BTC/USDT" in cfg.UNIVERSE

    # Fresh in-memory state
    eng = DonchianATREngine(persist_state=False, initial_state={})
    b = FakeBroker(side="short", entry_price=200.0, qty=2.0)

    # Run protective stop placement and state sync
    startup_sync(b, eng)

    st = eng.state.get("BTC/USDT", {})
    assert st.get("in_position") is True
    assert st.get("side") == "short"
    assert math.isclose(float(st.get("entry_price", 0.0)), 200.0)

    # _ensure_protective_stop_on_restart stores last_trail_stop
    # When side is short, default fallback stop is entry * (1 + pct)
    expected_stop = 200.0 * (1.0 + cfg.EMERGENCY_STOP_FALLBACK_PCT)
    lt = float(st.get("last_trail_stop", 0.0))
    assert lt == pytest.approx(expected_stop, rel=1e-6)


