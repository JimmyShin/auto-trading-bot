import pandas as pd
import pytest

from auto_trading_bot.strategy.ma_cross import MovingAverageCrossStrategy
from auto_trading_bot.strategy.base import Signal


@pytest.fixture
def strategy():
    return MovingAverageCrossStrategy()


def build_history(values):
    data = {
        "open": values,
        "high": [v + 1 for v in values],
        "low": [v - 1 for v in values],
        "close": values,
    }
    return pd.DataFrame(data)


def test_long_signal_when_ma_crosses_up(monkeypatch, strategy):
    history = build_history([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25])

    def fake_ma(df, fast_period, slow_period):
        assert fast_period == 5
        assert slow_period == 20
        assert df is history
        return {
            "long": True,
            "short": False,
            "fast_ma": 10.0,
            "slow_ma": 9.5,
            "regime": "RANGE",
            "alignment": {},
        }

    monkeypatch.setattr("auto_trading_bot.strategy.ma_cross.ma_crossover_signal", fake_ma)
    state = {}
    signal = strategy.on_bar({"history": history}, state)

    assert isinstance(signal, Signal)
    assert signal.side == "long"
    assert signal.strength == 1.0
    assert signal.meta["analysis"]["long"] is True
    assert state["ma_cross_last"]["analysis"]["long"] is True


def test_short_signal_when_ma_crosses_down(monkeypatch, strategy):
    history = build_history([10] * 20)

    def fake_ma(df, fast_period, slow_period):
        return {
            "long": False,
            "short": True,
            "fast_ma": 8.0,
            "slow_ma": 9.0,
            "regime": "RANGE",
            "alignment": {},
        }

    monkeypatch.setattr("auto_trading_bot.strategy.ma_cross.ma_crossover_signal", fake_ma)
    state = {}
    signal = strategy.on_bar({"df": history}, state)

    assert signal.side == "short"
    assert signal.meta["analysis"]["short"] is True


def test_neutral_signal_when_history_missing(strategy):
    state = {}
    signal = strategy.on_bar({}, state)

    assert signal.side == "neutral"
    assert signal.meta["reason"] == "missing_history"
    assert "ma_cross_last" not in state
