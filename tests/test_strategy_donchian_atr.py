import pytest
import pandas as pd

from auto_trading_bot.strategy.base import Signal
from auto_trading_bot.strategy.donchian_atr import DonchianATRStrategy


def make_history(values):
    rows = []
    for idx, price in enumerate(values):
        rows.append({
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
        })
    return pd.DataFrame(rows)


def run_signal(strategy, prices, state=None):
    df = make_history(prices)
    bar = {"history": df}
    state = {} if state is None else state
    signal = strategy.on_bar(bar, state)
    return signal, state


def test_long_signal_on_breakout_with_atr_gate(monkeypatch):
    strat = DonchianATRStrategy()
    big_jump = list(range(20)) + [30]
    signal, state = run_signal(strat, big_jump)
    assert signal.side == "long"
    assert signal.meta["donch_high"] < big_jump[-1]
    assert signal.meta["atr_pct"] >= 0
    assert state["donch_high_20"] == signal.meta["donch_high"]
    assert state["donch_low_20"] == signal.meta["donch_low"]
    assert state["atr_pct"] == signal.meta["atr_pct"]
    assert state["price"] == big_jump[-1]


def test_short_signal_on_breakdown_with_atr_gate():
    strat = DonchianATRStrategy()
    prices = [30] * 19 + [10]
    signal, state = run_signal(strat, prices)
    assert signal.side == "short"
    assert signal.meta["donch_low"] > prices[-1]
    assert signal.meta["exits"]["opp_band"] == signal.meta["donch_high"]


def test_breakout_blocked_when_atr_below_gate(monkeypatch):
    strat = DonchianATRStrategy()
    monkeypatch.setattr("auto_trading_bot.strategy.donchian_atr.ATR_GATE", 10.0)
    signal, state = run_signal(strat, list(range(20)) + [30])
    assert signal.side == "neutral"
    assert "atr_gate" in signal.meta["reasons"]


def test_meta_contains_exit_hints():
    strat = DonchianATRStrategy()
    signal, state = run_signal(strat, list(range(20)) + [30])
    assert signal.meta["sl_atr_k"] == 1.5
    exits = signal.meta["exits"]
    assert "midline" in exits and "opp_band" in exits
    assert signal.meta["exit_policy"]["midline"] is True
    assert signal.meta["exit_policy"]["band"] is True


def test_reentry_cooldown(monkeypatch):
    strat = DonchianATRStrategy()
    monkeypatch.setattr("auto_trading_bot.strategy.donchian_atr.REENTRY_COOLDOWN_BARS", 2)
    prices = list(range(20)) + [30]
    signal1, state = run_signal(strat, prices)
    assert signal1.side == "long"
    signal2, state = run_signal(strat, prices + [31], state=state)
    assert signal2.side == "neutral"
    assert "cooldown_active" in signal2.meta["reasons"]
    assert signal2.meta["cooldown_left"] > 0
