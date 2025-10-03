import pytest

import auto_trading_bot.strategy.donchian_atr  # noqa: F401

from auto_trading_bot.strategy.base import (
    Signal,
    Strategy,
    available_strategies,
    get_strategy,
    register_strategy,
)


@register_strategy("dummy_registry_strategy")
class _DummyStrategy(Strategy):
    def __init__(self):
        self.invocations = 0

    def on_bar(self, bar, state):
        self.invocations += 1
        return Signal(side="flat", strength=0.0, meta={"calls": self.invocations})


def test_registered_strategy_is_instantiated():
    strat = get_strategy("dummy_registry_strategy")
    assert isinstance(strat, _DummyStrategy)

    result = strat.on_bar({}, {})
    assert result.side == "flat"
    assert result.meta["calls"] == 1


def test_missing_strategy_raises_key_error():
    with pytest.raises(KeyError):
        get_strategy("does_not_exist")


def test_available_strategies_exposes_registry():
    names = available_strategies().keys()
    assert "dummy_registry_strategy" in names


def test_donchian_stub_returns_neutral():
    strat = get_strategy("donchian_atr")
    result = strat.on_bar({}, {})

    assert isinstance(result, Signal)
    assert result.side == "neutral"
    assert result.meta.get("reason") == "stub"
