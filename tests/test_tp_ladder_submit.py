import json
from typing import Any, Dict, List

import pytest

from auto_trading_bot.main import compute_tp_ladder, place_tp_ladder, _replace_stop_only


class DummyExchange:
    def __init__(self) -> None:
        self.limit_orders: List[Dict[str, Any]] = []
        self.stop_orders: List[Dict[str, Any]] = []
        self.cancel_reduce_calls: List[str] = []

    def create_limit_order(self, symbol: str, side: str, price: float, qty: float, reduce_only: bool = False, time_in_force: str = "GTC", **kwargs: Any) -> Dict[str, Any]:
        self.limit_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "reduce_only": reduce_only,
                "time_in_force": time_in_force,
            }
        )
        return {
            "symbol": symbol,
            "side": side,
            "price": price,
            "qty": qty,
            "reduce_only": reduce_only,
            "time_in_force": time_in_force,
        }

    def create_stop_market_safe(self, symbol: str, side: str, stop_price: float, qty: float, reduce_only: bool = True) -> Dict[str, Any]:
        self.stop_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "stop_price": stop_price,
                "qty": qty,
                "reduce_only": reduce_only,
            }
        )
        return {
            "symbol": symbol,
            "side": side,
            "stop_price": stop_price,
            "qty": qty,
            "reduce_only": reduce_only,
        }

    def cancel_reduce_only_stop_orders(self, symbol: str) -> None:
        self.cancel_reduce_calls.append(symbol)


@pytest.fixture
def dummy_exchange() -> DummyExchange:
    return DummyExchange()


def test_tp_ladder_places_reduce_only_orders(dummy_exchange: DummyExchange, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="donchian_bot")

    levels = compute_tp_ladder(100.0, "long", 3.0)
    place_tp_ladder(
        dummy_exchange,
        "BTC/USDT",
        "long",
        3.0,
        100.0,
        levels,
        precision={"tick_size": 0.1, "step_size": 0.001, "min_notional": 1.0},
    )

    assert len(dummy_exchange.limit_orders) == 3
    for order in dummy_exchange.limit_orders:
        assert order["reduce_only"] is True
        assert order["time_in_force"] == "GTC"

    place_logs = [json.loads(record.message) for record in caplog.records if "TP_PLACE" in record.message]
    assert len(place_logs) == 3


def test_tp_ladder_skip_min_notional(dummy_exchange: DummyExchange, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="donchian_bot")

    levels = [
        {"px": 100.0, "qty": 0.001},
        {"px": 100.0, "qty": 0.001},
        {"px": 100.0, "qty": 0.001},
    ]
    place_tp_ladder(
        dummy_exchange,
        "ETH/USDT",
        "long",
        0.003,
        100.0,
        levels,
        precision={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
    )

    assert not dummy_exchange.limit_orders

    skip_logs = [json.loads(record.message) for record in caplog.records if "TP_SKIP" in record.message]
    assert len(skip_logs) == 3
    assert all(log["reason"] == "minNotional" for log in skip_logs)


def test_replace_stop_only_preserves_tp(dummy_exchange: DummyExchange) -> None:
    _replace_stop_only(dummy_exchange, "BTC/USDT", "long", 99.0, 2.0)

    assert dummy_exchange.cancel_reduce_calls == ["BTC/USDT"]
    assert len(dummy_exchange.stop_orders) == 1
    stop = dummy_exchange.stop_orders[0]
    assert stop["side"].lower() == "sell"
    assert stop["reduce_only"] is True
