import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from auto_trading_bot.main import _replace_stop_only, compute_tp_ladder, place_tp_ladder


class DummyExchange:
    def __init__(self) -> None:
        self.limit_orders: List[Dict[str, Any]] = []
        self.stop_orders: List[Dict[str, Any]] = []
        self.cancel_reduce_calls: List[str] = []

    def market_precision(self, symbol: str) -> Dict[str, float]:
        return getattr(self, "_precision", {})

    def set_precision(self, tick_size: float, step_size: float, min_notional: float) -> None:
        self._precision = {
            "tick_size": tick_size,
            "step_size": step_size,
            "min_notional": min_notional,
        }

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
        return self.limit_orders[-1]

    def create_stop_market_safe(
        self, symbol: str, side: str, stop_price: float, qty: float, reduce_only: bool = True
    ) -> Dict[str, Any]:
        self.stop_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "stop_price": stop_price,
                "qty": qty,
                "reduce_only": reduce_only,
            }
        )
        return self.stop_orders[-1]

    def cancel_reduce_only_stop_orders(self, symbol: str) -> None:
        self.cancel_reduce_calls.append(symbol)


@pytest.fixture
def dummy_exchange() -> DummyExchange:
    ex = DummyExchange()
    ex.set_precision(0.1, 0.001, 1.0)
    return ex


def parse_tp_logs(caplog: pytest.LogCaptureFixture) -> List[Dict[str, Any]]:
    records = []
    for record in caplog.records:
        try:
            payload = json.loads(record.message)
        except json.JSONDecodeError:
            continue
        if payload.get("type", "").startswith("TP_"):
            payload.setdefault("price", 0.0)
            payload.setdefault("qty", 0.0)
            records.append(payload)
    return records


def test_tp_ladder_places_reduce_only_orders(
    dummy_exchange: DummyExchange, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="donchian_bot")

    levels = compute_tp_ladder(100.0, "long", 3.0)
    place_tp_ladder(
        dummy_exchange,
        "BTC/USDT",
        "long",
        3.0,
        100.0,
        levels,
        dummy_exchange.market_precision("BTC/USDT"),
    )

    assert len(dummy_exchange.limit_orders) == 3
    for order in dummy_exchange.limit_orders:
        assert order["reduce_only"] is True
        assert order["time_in_force"] == "GTC"

    place_logs = [log for log in parse_tp_logs(caplog) if log["type"] == "TP_PLACE"]
    assert len(place_logs) == 3


def test_tp_ladder_skip_reasons(
    dummy_exchange: DummyExchange, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="donchian_bot")

    # zero_qty skip (qty rounded down by step size)
    dummy_exchange.set_precision(0.1, 1.0, 1.0)
    levels = [{"px": 100.0, "qty": 0.4}]
    place_tp_ladder(
        dummy_exchange,
        "ETH/USDT",
        "long",
        1.0,
        100.0,
        levels,
        dummy_exchange.market_precision("ETH/USDT"),
    )

    # precision skip (invalid price after alignment)
    dummy_exchange.set_precision(0.1, 0.001, 1.0)
    levels = [{"px": 0.0, "qty": 1.0}]
    place_tp_ladder(
        dummy_exchange,
        "ETH/USDT",
        "long",
        1.0,
        100.0,
        levels,
        dummy_exchange.market_precision("ETH/USDT"),
    )

    # minNotional skip
    dummy_exchange.set_precision(0.1, 0.001, 1000.0)
    levels = [{"px": 100.0, "qty": 0.01}]
    place_tp_ladder(
        dummy_exchange,
        "ETH/USDT",
        "long",
        0.01,
        100.0,
        levels,
        dummy_exchange.market_precision("ETH/USDT"),
    )
    dummy_exchange.set_precision(0.1, 0.001, 1.0)

    skip_logs = [log for log in parse_tp_logs(caplog) if log["type"] == "TP_SKIP"]
    reasons = [log["reason"] for log in skip_logs]
    assert "zero_qty" in reasons
    assert "precision" in reasons
    assert "minNotional" in reasons


def test_replace_stop_only_preserves_tp(dummy_exchange: DummyExchange) -> None:
    _replace_stop_only(dummy_exchange, "BTC/USDT", "long", 99.0, 2.0)

    assert dummy_exchange.cancel_reduce_calls == ["BTC/USDT"]
    assert len(dummy_exchange.stop_orders) == 1
    stop = dummy_exchange.stop_orders[0]
    assert stop["side"].lower() == "sell"
    assert stop["reduce_only"] is True


def test_stop_refresh_preserves_tp_orders(
    dummy_exchange: DummyExchange, caplog: pytest.LogCaptureFixture
) -> None:
    dummy_exchange.set_precision(0.1, 0.001, 1.0)
    caplog.set_level("INFO", logger="donchian_bot")

    levels = compute_tp_ladder(100.0, "long", 3.0)
    place_tp_ladder(
        dummy_exchange,
        "BTC/USDT",
        "long",
        3.0,
        100.0,
        levels,
        dummy_exchange.market_precision("BTC/USDT"),
    )
    initial_tp_count = len(dummy_exchange.limit_orders)

    _replace_stop_only(dummy_exchange, "BTC/USDT", "long", 95.0, 3.0)
    _replace_stop_only(dummy_exchange, "BTC/USDT", "long", 96.0, 3.0)

    assert len(dummy_exchange.limit_orders) == initial_tp_count
    assert dummy_exchange.cancel_reduce_calls == ["BTC/USDT", "BTC/USDT"]


def test_no_cancel_all_usage_remains(repo_root: Path) -> None:
    offending = []
    approved = {"exchange_api.py"}
    for path in repo_root.joinpath("auto_trading_bot").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "cancel_all(" in text:
            if path.name not in approved:
                offending.append(str(path))
    assert not offending, f"cancel_all usage found in: {offending}"
