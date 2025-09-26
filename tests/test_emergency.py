from __future__ import annotations

from datetime import datetime, timezone

import pytest

from main import EmergencyManager


class DummyExchange:
    def __init__(self) -> None:
        self.testnet_calls = []
        self.orders: list[tuple[str, str, float]] = []
        self.positions: dict[str, dict[str, str]] = {}

    def set_testnet(self, enabled: bool) -> bool:
        self.testnet_calls.append(bool(enabled))
        return True

    def position_for(self, symbol: str) -> dict[str, str]:
        return self.positions.get(symbol, {})

    def cancel_all(self, symbol: str) -> None:  # pragma: no cover - noop
        return None

    def create_market_order_safe(self, symbol: str, side: str, qty: float):
        self.orders.append((symbol, side, qty))
        # emulate flatten
        if symbol in self.positions:
            self.positions[symbol]['positionAmt'] = '0'
        return {"symbol": symbol, "side": side, "amount": qty}

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": 110.0}


class DummyEngine:
    def __init__(self) -> None:
        self.state = {"daily": {"anchor": 1000.0}}

    def clear_position_state(self, symbol: str) -> None:
        entry = self.state.setdefault(symbol, {})
        entry['in_position'] = False


def make_manager(policy: str, exchange: DummyExchange, engine: DummyEngine, notifier):
    return EmergencyManager(
        exchange=exchange,
        engine=engine,
        universe=["BTC/USDT"],
        notify_callback=notifier,
        auto_testnet_on_dd=True,
        daily_dd_limit=0.05,
        emergency_policy=policy,
        kill_switch={"auth_failures": 2, "nonce_errors": 2, "max_retries": 2},
    )


def test_auto_testnet_on_dd_triggers_and_blocks_entries():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    exchange = DummyExchange()
    engine = DummyEngine()
    engine.state["BTC/USDT"] = {"in_position": False}
    manager = make_manager("protect_only", exchange, engine, notifier)

    triggered = manager.handle_daily_drawdown(0.06, datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert triggered is True
    assert exchange.testnet_calls == [True]
    assert manager.should_block_entries(datetime(2024, 1, 1, tzinfo=timezone.utc)) is True
    assert any("AUTO_TESTNET_ON_DD" in m for m in messages)


def test_emergency_policy_protect_only_blocks_entries():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    exchange = DummyExchange()
    engine = DummyEngine()
    manager = make_manager("protect_only", exchange, engine, notifier)
    manager.execute_cleanup("manual-test", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert manager.should_block_entries(datetime(2024, 1, 1, tzinfo=timezone.utc)) is True
    assert not exchange.orders
    assert any("protect_only" in msg for msg in messages)


def test_emergency_policy_flatten_all_closes_positions():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    exchange = DummyExchange()
    exchange.positions["BTC/USDT"] = {
        "positionAmt": "1",
        "entryPrice": "100",
        "unrealizedProfit": "5",
    }
    engine = DummyEngine()
    engine.state["BTC/USDT"] = {"in_position": True, "side": "long", "entry_price": 100}
    manager = make_manager("flatten_all", exchange, engine, notifier)

    manager.execute_cleanup("flatten-test", datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert exchange.orders == [("BTC/USDT", "sell", 1.0)]
    assert engine.state["BTC/USDT"]["in_position"] is False
    assert any("flatten_all" in msg for msg in messages)


def test_kill_switch_triggers_after_repeated_errors():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    manager = make_manager("protect_only", DummyExchange(), DummyEngine(), notifier)
    manager.record_error('auth')
    assert manager.should_trigger_kill_switch() is False
    manager.record_error('auth')
    assert manager.should_trigger_kill_switch() is True
    # Cleanup should emit policy message
    manager.execute_cleanup("kill-switch-test", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert any("kill-switch-test" in msg for msg in messages)
