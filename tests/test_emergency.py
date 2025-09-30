from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from auto_trading_bot.main import EmergencyManager
from tests.helpers import FakeStateStore, StubExchange, StubNotifier


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
        if symbol in self.positions:
            self.positions[symbol]["positionAmt"] = "0"
        return {"symbol": symbol, "side": side, "amount": qty}

    def flatten_all(
        self,
        symbols,
        *,
        retries: int = 0,
        backoff_sec: float = 0.0,
    ):
        results = []
        for symbol in symbols:
            attempts = 0
            while attempts <= retries:
                pos = self.position_for(symbol)
                try:
                    amt = float(
                        pos.get("positionAmt") or pos.get("contracts") or pos.get("amount") or 0.0
                    )
                except Exception:
                    amt = 0.0
                if abs(amt) <= 1e-9:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "closed",
                            "attempts": max(attempts - 1, 0),
                            "remaining_qty": 0.0,
                            "order_side": None,
                            "order_qty": 0.0,
                        }
                    )
                    break
                attempts += 1
                side = "sell" if amt > 0 else "buy"
                try:
                    self.create_market_order_safe(symbol, side, abs(amt))
                except Exception as exc:
                    if attempts > retries:
                        results.append(
                            {
                                "symbol": symbol,
                                "status": "failed",
                                "attempts": attempts,
                                "remaining_qty": abs(amt),
                                "order_side": side,
                                "order_qty": 0.0,
                                "reason": f"order:{exc}",
                            }
                        )
                        break
                    continue
                try:
                    remaining = float(self.position_for(symbol).get("positionAmt") or 0.0)
                except Exception:
                    remaining = 0.0
                if abs(remaining) <= 1e-9:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "closed",
                            "attempts": attempts,
                            "remaining_qty": 0.0,
                            "order_side": side,
                            "order_qty": abs(amt),
                        }
                    )
                    break
                if attempts > retries:
                    results.append(
                        {
                            "symbol": symbol,
                            "status": "failed",
                            "attempts": attempts,
                            "remaining_qty": abs(remaining),
                            "order_side": side,
                            "order_qty": abs(amt),
                            "reason": "remaining",
                        }
                    )
                    break
        return results

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {"last": 110.0}


class DummyEngine:
    def __init__(self) -> None:
        self.state = {"daily": {"anchor": 1000.0}}

    def clear_position_state(self, symbol: str) -> None:
        entry = self.state.setdefault(symbol, {})
        entry["in_position"] = False

def make_manager(
    policy: str,
    exchange: DummyExchange,
    engine: DummyEngine,
    notifier,
    *,
    retries: int = 2,
    backoff: float = 0.0,
):
    return EmergencyManager(
        exchange=exchange,
        engine=engine,
        universe=["BTC/USDT"],
        notify_callback=notifier,
        auto_testnet_on_dd=True,
        daily_dd_limit=0.05,
        emergency_policy=policy,
        kill_switch={"auth_failures": 2, "nonce_errors": 2, "max_retries": 2},
        order_retry_attempts=retries,
        order_retry_backoff_sec=backoff,
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
    assert any("triggered by flatten-test" in msg for msg in messages)
    assert any("executed successfully" in msg for msg in messages)
    assert not any("incomplete" in msg for msg in messages)


class FailingExchange(DummyExchange):
    def create_market_order_safe(self, symbol: str, side: str, qty: float):
        raise RuntimeError("order rejected")


def test_emergency_flatten_all_failure_alerts():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    exchange = FailingExchange()
    exchange.positions["BTC/USDT"] = {
        "positionAmt": "2",
        "entryPrice": "95",
        "unrealizedProfit": "-10",
    }
    engine = DummyEngine()
    engine.state["BTC/USDT"] = {"in_position": True, "side": "long", "entry_price": 95}
    manager = make_manager("flatten_all", exchange, engine, notifier, retries=1)

    manager.execute_cleanup("signal", datetime(2024, 1, 1, tzinfo=timezone.utc))

    failure_msgs = [m for m in messages if "incomplete" in m]
    assert failure_msgs, f"Expected failure alert, got: {messages}"
    fail_msg = failure_msgs[0]
    assert "BTC/USDT" in fail_msg
    assert "order rejected" in fail_msg
    assert engine.state["BTC/USDT"]["in_position"] is True
    assert exchange.orders == []


def test_kill_switch_triggers_after_repeated_errors():
    messages = []

    def notifier(msg: str) -> bool:
        messages.append(msg)
        return True

    manager = make_manager("protect_only", DummyExchange(), DummyEngine(), notifier)
    manager.record_error("auth")
    assert manager.should_trigger_kill_switch() is False
    manager.record_error("auth")
    assert manager.should_trigger_kill_switch() is True
    manager.execute_cleanup("kill-switch-test", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert any("kill-switch-test" in msg for msg in messages)


def _make_guard_manager(*, state_store: FakeStateStore, notifier):
    exchange = StubExchange(account_mode="live")
    manager = EmergencyManager(
        exchange=exchange,
        engine=DummyEngine(),
        universe=["BTC/USDT"],
        notify_callback=lambda msg: True,
        auto_testnet_on_dd=True,
        daily_dd_limit=0.05,
        emergency_policy="protect_only",
        kill_switch={"auth_failures": 5, "nonce_errors": 5, "max_retries": 5},
        state_store=state_store,
        notify_func=notifier,
    )
    return manager, exchange


def test_guard_dedupe_same_day(monkeypatch):
    state = FakeStateStore()
    messages: list[str] = []
    manager, exchange = _make_guard_manager(
        state_store=state,
        notifier=lambda msg: messages.append(msg) or True,
    )

    now = datetime(2025, 9, 30, 10, 0, tzinfo=timezone.utc)
    assert manager.handle_daily_drawdown(0.06, now) is True
    assert exchange.set_testnet_calls == [True]
    assert messages == [
        "[EMERGENCY] AUTO_TESTNET_ON_DD triggered → switched to testnet, live entries disabled until UTC reset."
    ]
    assert state.get(manager._guard_state_key) == "2025-09-30"

    later_same_day = now + timedelta(hours=2)
    assert manager.handle_daily_drawdown(0.06, later_same_day) is False
    assert exchange.set_testnet_calls == [True]
    assert messages == [
        "[EMERGENCY] AUTO_TESTNET_ON_DD triggered → switched to testnet, live entries disabled until UTC reset."
    ]


@pytest.mark.parametrize(
    "delta, expected_triggers",
    [
        (timedelta(milliseconds=1), 1),  # same UTC date
        (timedelta(seconds=1), 2),       # crosses to next UTC date
    ],
)
def test_guard_dedupe_utc_boundary(delta, expected_triggers):
    state = FakeStateStore()
    messages: list[str] = []
    manager, exchange = _make_guard_manager(
        state_store=state,
        notifier=lambda msg: messages.append(msg) or True,
    )

    base = datetime(2025, 9, 30, 23, 59, 59, tzinfo=timezone.utc)
    assert manager.handle_daily_drawdown(0.06, base) is True

    second_moment = base + delta
    triggered = manager.handle_daily_drawdown(0.06, second_moment)

    assert len(messages) == expected_triggers
    assert len(exchange.set_testnet_calls) == expected_triggers
    if expected_triggers == 2:
        assert triggered is True
        assert state.get(manager._guard_state_key) == "2025-10-01"
    else:
        assert triggered is False


def test_kill_switch_does_not_trigger_below_threshold():
    events: list[str] = []

    manager = EmergencyManager(
        exchange=StubExchange(),
        engine=DummyEngine(),
        universe=["BTC/USDT"],
        notify_callback=lambda msg: events.append("notify") or True,
        auto_testnet_on_dd=False,
        daily_dd_limit=1.0,
        emergency_policy="protect_only",
        kill_switch={"auth_failures": 3, "nonce_errors": 3, "max_retries": 3},
    )

    manager.record_error("auth")
    assert manager.should_trigger_kill_switch() is False
    assert events == []
    assert manager.kill_switch_triggered is False


def test_kill_switch_log_sequence(monkeypatch, caplog):
    events: list[tuple[str, str | int]] = []

    def notifier(message: str) -> bool:
        events.append(("notify", message))
        return True

    manager = EmergencyManager(
        exchange=StubExchange(),
        engine=DummyEngine(),
        universe=["BTC/USDT"],
        notify_callback=lambda msg: True,
        auto_testnet_on_dd=False,
        daily_dd_limit=1.0,
        emergency_policy="protect_only",
        kill_switch={"auth_failures": 1, "nonce_errors": 1, "max_retries": 1},
        notify_func=notifier,
    )

    manager.record_error("auth")

    def cleanup(reason: str) -> None:
        events.append(("cleanup", reason))

    logger = logging.getLogger("donchian_bot")

    with caplog.at_level(logging.ERROR, logger="donchian_bot"):
        with pytest.raises(SystemExit) as excinfo:
            if manager.should_trigger_kill_switch():
                logger.error("[kill-switch] triggered")
                manager.notify("[CRITICAL] Kill-switch activated due to repeated failures -> bot terminated.")
                cleanup("kill-switch")
                raise SystemExit(1)

    events.append(("exit", excinfo.value.code))

    assert events[0][0] == "notify"
    assert events[1] == ("cleanup", "kill-switch")
    assert events[2] == ("exit", 1)
    assert any("[kill-switch]" in record.message for record in caplog.records)
