from __future__ import annotations

from datetime import datetime, timezone

import pytest

from auto_trading_bot.main import EmergencyManager


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
                        pos.get("positionAmt")
                        or pos.get("contracts")
                        or pos.get("amount")
                        or 0.0
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
    manager.record_error('auth')
    assert manager.should_trigger_kill_switch() is False
    manager.record_error('auth')
    assert manager.should_trigger_kill_switch() is True
    manager.execute_cleanup("kill-switch-test", datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert any("kill-switch-test" in msg for msg in messages)

