from __future__ import annotations

import json
import logging
import time
import urllib.request

import pytest

from auto_trading_bot.metrics import dump_current_metrics, start_metrics_server


def test_metrics_endpoint_exposes_expected_series(monkeypatch, caplog):
    monkeypatch.setenv("OBS_DEBUG_ALERTS", "1")
    port = 9200
    with caplog.at_level(logging.INFO, logger="auto_trading_bot.metrics"):
        mgr = start_metrics_server(port, "test", start_heartbeat_thread=False)
        mgr.set_equity(1234.5)
        mgr.set_daily_drawdown(0.15)
        mgr.set_avg_r(0.42)
        mgr.set_time_drift("exchange", 321.0)
        mgr.inc_trade_count(3)
        mgr.inc_signal("BTC/USDT", "1h", 2)
        mgr.inc_order_error("auth", 1)
        mgr.inc_flatten_failure("order_error", 2)
        mgr.inc_flatten_partial("not_profitable", 1)
        mgr.inc_tp_orders("BTC/USDT", "sell", 3)
        mgr.inc_tp_fill("BTC/USDT", "sell", "pyramid", 2)
        mgr.observe_order_latency("entry_market", 0.75)
        mgr.set_heartbeat(time.time())
        mgr.observe_loop_latency(75.0)

        # Allow HTTP server to start
        time.sleep(0.1)
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()

    assert 'bot_equity{account="test"} 1234.5' in body
    assert 'bot_daily_drawdown{account="test"} 0.15' in body
    assert 'bot_avg_r_atr_30{account="test"} 0.42' in body
    assert 'bot_time_drift_ms{source="exchange"} 321.0' in body
    assert 'bot_trade_count_total{account="test"} 3.0' in body
    assert 'bot_signal_emitted_total{symbol="BTC/USDT",timeframe="1h"} 2.0' in body
    assert 'bot_order_errors_total{account="test",reason="auth"} 1.0' in body
    assert 'flatten_failures_total{account="test",reason="order_error"} 2.0' in body
    assert 'flatten_partial_total{account="test",reason="not_profitable"} 1.0' in body
    assert 'tp_orders_placed_total{account="test",side="SELL",symbol="BTC/USDT"} 3.0' in body
    assert 'tp_fills_total{account="test",reason="pyramid",side="SELL",symbol="BTC/USDT"} 2.0' in body
    assert any(
        line.startswith('order_latency_seconds_bucket{') and 'le="0.5"' in line
        for line in body.splitlines()
    ), "order_latency_seconds histogram missing bucket line"
    assert 'order_latency_seconds_sum{account="test",kind="entry_market"}' in body
    assert "bot_heartbeat_ts" in body

    metrics_state = dump_current_metrics()
    assert metrics_state.get("equity") == pytest.approx(1234.5)
    assert metrics_state.get("daily_dd") == pytest.approx(0.15)
    assert metrics_state["flatten_failures"]["order_error"] == pytest.approx(2.0)
    assert metrics_state["flatten_partial"]["not_profitable"] == pytest.approx(1.0)
    assert metrics_state["tp_orders"].get(("BTC/USDT", "SELL")) == pytest.approx(3.0)
    assert metrics_state["tp_fills"].get(("BTC/USDT", "SELL", "pyramid")) == pytest.approx(2.0)
    assert metrics_state["order_latency"]["entry_market"] == pytest.approx(0.75)

    updates = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith("METRICS_UPDATE ")
    ]
    assert any(
        update.get("name") == "bot_equity" and update.get("value") == pytest.approx(1234.5)
        for update in updates
    )
    assert any(
        update.get("name") == "flatten_failures_total"
        and update.get("labels", {}).get("reason") == "order_error"
        for update in updates
    )
    assert any(
        update.get("name") == "tp_orders_placed_total"
        and update.get("labels", {}).get("symbol") == "BTC/USDT"
        for update in updates
    )
