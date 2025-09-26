from __future__ import annotations
import json
import logging
import pytest

import time
import urllib.request

from auto_trading_bot.metrics import start_metrics_server, get_metrics_manager, dump_current_metrics


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
    assert 'bot_heartbeat_ts' in body

    metrics_state = dump_current_metrics()
    assert metrics_state.get('equity') == pytest.approx(1234.5)
    assert metrics_state.get('daily_dd') == pytest.approx(0.15)

    updates = [json.loads(record.message.split(' ', 1)[1]) for record in caplog.records if record.message.startswith('METRICS_UPDATE ')]
    assert any(update.get('name') == 'bot_equity' and update.get('value') == pytest.approx(1234.5) for update in updates)