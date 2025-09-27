import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from auto_trading_bot.exchange_api import EquitySnapshot
from auto_trading_bot.reporter import Reporter

pytestmark = pytest.mark.unit


class DummyMetrics:
    def __init__(self) -> None:
        self.calls = []

    def update_daily_drawdown(self, ratio: float, *, equity: float, ts: datetime) -> None:
        self.calls.append((ratio, equity, ts))


def _dd_logs(records):
    logs = []
    for record in records:
        try:
            payload = json.loads(record.message)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "DD_CALC":
            logs.append(payload)
    return logs


def _make_snapshot(
    *,
    ts: datetime,
    margin: Decimal | float,
    wallet: Decimal | float | None = None,
    available: Decimal | float | None = None,
    unrealized: Decimal | float = 0,
    source: str = "binance-usdm-testnet",
    mode: str = "testnet",
) -> EquitySnapshot:
    wallet_val = wallet if wallet is not None else margin
    available_val = available if available is not None else wallet_val
    return EquitySnapshot(
        ts_utc=ts,
        wallet_balance=float(wallet_val),
        margin_balance=float(margin),
        unrealized_pnl=float(unrealized),
        available_balance=float(available_val),
        source=source,
        account_mode=mode,
    )


def test_apply_equity_snapshot_daily_flow(tmp_path, caplog):
    metrics = DummyMetrics()
    reporter = Reporter(base_dir=str(tmp_path), environment="testnet", metrics=metrics)

    caplog.set_level(logging.INFO, logger="auto_trading_bot.reporter")
    day0 = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

    snap1 = _make_snapshot(ts=day0, margin=Decimal("1000.0"))
    ratio1 = reporter.apply_equity_snapshot(snap1, now_utc=day0)
    assert ratio1 == pytest.approx(0.0)
    assert len(metrics.calls) == 1
    call_ratio, call_equity, call_ts = metrics.calls[-1]
    assert call_ratio == pytest.approx(0.0)
    assert call_equity == pytest.approx(1000.0)
    assert call_ts == day0

    logs = _dd_logs(caplog.records)
    assert len(logs) == 1
    payload = logs[0]
    expected_keys = {"type", "ts_utc", "equity", "peak_equity", "dd_ratio", "source", "account_mode"}
    assert set(payload.keys()) == expected_keys
    assert payload["type"] == "DD_CALC"
    assert payload["equity"] == pytest.approx(1000.0)
    assert payload["peak_equity"] == pytest.approx(1000.0)
    assert payload["dd_ratio"] == pytest.approx(0.0)
    assert payload["ts_utc"].endswith("Z")
    assert payload["equity"] >= 0 and payload["peak_equity"] >= 0 and payload["dd_ratio"] >= 0

    caplog.clear()

    snap2_time = day0 + timedelta(hours=1)
    snap2 = _make_snapshot(ts=snap2_time, margin=Decimal("900.0"))
    ratio2 = reporter.apply_equity_snapshot(snap2, now_utc=snap2_time)
    expected_dd2 = float((Decimal("1000.0") - Decimal("900.0")) / Decimal("1000.0"))
    assert ratio2 == pytest.approx(expected_dd2, abs=1e-12)
    assert len(metrics.calls) == 2
    call_ratio, call_equity, call_ts = metrics.calls[-1]
    assert call_ratio == pytest.approx(expected_dd2, abs=1e-12)
    assert call_equity == pytest.approx(900.0)
    assert call_ts == snap2_time

    logs = _dd_logs(caplog.records)
    assert len(logs) == 1
    assert logs[0]["dd_ratio"] == pytest.approx(expected_dd2, abs=1e-12)

    caplog.clear()

    snap3_time = day0 + timedelta(hours=2)
    snap3 = _make_snapshot(ts=snap3_time, margin=Decimal("1100.0"))
    ratio3 = reporter.apply_equity_snapshot(snap3, now_utc=snap3_time)
    assert ratio3 == pytest.approx(0.0)
    assert len(metrics.calls) == 3
    assert metrics.calls[-1][0] == pytest.approx(0.0)
    assert metrics.calls[-1][1] == pytest.approx(1100.0)

    caplog.clear()

    next_day = day0 + timedelta(days=1)
    snap4 = _make_snapshot(ts=next_day, margin=Decimal("1000.0"))
    ratio4 = reporter.apply_equity_snapshot(snap4, now_utc=next_day)
    assert ratio4 == pytest.approx(0.0)
    assert len(metrics.calls) == 4
    assert metrics.calls[-1][0] == pytest.approx(0.0)
    assert metrics.calls[-1][1] == pytest.approx(1000.0)

    day0_key = day0.strftime("%Y-%m-%d")
    next_day_key = next_day.strftime("%Y-%m-%d")
    peaks = reporter.daily_peak_equity
    assert peaks[day0_key] == pytest.approx(1100.0)
    assert peaks[next_day_key] == pytest.approx(1000.0)

    logs = _dd_logs(caplog.records)
    assert len(logs) == 1
    assert logs[0]["dd_ratio"] == pytest.approx(0.0)
    assert logs[0]["ts_utc"].endswith("Z")


def test_apply_equity_snapshot_decimal_precision(tmp_path, caplog):
    metrics = DummyMetrics()
    reporter = Reporter(base_dir=str(tmp_path), environment="testnet", metrics=metrics)

    caplog.set_level(logging.INFO, logger="auto_trading_bot.reporter")

    base_time = datetime(2025, 1, 2, 15, 30, tzinfo=timezone.utc)
    peak_dec = Decimal("1234.567890123456")
    snap_peak = _make_snapshot(ts=base_time, margin=peak_dec)
    reporter.apply_equity_snapshot(snap_peak, now_utc=base_time)
    assert metrics.calls[-1][0] == pytest.approx(0.0)

    caplog.clear()

    equity_dec = peak_dec - Decimal("0.123456789012")
    later_time = base_time + timedelta(minutes=10)
    snap_drawdown = _make_snapshot(ts=later_time, margin=equity_dec)
    ratio = reporter.apply_equity_snapshot(snap_drawdown, now_utc=later_time)

    expected_dd = float((peak_dec - equity_dec) / peak_dec)
    assert ratio == pytest.approx(expected_dd, abs=1e-9)

    assert len(metrics.calls) == 2
    call_ratio, call_equity, call_ts = metrics.calls[-1]
    assert call_ratio == pytest.approx(expected_dd, abs=1e-9)
    assert call_equity == pytest.approx(float(equity_dec), abs=1e-9)
    assert call_ts == later_time

    logs = _dd_logs(caplog.records)
    assert len(logs) == 1
    payload = logs[0]
    assert payload["dd_ratio"] == pytest.approx(expected_dd, abs=1e-9)
    assert payload["equity"] == pytest.approx(float(equity_dec), abs=1e-9)
    assert payload["peak_equity"] >= payload["equity"]
    assert payload["ts_utc"].endswith("Z")
