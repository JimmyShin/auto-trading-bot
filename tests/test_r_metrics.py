import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from auto_trading_bot.exchange_api import EquitySnapshot
from auto_trading_bot.metrics import Metrics
from auto_trading_bot.reporter import Reporter

pytestmark = pytest.mark.unit


class DummyMetrics(Metrics):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def update_daily_drawdown(self, ratio: float, *, equity: float, ts: datetime) -> None:
        super().update_daily_drawdown(ratio, equity=equity, ts=ts)
        self.calls.append((ratio, equity, ts))


def _make_snapshot(
    *,
    ts: datetime,
    wallet: Decimal | float,
    margin: Decimal | float,
    available: Decimal | float,
    unrealized: Decimal | float = 0,
    source: str = "binance-usdm-testnet",
    mode: str = "testnet",
) -> EquitySnapshot:
    return EquitySnapshot(
        ts_utc=ts,
        wallet_balance=float(wallet),
        margin_balance=float(margin),
        unrealized_pnl=float(unrealized),
        available_balance=float(available),
        source=source,
        account_mode=mode,
    )


def _extract(payloads, record_type):
    out = []
    for rec in payloads:
        try:
            data = json.loads(rec.message)
        except json.JSONDecodeError:
            continue
        if data.get("type") == record_type:
            out.append(data)
    return out


def test_reporter_metrics_equity_flow(caplog):
    metrics = DummyMetrics()
    reporter = Reporter(base_dir="/tmp", environment="testnet", metrics=metrics)

    caplog.set_level(logging.INFO, logger="auto_trading_bot.reporter")
    base = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    snap1 = _make_snapshot(
        ts=base, wallet=Decimal("1000"), margin=Decimal("1000"), available=Decimal("1000")
    )
    dd1 = reporter.apply_equity_snapshot(snap1, now_utc=base)
    assert dd1 == pytest.approx(0.0, abs=1e-12)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.0, abs=1e-12)
    assert len(metrics.calls) == 1
    assert metrics.calls[-1][1] == pytest.approx(1000.0)

    dd_logs = _extract(caplog.records, "DD_CALC")
    assert len(dd_logs) == 1
    payload = dd_logs[0]
    assert set(payload.keys()) == {
        "type",
        "ts_utc",
        "equity",
        "peak_equity",
        "dd_ratio",
        "source",
        "account_mode",
    }
    assert payload["dd_ratio"] == pytest.approx(0.0)
    assert payload["equity"] == pytest.approx(1000.0)
    assert payload["peak_equity"] == pytest.approx(1000.0)
    assert payload["ts_utc"].endswith("Z")

    caplog.clear()
    snap2 = _make_snapshot(
        ts=base + timedelta(hours=1),
        wallet=Decimal("1000"),
        margin=Decimal("900"),
        available=Decimal("900"),
    )
    dd2 = reporter.apply_equity_snapshot(snap2, now_utc=base + timedelta(hours=1))
    expected_dd2 = float((Decimal("1000") - Decimal("900")) / Decimal("1000"))
    assert dd2 == pytest.approx(expected_dd2, abs=1e-12)
    assert metrics.bot_daily_drawdown() == pytest.approx(expected_dd2, abs=1e-12)
    dd_logs = _extract(caplog.records, "DD_CALC")
    assert len(dd_logs) == 1
    assert dd_logs[0]["dd_ratio"] == pytest.approx(expected_dd2, abs=1e-12)

    caplog.clear()
    snap3 = _make_snapshot(
        ts=base + timedelta(hours=2),
        wallet=Decimal("1100"),
        margin=Decimal("1100"),
        available=Decimal("1100"),
    )
    dd3 = reporter.apply_equity_snapshot(snap3, now_utc=base + timedelta(hours=2))
    assert dd3 == pytest.approx(0.0, abs=1e-12)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.0, abs=1e-12)
    dd_logs = _extract(caplog.records, "DD_CALC")
    assert len(dd_logs) == 1
    assert dd_logs[0]["peak_equity"] == pytest.approx(1100.0)

    caplog.clear()
    next_day = base + timedelta(days=1)
    snap4 = _make_snapshot(
        ts=next_day,
        wallet=Decimal("1000"),
        margin=Decimal("1000"),
        available=Decimal("1000"),
    )
    dd4 = reporter.apply_equity_snapshot(snap4, now_utc=next_day)
    assert dd4 == pytest.approx(0.0)
    dd_logs = _extract(caplog.records, "DD_CALC")
    assert len(dd_logs) == 1
    assert dd_logs[0]["peak_equity"] == pytest.approx(1000.0)


def test_decimal_precision(caplog):
    metrics = DummyMetrics()
    reporter = Reporter(base_dir="/tmp", environment="testnet", metrics=metrics)

    caplog.set_level(logging.INFO, logger="auto_trading_bot.reporter")

    base = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    wallet = Decimal("15182.431456789012345")
    margin_peak = Decimal("15182.431456789012345")
    snap_peak = _make_snapshot(ts=base, wallet=wallet, margin=margin_peak, available=wallet)
    reporter.apply_equity_snapshot(snap_peak, now_utc=base)

    caplog.clear()
    later = base + timedelta(minutes=5)
    margin_low = margin_peak - Decimal("0.123456789012345")
    snap_low = _make_snapshot(ts=later, wallet=wallet, margin=margin_low, available=wallet)
    dd = reporter.apply_equity_snapshot(snap_low, now_utc=later)

    expected_dd = float((margin_peak - margin_low) / margin_peak)
    assert dd == pytest.approx(expected_dd, abs=1e-9)
    assert metrics.bot_daily_drawdown() == pytest.approx(expected_dd, abs=1e-9)

    dd_logs = _extract(caplog.records, "DD_CALC")
    assert len(dd_logs) == 1
    payload = dd_logs[0]
    assert payload["dd_ratio"] == pytest.approx(expected_dd, abs=1e-9)
    assert payload["equity"] == pytest.approx(float(margin_low), abs=1e-9)
    assert payload["peak_equity"] == pytest.approx(float(margin_peak), abs=1e-9)
    assert payload["ts_utc"].endswith("Z")
