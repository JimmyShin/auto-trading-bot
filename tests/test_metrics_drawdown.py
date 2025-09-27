import pytest
from datetime import datetime, timedelta, timezone

from auto_trading_bot.metrics import Metrics

pytestmark = pytest.mark.unit


def test_initial_state():
    metrics = Metrics()
    assert metrics.bot_daily_drawdown() == 0.0
    assert metrics.last_drawdown_timestamp() is None
    assert metrics.last_drawdown_equity() is None


def test_first_update_and_helpers():
    metrics = Metrics()
    ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    metrics.update_daily_drawdown(0.1, equity=1000.0, ts=ts1)

    assert metrics.bot_daily_drawdown() == pytest.approx(0.1)
    assert metrics.last_drawdown_timestamp() == ts1
    assert metrics.last_drawdown_equity() == pytest.approx(1000.0)


def test_clamp_ratio_bounds():
    metrics = Metrics()
    ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)
    ts3 = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)

    metrics.update_daily_drawdown(0.5, equity=900.0, ts=ts1)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.5)

    metrics.update_daily_drawdown(1.5, equity=950.0, ts=ts2)
    assert metrics.bot_daily_drawdown() == pytest.approx(1.0)

    metrics.update_daily_drawdown(-0.2, equity=970.0, ts=ts3)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.0)


def test_ordering_ignores_stale_updates():
    metrics = Metrics()
    ts1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2025, 1, 1, 13, 0, tzinfo=timezone.utc)

    metrics.update_daily_drawdown(0.2, equity=1000.0, ts=ts1)
    metrics.update_daily_drawdown(0.3, equity=1100.0, ts=ts2)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.3)

    # Equal timestamp ignored
    metrics.update_daily_drawdown(0.9, equity=1200.0, ts=ts2)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.3)

    # Older timestamp ignored
    metrics.update_daily_drawdown(0.9, equity=1200.0, ts=ts1)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.3)

    # Newer timestamp accepted
    ts3 = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
    metrics.update_daily_drawdown(0.4, equity=1300.0, ts=ts3)
    assert metrics.bot_daily_drawdown() == pytest.approx(0.4)


def test_timestamp_validation():
    metrics = Metrics()
    naive_ts = datetime(2025, 1, 1, 12, 0)
    with pytest.raises(ValueError):
        metrics.update_daily_drawdown(0.1, equity=1000.0, ts=naive_ts)

    non_utc = timezone(timedelta(hours=9))
    with pytest.raises(ValueError):
        metrics.update_daily_drawdown(0.1, equity=1000.0, ts=datetime(2025, 1, 1, 12, 0, tzinfo=non_utc))


def test_precision_preserved_within_tolerance():
    metrics = Metrics()
    peak_ts = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    fine_ratio = 0.123456789012
    metrics.update_daily_drawdown(fine_ratio, equity=1500.0, ts=peak_ts)
    assert metrics.bot_daily_drawdown() == pytest.approx(fine_ratio, abs=1e-12)
