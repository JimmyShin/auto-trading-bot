import json
from decimal import Decimal

import pytest

from auto_trading_bot.exchange_api import EquitySnapshotError, ExchangeAPI

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("auto_trading_bot.exchange_api.time.sleep", lambda *_: None)


def _make_api(monkeypatch, *, testnet: bool, responses):
    class _Client:
        def __init__(self, flag: bool) -> None:
            self.testnet = flag
            self.exchange = None

    client = _Client(testnet)
    api = ExchangeAPI(client=client)

    queues = {path: list(items) for path, items in responses.items()}

    def _fake_http_get(path: str):
        queue = queues.get(path)
        if queue is None:
            raise AssertionError(f"Unexpected path {path}")
        if not queue:
            raise AssertionError(f"No more responses for {path}")
        result = queue.pop(0)
        if callable(result):
            result = result()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(api, "_http_get", _fake_http_get)
    return api


def _extract_bal_raw(records):
    bal_logs = [json.loads(r.message) for r in records if _is_bal_raw(r.message)]
    assert len(bal_logs) <= 1, "Expected a single BAL_RAW log"
    return bal_logs[0] if bal_logs else None


def _is_bal_raw(message: str) -> bool:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return False
    return payload.get("type") == "BAL_RAW"


def _basic_payloads():
    balance = [
        {"asset": "USDT", "balance": "15182.4314", "availableBalance": "12000.0000"},
        {"asset": "BUSD", "balance": "1.0000", "availableBalance": "1.0000"},
    ]
    account = {
        "positions": [
            {"symbol": "BTCUSDT", "unrealizedProfit": "12.3456789"},
            {"symbol": "ETHUSDT", "unrealizedProfit": "0.1000001"},
        ],
        "totalMarginBalance": "5956.9841",
    }
    return balance, account


def test_fetch_equity_snapshot_success(monkeypatch, caplog):
    balance, account = _basic_payloads()
    api = _make_api(
        monkeypatch,
        testnet=True,
        responses={
            "/fapi/v2/balance": [balance, balance],
            "/fapi/v2/account": [account, account],
        },
    )

    caplog.set_level("INFO", logger="auto_trading_bot.exchange_api")
    snapshot = api.fetch_equity_snapshot()

    assert snapshot.account_mode == "testnet"
    assert snapshot.source == "binance-usdm-testnet"
    assert (
        snapshot.ts_utc.tzinfo is not None
        and snapshot.ts_utc.tzinfo.utcoffset(snapshot.ts_utc).total_seconds() == 0
    )
    assert snapshot.wallet_balance == pytest.approx(15182.4314)
    expected_upnl = float(Decimal("12.3456789") + Decimal("0.1000001"))
    assert snapshot.unrealized_pnl == pytest.approx(expected_upnl, abs=1e-9)
    assert snapshot.margin_balance == pytest.approx(5956.9841)
    assert snapshot.available_balance == pytest.approx(12000.0)

    payload = _extract_bal_raw(caplog.records)
    assert payload is not None
    assert payload["wallet_balance"] == pytest.approx(snapshot.wallet_balance)
    assert payload["margin_balance"] == pytest.approx(snapshot.margin_balance)
    assert payload["unrealized_pnl"] == pytest.approx(snapshot.unrealized_pnl, abs=1e-9)
    assert payload["available_balance"] == pytest.approx(snapshot.available_balance)
    assert payload["source"] == snapshot.source
    assert payload["account_mode"] == snapshot.account_mode
    assert payload["ts_utc"].endswith("Z")

    assert api.get_equity_usdt() == pytest.approx(snapshot.margin_balance)


def test_fetch_equity_snapshot_high_precision(monkeypatch):
    balance = [
        {"asset": "USDT", "balance": "123.456789012345", "availableBalance": "120.000000987654"},
    ]
    account = {
        "positions": [
            {"symbol": "ADAUSDT", "unrealizedProfit": "0.000000123456"},
            {"symbol": "BNBUSDT", "unrealizedProfit": "-0.000000023456"},
        ],
        "totalMarginBalance": "200.000000987654",
    }
    api = _make_api(
        monkeypatch,
        testnet=False,
        responses={
            "/fapi/v2/balance": [balance],
            "/fapi/v2/account": [account],
        },
    )

    snapshot = api.fetch_equity_snapshot()

    expected_wallet = float(Decimal("123.456789012345"))
    expected_available = float(Decimal("120.000000987654"))
    expected_upnl = float(Decimal("0.000000123456") + Decimal("-0.000000023456"))
    expected_margin = float(Decimal("200.000000987654"))

    assert snapshot.wallet_balance == pytest.approx(expected_wallet, abs=1e-9)
    assert snapshot.available_balance == pytest.approx(expected_available, abs=1e-9)
    assert snapshot.unrealized_pnl == pytest.approx(expected_upnl, abs=1e-9)
    assert snapshot.margin_balance == pytest.approx(expected_margin, abs=1e-9)


def test_fetch_equity_snapshot_retry_success(monkeypatch, caplog):
    balance, account = _basic_payloads()

    api = _make_api(
        monkeypatch,
        testnet=False,
        responses={
            "/fapi/v2/balance": [TimeoutError("boom"), balance, balance],
            "/fapi/v2/account": [account, account],
        },
    )

    caplog.set_level("INFO", logger="auto_trading_bot.exchange_api")
    snapshot = api.fetch_equity_snapshot()

    assert snapshot.account_mode == "live"
    payload = _extract_bal_raw(caplog.records)
    assert payload is not None


def test_fetch_equity_snapshot_failure(monkeypatch):
    api = _make_api(
        monkeypatch,
        testnet=True,
        responses={
            "/fapi/v2/balance": [TimeoutError("boom"), TimeoutError("boom"), TimeoutError("boom")],
            "/fapi/v2/account": [TimeoutError("boom"), TimeoutError("boom"), TimeoutError("boom")],
        },
    )

    with pytest.raises(EquitySnapshotError):
        api.fetch_equity_snapshot()
