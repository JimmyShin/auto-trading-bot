import csv
import json
from datetime import datetime, timezone

import pandas as pd

from auto_trading_bot.strategy.base import Signal
from auto_trading_bot.strategy.ma_cross import MovingAverageCrossStrategy
from auto_trading_bot.strategy.donchian_atr import DonchianATRStrategy
from auto_trading_bot.strategy.utils import strategy_tags, params_hash, extract_strategy_tags
from auto_trading_bot.reporter import Reporter


def make_history(values):
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": values,
        }
    )


def test_ma_cross_emits_strategy_tags():
    strat = MovingAverageCrossStrategy()
    history = make_history(list(range(1, 40)))
    signal = strat.on_bar({"history": history}, {})

    meta = signal.meta
    assert meta["strategy_name"] == "ma_cross_5_20"
    assert meta["strategy_version"] == "1"
    expected_hash = params_hash({"fast_period": 5, "slow_period": 20})
    assert meta["strategy_params_hash"] == expected_hash
    assert len(meta["strategy_params_hash"]) == 8
    assert meta["strategy_params"] == {"fast_period": 5, "slow_period": 20}

    repeat_signal = strat.on_bar({"history": history}, {})
    assert repeat_signal.meta["strategy_params_hash"] == expected_hash


def test_donchian_atr_emits_strategy_tags():
    strat = DonchianATRStrategy()
    values = list(range(1, 25)) + [60]
    history = make_history(values)
    signal = strat.on_bar({"history": history}, {})

    meta = signal.meta
    assert meta["strategy_name"] == "donchian_atr"
    assert meta["strategy_version"] == "1"
    assert "strategy_params_hash" in meta and len(meta["strategy_params_hash"]) == 8
    params = meta["strategy_params"]
    assert params["donchian_len"] == 20
    assert params["atr_len"] == 14


def test_strategy_tags_propagate_to_decision_records():
    base_meta = strategy_tags("demo_strategy", "7", {"foo": 1})
    signal = Signal(side="long", strength=1.0, meta=base_meta)

    tags = extract_strategy_tags(signal.meta)
    plan = {}
    if tags:
        plan.setdefault("strategy_tags", tags)
    decision_payload = {"decision": "ENTER_LONG"}
    decision_payload.update(plan.get("strategy_tags", {}))

    assert decision_payload["strategy_name"] == "demo_strategy"
    assert decision_payload["strategy_params_hash"] == params_hash({"foo": 1})


def test_reporter_includes_strategy_tags(tmp_path):
    reporter = Reporter(
        base_dir=str(tmp_path),
        environment="test",
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    tags = strategy_tags("demo_strategy", "1", {"alpha": 42})

    reporter.apply_strategy_tags(tags)
    reporter.log_trade("BTC/USDT", "buy", 100.0, 1.0)

    trades_path = tmp_path / "test" / "trades_2025-01-01.csv"
    with open(trades_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    last_row = rows[-1]
    assert last_row["strategy_name"] == "demo_strategy"
    assert last_row["strategy_version"] == "1"
    assert last_row["strategy_params_hash"] == params_hash({"alpha": 42})
    assert json.loads(last_row["strategy_params"]) == {"alpha": 42}

    reporter.apply_strategy_tags(tags)
    reporter.log_detailed_entry(
        "BTC/USDT",
        "long",
        100.0,
        1.0,
        95.0,
        1.0,
        atr_abs=1.5,
        sig={"fast_ma": 1.0, "slow_ma": 1.0, "regime": "UP"},
        equity_usdt=1000.0,
        reason="entry",
        reasons_list=["test"],
    )

    entries_path = tmp_path / "test" / "detailed_entries_2025-01-01.csv"
    with open(entries_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) >= 2
    header = rows[0]
    body = rows[1]
    header_index = {name: idx for idx, name in enumerate(header)}
    assert header_index["strategy_name"]
    assert body[header_index["strategy_name"]] == "demo_strategy"
    assert body[header_index["strategy_version"]] == "1"
    assert body[header_index["strategy_params_hash"]] == params_hash({"alpha": 42})
    assert json.loads(body[header_index["strategy_params"]]) == {"alpha": 42}
