import csv
from pathlib import Path
from datetime import datetime

from scripts.migrate_trades_schema import migrate_file


def test_migration_adds_schema_and_fields(tmp_path):
    # Create an old-format CSV
    date = datetime.utcnow().strftime("%Y-%m-%d")
    p = tmp_path / "data" / "testnet"
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"trades_{date}.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "timestamp","symbol","side","entry_price","qty","reason","status","exit_timestamp","exit_price","pnl_pct","exit_reason","risk_usdt","R_multiple"
        ])
        writer.writeheader()
        writer.writerow({
            "timestamp": "2025-01-01 00:00:00",
            "symbol": "BTC/USDT",
            "side": "short",
            "entry_price": "200",
            "qty": "1",
            "reason": "test",
            "status": "CLOSED",
            "exit_timestamp": "2025-01-01 01:00:00",
            "exit_price": "180",
            "pnl_pct": "10.0",
            "exit_reason": "test",
            "risk_usdt": "10",
            "R_multiple": "2.0",
        })

    ok = migrate_file(f)
    assert ok is True
    with f.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "no rows after migration"
    row = rows[0]
    # schema_version present
    assert row.get("schema_version") == "4"
    # new fields exist (may be blank)
    for key in ("entry_ts_utc","stop_basis","risk_usdt_planned","R_atr_expost"):
        assert key in row

