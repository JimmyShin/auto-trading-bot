#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List


SCHEMA_VERSION = 4


def _ensure_fields(row: Dict[str, str], fieldnames: List[str]) -> Dict[str, str]:
    for f in fieldnames:
        row.setdefault(f, "")
    return row


def migrate_file(path: Path) -> bool:
    """Migrate a single trades_*.csv file to schema v4.

    - Adds schema_version column
    - Adds new RAW snapshot columns if absent (left blank if unknown)
    - Leaves existing values untouched
    """
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            orig_fields = list(reader.fieldnames or [])
    except Exception:
        return False

    # Target field order aligned with reporter._trade_fieldnames
    target_fields = [
        "schema_version",
        "timestamp",
        "entry_ts_utc",
        "symbol",
        "side",
        "timeframe",
        "leverage",
        "entry_price",
        "qty",
        "reason",
        "entry_atr_abs",
        "atr_period",
        "atr_source",
        "stop_basis",
        "stop_k",
        "fallback_pct",
        "stop_distance",
        "risk_usdt_planned",
        "status",
        "exit_timestamp",
        "exit_ts_utc",
        "exit_price",
        "pnl_pct",
        "exit_reason",
        "fees_quote_actual",
        "pnl_quote_expost",
        "R_atr_expost",
        "R_usd_expost",
        "risk_usdt",
        "R_multiple",
    ]

    migrated: List[Dict[str, str]] = []
    for r in rows:
        # Provide defaults; assume ATR basis if unknown but leave values blank
        r.setdefault("schema_version", str(SCHEMA_VERSION))
        # duplicate legacy timestamp if present
        if r.get("timestamp") and not r.get("entry_ts_utc"):
            r["entry_ts_utc"] = r["timestamp"]
        # leave stop_basis empty if unknown; downstream handles N/A
        migrated.append(_ensure_fields(r, target_fields))

    try:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=target_fields)
            writer.writeheader()
            for r in migrated:
                writer.writerow({f: r.get(f, "") for f in target_fields})
        return True
    except Exception:
        # attempt to restore file on failure
        try:
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=orig_fields)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        except Exception:
            pass
        return False


def migrate_all(base_dir: str) -> int:
    base = Path(base_dir)
    count = 0
    for env in ("testnet", "live"):
        root = base / env
        if not root.exists():
            continue
        for p in sorted(root.glob("trades_*.csv")):
            if migrate_file(p):
                count += 1
    return count


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Migrate trades CSVs to schema v4")
    ap.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    args = ap.parse_args()
    updated = migrate_all(args.data_dir)
    print(f"Migrated {updated} trade files under {args.data_dir}")


if __name__ == "__main__":
    main()

