import csv
import argparse
from pathlib import Path
from statistics import mean, median
from collections import defaultdict
from typing import Dict, List, Any


def discover_trade_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    files = sorted([p for p in base_dir.iterdir() if p.name.startswith("trades_") and p.suffix == ".csv"])
    return files


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def summary_from_pnls(pnls: List[float]) -> Dict[str, Any]:
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    total = len(pnls)
    win_rate = (len(wins) / total * 100) if total else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = float('inf') if gross_win > 0 else 0.0
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_pnl_pct": round(mean(pnls), 3) if pnls else 0.0,
        "median_pnl_pct": round(median(pnls), 3) if pnls else 0.0,
        "gross_win_pct": round(gross_win, 3),
        "gross_loss_pct": round(gross_loss, 3),
        "profit_factor": (round(pf, 3) if pf not in (float('inf'),) else "inf"),
    }


def build_reports(env: str, base: str = "data") -> Dict[str, Path]:
    base_dir = Path(base) / env
    trade_files = discover_trade_files(base_dir)

    # Aggregators
    pnls_by_day: Dict[str, List[float]] = defaultdict(list)
    pnls_by_symbol: Dict[str, List[float]] = defaultdict(list)
    pnls_by_day_symbol: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for f in trade_files:
        day = f.stem.replace("trades_", "")
        rows = load_rows(f)
        for r in rows:
            status = (r.get("status") or "").upper()
            if status != "CLOSED":
                continue
            symbol = r.get("symbol") or "?"
            try:
                pnl = float(r.get("qty") or 0)
            except Exception:
                pnl = 0.0
            pnls_by_day[day].append(pnl)
            pnls_by_symbol[symbol].append(pnl)
            pnls_by_day_symbol[day][symbol].append(pnl)

    # Write by-day report
    out_by_day = base_dir / "report_by_day.csv"
    with out_by_day.open("w", encoding="utf-8", newline="") as fh:
        headers = [
            "date", "trades", "wins", "losses", "win_rate_pct",
            "avg_pnl_pct", "median_pnl_pct", "gross_win_pct", "gross_loss_pct", "profit_factor"
        ]
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for day in sorted(pnls_by_day.keys()):
            s = summary_from_pnls(pnls_by_day[day])
            s_row = {"date": day, **s}
            w.writerow(s_row)

    # Write by-symbol report
    out_by_symbol = base_dir / "report_by_symbol.csv"
    with out_by_symbol.open("w", encoding="utf-8", newline="") as fh:
        headers = [
            "symbol", "trades", "wins", "losses", "win_rate_pct",
            "avg_pnl_pct", "median_pnl_pct", "gross_win_pct", "gross_loss_pct", "profit_factor"
        ]
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for sym in sorted(pnls_by_symbol.keys()):
            s = summary_from_pnls(pnls_by_symbol[sym])
            s_row = {"symbol": sym, **s}
            w.writerow(s_row)

    # Write by-day-and-symbol report
    out_by_day_symbol = base_dir / "report_by_day_symbol.csv"
    with out_by_day_symbol.open("w", encoding="utf-8", newline="") as fh:
        headers = [
            "date", "symbol", "trades", "wins", "losses", "win_rate_pct",
            "avg_pnl_pct", "median_pnl_pct", "gross_win_pct", "gross_loss_pct", "profit_factor"
        ]
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for day in sorted(pnls_by_day_symbol.keys()):
            for sym in sorted(pnls_by_day_symbol[day].keys()):
                s = summary_from_pnls(pnls_by_day_symbol[day][sym])
                w.writerow({"date": day, "symbol": sym, **s})

    return {
        "by_day": out_by_day,
        "by_symbol": out_by_symbol,
        "by_day_symbol": out_by_day_symbol,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate daily and symbol performance reports from trade logs.")
    ap.add_argument("env", choices=["testnet", "live"], help="Environment folder under base dir")
    ap.add_argument("--base", default="data", help="Base data directory (default: data)")
    args = ap.parse_args()

    paths = build_reports(args.env, args.base)
    print("Report files written:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()

