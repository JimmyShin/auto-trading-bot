import csv
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, List, Any, Tuple


BASE_DIR = Path("data/testnet")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def discover_files(base: Path) -> Dict[str, List[Path]]:
    kinds = {
        "trades": [],
        "signals": [],
        "entries": [],
    }
    if not base.exists():
        return kinds
    for p in base.iterdir():
        if not p.name.endswith(".csv"):
            continue
        name = p.name
        if name.startswith("trades_"):
            kinds["trades"].append(p)
        elif name.startswith("signal_analysis_"):
            kinds["signals"].append(p)
        elif name.startswith("detailed_entries_"):
            kinds["entries"].append(p)
    # sort by date in filename
    for k in kinds:
        kinds[k].sort()
    return kinds


def summarize_trades(trade_files: List[Path]) -> Dict[str, Any]:
    total_open = 0
    total_closed = 0
    pnls: List[float] = []
    by_day: Dict[str, Dict[str, Any]] = {}
    for f in sorted(trade_files):
        rows = read_csv_rows(f)
        open_rows = [r for r in rows if (r.get("status", "").upper() == "OPEN")]
        closed_rows = [r for r in rows if (r.get("status", "").upper() == "CLOSED")]
        # schema: CLOSED uses entry_price=exit_price, qty=pnl_pct
        day_pnls: List[float] = []
        for r in closed_rows:
            try:
                day_pnls.append(float(r.get("qty") or 0))
            except Exception:
                pass
        by_day[f.stem] = {
            "open": len(open_rows),
            "closed": len(closed_rows),
            "realized_pnl_pct_sum": round(sum(day_pnls), 4),
        }
        total_open += len(open_rows)
        total_closed += len(closed_rows)
        pnls.extend(day_pnls)

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    return {
        "files": [f.name for f in sorted(trade_files)],
        "total_open": total_open,
        "total_closed": total_closed,
        "closed_win_rate_pct": round(win_rate, 2) if pnls else 0.0,
        "avg_closed_pnl_pct": round(mean(pnls), 4) if pnls else 0.0,
        "median_closed_pnl_pct": round(median(pnls), 4) if pnls else 0.0,
        "gross_win_pct": round(sum(wins), 4) if pnls else 0.0,
        "gross_loss_pct": round(abs(sum(losses)), 4) if pnls else 0.0,
        "profit_factor": (round(profit_factor, 4) if profit_factor not in (float("inf"),) else "inf"),
        "by_day": by_day,
    }


def equity_metrics_from_signals(signal_files: List[Path]) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    series: List[Tuple[str, float]] = []  # (timestamp, equity)
    per_file_stats: Dict[str, Dict[str, Any]] = {}

    for f in sorted(signal_files):
        rows = read_csv_rows(f)
        # We will take (timestamp, equity_usdt) where available
        day_series: List[Tuple[str, float]] = []
        for r in rows:
            ts = r.get("timestamp") or ""
            eq_raw = r.get("equity_usdt")
            if eq_raw is None or eq_raw == "":
                continue
            try:
                eq = float(eq_raw)
            except Exception:
                continue
            series.append((ts, eq))
            day_series.append((ts, eq))
        if day_series:
            start = day_series[0][1]
            end = day_series[-1][1]
            high = max(x for _, x in day_series)
            low = min(x for _, x in day_series)
            # compute simple max drawdown in this day's series
            peak = day_series[0][1]
            max_dd = 0.0
            for _, v in day_series:
                if v > peak:
                    peak = v
                dd = (peak - v)
                if dd > max_dd:
                    max_dd = dd
            per_file_stats[f.name] = {
                "start": round(start, 4),
                "end": round(end, 4),
                "change": round(end - start, 4),
                "min": round(low, 4),
                "max": round(high, 4),
                "max_drawdown_abs": round(max_dd, 4),
            }

    overall = {}
    if series:
        start = series[0][1]
        end = series[-1][1]
        high = max(x for _, x in series)
        low = min(x for _, x in series)
        peak = series[0][1]
        max_dd = 0.0
        for _, v in series:
            if v > peak:
                peak = v
            dd = (peak - v)
            if dd > max_dd:
                max_dd = dd
        overall = {
            "start": round(start, 4),
            "end": round(end, 4),
            "change": round(end - start, 4),
            "min": round(low, 4),
            "max": round(high, 4),
            "max_drawdown_abs": round(max_dd, 4),
        }

    return series, {"per_file": per_file_stats, "overall": overall}


def summarize_signals(signal_files: List[Path]) -> Dict[str, Any]:
    decisions = Counter()
    skip_reasons = Counter()
    per_file_counts: Dict[str, Dict[str, int]] = {}

    for f in sorted(signal_files):
        rows = read_csv_rows(f)
        dc = Counter()
        for r in rows:
            dec = (r.get("decision") or "").strip()
            if dec:
                decisions[dec] += 1
                dc[dec] += 1
            if dec.upper() == "SKIP":
                reason = (r.get("skip_reason") or "").strip() or "(none)"
                skip_reasons[reason] += 1
        per_file_counts[f.name] = dict(dc)

    eq_series, eq_stats = equity_metrics_from_signals(signal_files)

    return {
        "files": [f.name for f in sorted(signal_files)],
        "decisions_total": dict(decisions),
        "decisions_by_file": per_file_counts,
        "skip_reasons_top": dict(skip_reasons.most_common(10)),
        "equity": eq_stats,
    }


def summarize_entries(entry_files: List[Path]) -> Dict[str, Any]:
    risk_pcts: List[float] = []
    stop_dists: List[float] = []
    atr_vals: List[float] = []
    risk_mults: Counter = Counter()
    symbols: Counter = Counter()
    reasons: Counter = Counter()
    entries_by_file: Dict[str, int] = {}

    for f in sorted(entry_files):
        rows = read_csv_rows(f)
        entries_by_file[f.name] = len(rows)
        for r in rows:
            symbols[(r.get("symbol") or "")] += 1
            reasons[(r.get("reason") or "")] += 1
            try:
                rp = float(r.get("risk_pct") or 0)
                sd = float(r.get("stop_distance_pct") or 0)
                atr = float(r.get("atr_abs") or 0)
                rm = float(r.get("risk_multiplier") or r.get("risk_multiplier_long") or 0)
            except Exception:
                continue
            risk_pcts.append(rp)
            stop_dists.append(sd)
            atr_vals.append(atr)
            risk_mults[rm] += 1

    summary = {
        "files": [f.name for f in sorted(entry_files)],
        "entries_by_file": entries_by_file,
        "symbols_top": dict(symbols.most_common(10)),
        "reasons_top": dict(reasons.most_common(10)),
        "risk_pct_avg": round(mean(risk_pcts), 4) if risk_pcts else 0.0,
        "risk_pct_median": round(median(risk_pcts), 4) if risk_pcts else 0.0,
        "stop_distance_pct_avg": round(mean(stop_dists), 4) if stop_dists else 0.0,
        "atr_abs_avg": round(mean(atr_vals), 4) if atr_vals else 0.0,
        "risk_multiplier_dist": {str(k): v for k, v in risk_mults.most_common()},
    }
    return summary


def main():
    files = discover_files(BASE_DIR)
    print("== Files discovered ==")
    for k, lst in files.items():
        print(f"{k}: {len(lst)} files")
        for p in lst:
            print(f"  - {p.name}")

    print("\n== Trades summary ==")
    t_sum = summarize_trades(files["trades"])
    for k in ("total_open", "total_closed", "closed_win_rate_pct", "avg_closed_pnl_pct", "median_closed_pnl_pct", "gross_win_pct", "gross_loss_pct", "profit_factor"):
        print(f"{k}: {t_sum.get(k)}")
    print("by_day:")
    for day, stats in t_sum["by_day"].items():
        print(f"  {day}: open={stats['open']}, closed={stats['closed']}, realized_pnl_pct_sum={stats['realized_pnl_pct_sum']}")

    print("\n== Signal summary ==")
    s_sum = summarize_signals(files["signals"])
    print("decisions_total:")
    for k, v in s_sum["decisions_total"].items():
        print(f"  {k}: {v}")
    print("skip_reasons_top:")
    for k, v in s_sum["skip_reasons_top"].items():
        print(f"  {k}: {v}")
    print("equity overall:")
    for k, v in s_sum["equity"].get("overall", {}).items():
        print(f"  {k}: {v}")
    print("equity by file:")
    for fname, stats in s_sum["equity"].get("per_file", {}).items():
        print(f"  {fname}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    print("\n== Entry detail summary ==")
    e_sum = summarize_entries(files["entries"])
    for k in ("risk_pct_avg", "risk_pct_median", "stop_distance_pct_avg", "atr_abs_avg"):
        print(f"{k}: {e_sum.get(k)}")
    print("risk_multiplier_dist:")
    for k, v in e_sum["risk_multiplier_dist"].items():
        print(f"  {k}: {v}")
    print("symbols_top:")
    for k, v in e_sum["symbols_top"].items():
        print(f"  {k}: {v}")
    print("reasons_top:")
    for k, v in e_sum["reasons_top"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

