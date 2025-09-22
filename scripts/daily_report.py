import csv
import argparse
from pathlib import Path
from statistics import mean, median


def load_trades(trades_path: Path):
    rows = []
    if not trades_path.exists():
        return rows
    with trades_path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append(row)
    return rows


def parse_side(side: str) -> str:
    s = (side or "").upper()
    if "LONG" in s:
        return "LONG"
    if "SHORT" in s:
        return "SHORT"
    return s


def pair_trades(trade_rows):
    """Very simple pairing by symbol: first OPEN then next CLOSED.
    Assumes one open per symbol at a time.
    Returns list of dicts with entry/exit and pnl_pct.
    """
    open_by_symbol = {}
    pairs = []
    for row in trade_rows:
        symbol = row.get("symbol")
        side = row.get("side") or ""
        status = row.get("status") or ""
        if status.upper() == "OPEN":
            # overwrite any stale
            open_by_symbol[symbol] = {
                "symbol": symbol,
                "side": parse_side(side),
                "entry_price": float(row.get("entry_price") or 0),
                "qty": float(row.get("qty") or 0),
                "ts": row.get("timestamp"),
            }
        elif status.upper() == "CLOSED":
            o = open_by_symbol.get(symbol)
            # In CLOSED rows, our schema logs: entry_price=exit_price, qty=pnl_pct
            exit_price = float(row.get("entry_price") or 0)
            pnl_pct = float(row.get("qty") or 0)
            pairs.append({
                "symbol": symbol,
                "open": o,
                "close": {
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "ts": row.get("timestamp"),
                }
            })
            # clear open
            open_by_symbol.pop(symbol, None)
    return pairs


def summarize(pairs):
    # use pnl_pct for win/loss stats
    pnls = [p["close"].get("pnl_pct") for p in pairs if p and p.get("close") is not None]
    pnls = [float(x) for x in pnls if x is not None]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    total = len(pnls)
    win_rate = (len(wins) / total * 100) if total else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf') if gross_win > 0 else 0.0
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_pnl_pct": round(mean(pnls), 3) if pnls else 0.0,
        "median_pnl_pct": round(median(pnls), 3) if pnls else 0.0,
        "gross_win_pct": round(gross_win, 3),
        "gross_loss_pct": round(gross_loss, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor not in (float('inf'),) else "inf",
    }


def generate_report(date: str, env: str, base_dir: str = "data"):
    base = Path(base_dir) / env
    trades_path = base / f"trades_{date}.csv"
    rows = load_trades(trades_path)
    pairs = pair_trades(rows)
    summary = summarize(pairs)

    out_path = base / f"report_{date}.csv"
    headers = list(summary.keys())
    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        if write_header:
            w.writeheader()
        w.writerow(summary)
    return out_path, summary


def main():
    ap = argparse.ArgumentParser(description="Generate daily trading report from CSV logs.")
    ap.add_argument("date", help="Date in YYYY-MM-DD")
    ap.add_argument("env", choices=["testnet", "live"], help="Environment folder")
    ap.add_argument("--base", default="data", help="Base data directory (default: data)")
    args = ap.parse_args()

    out_path, summary = generate_report(args.date, args.env, args.base)

    print(f"Report written: {out_path}")
    print(", ".join([f"{k}={summary[k]}" for k in summary.keys()]))


if __name__ == "__main__":
    main()
