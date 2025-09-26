import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import math
import pandas as pd

from baseline import DEFAULT_BASELINE_PATH, generate_baseline
from auto_trading_bot.reporter import generate_report


def _extract_trades_from_records(symbol_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] | None = None

    for r in symbol_records:
        ts = pd.to_datetime(r.get("ts"), utc=True)
        decision = r.get("decision") or ""
        side = (r.get("side") or "").lower()
        qty = float(r.get("qty") or 0.0)
        close_price = float(r.get("close") or 0.0)

        # Entry
        if open_trade is None and decision in ("ENTER_LONG", "ENTER_SHORT") and qty > 0:
            open_trade = {
                "side": side if side in ("long", "short") else ("long" if "LONG" in decision else "short"),
                "entry": close_price,
                "entry_ts": ts,
                "qty": qty,
            }
            continue

        # Exit
        exit_info = r.get("exit") or None
        if open_trade is not None and exit_info is not None:
            try:
                exit_price = float(exit_info.get("price") or 0.0)
            except Exception:
                exit_price = 0.0
            trade = {
                **open_trade,
                "exit": exit_price,
                "exit_ts": ts,
            }
            trades.append(trade)
            open_trade = None

    return trades


def _trades_from_baseline(baseline_obj: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, records in (baseline_obj.get("symbols") or {}).items():
        rows.extend(_extract_trades_from_records(records))
    return pd.DataFrame(rows)


def _to_jsonable_number(x: float) -> Any:
    if math.isinf(x):
        return "inf"
    if math.isnan(x):
        return 0.0
    return float(x)


def refresh_metrics(input_path: Path, output_path: Path, regen: bool = False) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Baseline missing: {input_path}")
    baseline_data = json.loads(input_path.read_text(encoding="utf-8"))

    if regen:
        timeframe = baseline_data.get("timeframe")
        bars = int(baseline_data.get("bars") or 180)
        equity = float(baseline_data.get("equity") or 20000.0)
        symbols = list((baseline_data.get("symbols") or {}).keys())
        if not symbols:
            raise ValueError("Baseline has no symbols to regenerate")
        baseline_data = generate_baseline(symbols=symbols, timeframe=timeframe, bars=bars, equity=equity, output_path=output_path)

    # Compute metrics from baseline trades
    trades_df = _trades_from_baseline(baseline_data)
    rep = generate_report(trades_df)
    metrics = rep.iloc[0]
    report_metrics = {
        "win_rate": _to_jsonable_number(float(metrics["win_rate"])) ,
        "expectancy": _to_jsonable_number(float(metrics["expectancy"])) ,
        "mdd": _to_jsonable_number(float(metrics["mdd"])) ,
        "profit_factor": _to_jsonable_number(float(metrics["profit_factor"])) ,
        "sharpe": _to_jsonable_number(float(metrics["sharpe"])) ,
    }

    baseline_data["report_metrics"] = report_metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    return baseline_data


def main():
    ap = argparse.ArgumentParser(description="Embed report metrics into baseline.json")
    ap.add_argument("--input", default=str(DEFAULT_BASELINE_PATH), help="Path to input baseline.json")
    ap.add_argument("--output", default=str(DEFAULT_BASELINE_PATH), help="Path to output baseline.json")
    ap.add_argument("--regen", action="store_true", help="Regenerate baseline decisions before computing metrics")
    args = ap.parse_args()

    updated = refresh_metrics(Path(args.input), Path(args.output), regen=args.regen)
    print(f"Updated baseline written to {args.output}")
    print("report_metrics:")
    for k, v in updated.get("report_metrics", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


