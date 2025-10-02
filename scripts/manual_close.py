#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a manual trade exit")
    parser.add_argument("symbol", help="Trading pair symbol, e.g. BTC/USDT")
    parser.add_argument("side", choices=["long", "short"], help="Side of the original position")
    parser.add_argument("exit_price", type=float, help="Executed exit price")
    parser.add_argument("pnl_pct", type=float, help="Realized PnL in percent")
    parser.add_argument("pnl_usd", type=float, help="Realized PnL in quote currency")
    parser.add_argument("qty", type=float, help="Executed quantity")
    parser.add_argument("entry_price", type=float, help="Recorded entry price")
    parser.add_argument("env", choices=["testnet", "live"], help="Environment to log against")
    parser.add_argument("data_dir", help="Base data directory (e.g. data)")
    parser.add_argument("timestamp", help="ISO timestamp for exit event (UTC)")
    parser.add_argument("manual_reason", help="Free-form reason for manual exit")
    parser.add_argument("trade_id", help="Unique identifier for this exit")
    parser.add_argument("fees_usd", type=float, help="Fees paid for this exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        from auto_trading_bot.reporter import Reporter
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import Reporter: {exc}", file=sys.stderr)
        return 2

    reporter = Reporter(base_dir=args.data_dir, environment=args.env)
    reporter.log_exit(
        args.symbol,
        args.side,
        exit_price=args.exit_price,
        pnl_pct=args.pnl_pct,
        reason="manual_exit",
        reason_code="manual_close",
        fees_quote_actual=args.fees_usd,
    )
    reporter.log_signal_analysis(
        args.symbol,
        current_price=args.exit_price,
        signal={"reason": args.manual_reason, "trade_id": args.trade_id},
        decision="MANUAL_EXIT",
    )
    reporter.save_report("manual_exit", {
        "symbol": args.symbol,
        "side": args.side,
        "exit_price": args.exit_price,
        "pnl_pct": args.pnl_pct,
        "pnl_quote": args.pnl_usd,
        "qty": args.qty,
        "entry_price": args.entry_price,
        "timestamp": args.timestamp,
        "reason": args.manual_reason,
        "trade_id": args.trade_id,
        "fees_quote": args.fees_usd,
    })
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

