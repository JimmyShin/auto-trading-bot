"""Baseline data generation utilities for the trading bot."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config import ATR_LEN, TF, UNIVERSE
from indicators import atr
from strategy import DonchianATREngine

DATA_ROOT = Path("data") / "backtest" / "ohlcv"
DEFAULT_BASELINE_PATH = Path("data") / "testnet" / "baseline.json"


def _round(value: Any, digits: int) -> float:
    return round(float(value or 0.0), digits)


def _symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_offline_candles(symbol: str, timeframe: str, bars: Optional[int] = None) -> pd.DataFrame:
    """Load historical candles stored under data/backtest/ohlcv."""
    path = DATA_ROOT / timeframe / f"{_symbol_to_filename(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OHLCV data for {symbol} ({timeframe}): {path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"Unexpected OHLCV format in {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if bars is not None:
        df = df.tail(int(bars)).reset_index(drop=True)
    return df


def _serialise_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    alignment = signal.get("alignment") or {}
    candle_filter = signal.get("candle_filter") or {}
    risk_mult = signal.get("risk_multiplier") or {}
    return {
        "long": bool(signal.get("long")),
        "short": bool(signal.get("short")),
        "fast_ma": _round(signal.get("fast_ma"), 6),
        "slow_ma": _round(signal.get("slow_ma"), 6),
        "regime": signal.get("regime"),
        "alignment": {k: bool(v) for k, v in alignment.items()},
        "risk_multiplier": {k: _round(v, 4) for k, v in risk_mult.items()},
        "candle_filter": {
            "original_long": bool(candle_filter.get("original_long")),
            "original_short": bool(candle_filter.get("original_short")),
            "candle_safe_long": bool(candle_filter.get("candle_safe_long")),
            "candle_safe_short": bool(candle_filter.get("candle_safe_short")),
            "candle_position_ratio": _round(candle_filter.get("candle_position_ratio"), 4),
            "candle_range": _round(candle_filter.get("candle_range"), 4),
            "filter_reason": candle_filter.get("filter_reason"),
        },
    }


def _serialise_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return {}
    return {
        "in_position": bool(state.get("in_position")),
        "side": state.get("side"),
        "entry_price": _round(state.get("entry_price"), 4),
        "be_promoted": bool(state.get("be_promoted")),
        "pyramid_level": int(state.get("pyramid_level", 0)),
    }


def generate_baseline(
    symbols: Optional[Iterable[str]] = None,
    timeframe: str = TF,
    bars: int = 180,
    equity: float = 20000.0,
    output_path: Path = DEFAULT_BASELINE_PATH,
) -> Dict[str, Any]:
    """Generate baseline decisions for the provided symbols."""
    symbols = list(symbols or UNIVERSE)
    engine = DonchianATREngine(persist_state=False, initial_state={})
    engine.reset_daily_anchor(equity)

    baseline: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "equity": equity,
        "atr_length": ATR_LEN,
        "bars": bars,
        "symbols": {},
    }

    for symbol in symbols:
        df = load_offline_candles(symbol, timeframe, bars + ATR_LEN + 5)
        if len(df) < ATR_LEN + 5:
            raise ValueError(f"Not enough candles for {symbol} in {timeframe}")
        records: List[Dict[str, Any]] = []
        engine.state[symbol] = {}

        for idx in range(ATR_LEN, len(df)):
            window = df.iloc[: idx + 1]
            atr_abs = atr(window, ATR_LEN)
            if pd.isna(atr_abs) or float(atr_abs or 0) <= 0:
                continue

            candle = window.iloc[-1]
            price = float(candle["close"])

            state_snapshot = engine.state.get(symbol, {}).copy()
            if state_snapshot.get("in_position"):
                engine.update_after_move(symbol, float(atr_abs), price)
                state_snapshot = engine.state.get(symbol, {}).copy()

            engine.update_reset_tracking(symbol, window)
            can_reenter = engine.can_re_enter_after_stop(symbol)
            plan = engine.make_entry_plan(
                symbol=symbol,
                df=window,
                equity=equity,
                price=price,
                atr_abs=float(atr_abs),
                is_new_bar=True,
                can_reenter=can_reenter,
                funding_avoid=False,
                daily_loss_hit=False,
            )

            signal = plan.get("signal", {})
            serialised = {
                "ts": candle["ts"].isoformat(),
                "open": _round(candle["open"], 4),
                "high": _round(candle["high"], 4),
                "low": _round(candle["low"], 4),
                "close": _round(price, 4),
                "volume": _round(candle.get("volume"), 4),
                "atr": _round(atr_abs, 6),
                "decision": plan.get("decision"),
                "side": plan.get("side"),
                "qty": _round(plan.get("qty"), 6),
                "stop_price": _round(plan.get("stop_price"), 4),
                "risk_multiplier": _round(plan.get("risk_multiplier"), 4),
                "skip_reason": plan.get("skip_reason"),
                "reasons": list(plan.get("reasons", [])),
                "signal": _serialise_signal(signal),
                "state": _serialise_state(state_snapshot),
                "exit": None,
            }

            position_after = engine.state.get(symbol, {}).copy()
            if position_after.get("in_position"):
                trail_stop = engine.trail_stop_price(
                    position_after.get("side"),
                    float(position_after.get("entry_price", price)),
                    price,
                    float(atr_abs),
                    bool(position_after.get("be_promoted")),
                )
                serialised["state"]["trail_stop"] = _round(trail_stop, 4)

            decision = plan.get("decision")
            if decision in ("ENTER_LONG", "ENTER_SHORT"):
                qty = float(plan.get("qty") or 0)
                if qty > 0:
                    engine.update_symbol_state_on_entry(symbol, plan.get("side"), price, qty)

            # Check for stop-out based on current candle extremes
            current_state = engine.state.get(symbol, {}).copy()
            if current_state.get("in_position"):
                side = current_state.get("side")
                trail_stop = engine.trail_stop_price(
                    side,
                    float(current_state.get("entry_price", price)),
                    price,
                    float(atr_abs),
                    bool(current_state.get("be_promoted")),
                )
                hit_stop = False
                if side == "long" and float(candle["low"]) <= trail_stop:
                    hit_stop = True
                elif side == "short" and float(candle["high"]) >= trail_stop:
                    hit_stop = True
                if hit_stop:
                    engine.record_stop_loss_exit(symbol, side, trail_stop)
                    serialised["exit"] = {
                        "type": "stop",
                        "price": _round(trail_stop, 4),
                    }

            records.append(serialised)

        baseline["symbols"][symbol] = records[-bars:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate baseline trading signals.")
    parser.add_argument("--symbols", nargs="*", help="Symbols to include (default: config.UNIVERSE)")
    parser.add_argument("--timeframe", default=TF, help="Timeframe to analyse")
    parser.add_argument("--bars", type=int, default=180, help="Number of bars to keep in baseline")
    parser.add_argument("--equity", type=float, default=20000.0, help="Equity to assume for sizing")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BASELINE_PATH), help="Path to write baseline JSON")
    args = parser.parse_args()

    symbols = args.symbols or UNIVERSE
    generate_baseline(symbols=symbols, timeframe=args.timeframe, bars=args.bars, equity=args.equity, output_path=Path(args.output))
    print(f"Baseline written to {args.output}")


if __name__ == "__main__":
    main()
