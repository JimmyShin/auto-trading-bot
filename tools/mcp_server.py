"""
Simple MCP server for donchian-atr-bot.

Exposes tools to inspect bot state and optionally control safe actions.

Core tools
- get_state(): Return JSON from state.json if present
- list_csv(kind, limit): List recent trades/signals CSVs
- read_file(path, max_bytes): Read a text file under project root

Automation tools (trading-aware)
- context7(): High-level 7-part context snapshot
- get_status(): Exchange, equity, config summary
- scan_signals(universe, timeframe, lookback): Signal snapshot per symbol
- calc_entry(symbol, side, price): Risk-based qty, stop, risk amount
- sync_state(): Reconcile state.json with actual positions
- start_bot(): Spawn trading loop (guarded)
- stop_bot(pid): Stop spawned loop by PID (guarded)
- emergency_close_all(confirm): Close positions + cancel orders (guarded)

Run
  python mcp_server.py

Then register this server in your MCP client (e.g., Claude Desktop).
See MCP_SETUP.md for details.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

try:
    # FastMCP is a lightweight server wrapper from the `mcp` package
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover - guidance for users without deps
    raise SystemExit(
        "The 'mcp' package is required. Install with: pip install mcp\n"
        f"Import error: {e}"
    )


ROOT = Path(__file__).resolve().parent.parent
ALLOW_MCP_TRADING = os.getenv("ALLOW_MCP_TRADING", "false").lower() == "true"
app = FastMCP("donchian-atr-bot")


@app.tool()
def get_state() -> dict:
    """Return the current bot state from `state.json` (if it exists)."""
    path = ROOT / "state.json"
    if not path.exists():
        return {"ok": False, "error": "state.json not found"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"failed to read state.json: {e}"}


@app.tool()
def list_csv(kind: Literal["trades", "signals"], limit: int = 5) -> list[str]:
    """List recent CSV files.

    - kind: "trades" or "signals"
    - limit: number of filenames to return (default 5)
    """
    if limit <= 0:
        return []

    pattern = {
        "trades": "trades_*.csv",
        "signals": "signal_analysis_*.csv",
    }[kind]

    files = sorted(
        ROOT.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [f.name for f in files[:limit]]


@app.tool()
def read_file(path: str, max_bytes: int = 100_000) -> str:
    """Read a text file under the project root, returning up to `max_bytes`.

    Guards against path traversal and directory reads.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    p = (ROOT / path).resolve()
    # Ensure the resolved path stays within the project root
    if not str(p).startswith(str(ROOT)):
        raise ValueError("path is outside project root")
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p.relative_to(ROOT)}")
    if p.is_dir():
        raise IsADirectoryError("path is a directory")

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes)
        return data
    except Exception as e:  # defensive: surface readable errors to the client
        return f"<error reading file: {e}>"


@app.tool()
def summarize_skips(days: int = 1) -> dict:
    """Summarize decisions and skip reasons from recent signal_analysis_*.csv files.

    Returns counts of decision and skip_reason to help tune strictness.
    """
    import csv
    from datetime import datetime, timedelta

    def daterange(n):
        today = datetime.utcnow().date()
        for i in range(n):
            yield (today - timedelta(days=i)).isoformat()

    files: List[Path] = []
    for d in daterange(max(1, days)):
        files.extend(sorted((ROOT / f"data/testnet").glob(f"signal_analysis_{d}.csv")))

    counts_decision: Dict[str, int] = {}
    counts_skip: Dict[str, int] = {}
    for p in files:
        try:
            with p.open("r", encoding="utf-8", newline="") as fh:
                r = csv.DictReader(fh)
                for row in r:
                    dec = (row.get("decision") or "").strip()
                    sk = (row.get("skip_reason") or "").strip()
                    if dec:
                        counts_decision[dec] = counts_decision.get(dec, 0) + 1
                    if sk:
                        counts_skip[sk] = counts_skip.get(sk, 0) + 1
        except Exception:
            continue
    return {"decisions": counts_decision, "skip_reasons": counts_skip}


@app.tool()
def context7() -> dict:
    """High-level 7-part context snapshot: config, env, equity, state, last trades, signals, logs."""
    from config import TF, LOOKBACK, UNIVERSE, RISK_PCT
    try:
        return {
            "tf": TF,
            "lookback": LOOKBACK,
            "universe": list(UNIVERSE),
            "risk_pct": RISK_PCT,
            "recent_trades": list_csv("trades", 5),
            "recent_signals": list_csv("signals", 5),
            "state": get_state(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _load_runtime():
    from config import TF, LOOKBACK, ATR_LEN, ATR_STOP_K, RISK_PCT, LEVERAGE
    from broker_binance import BinanceUSDM
    from strategy import DonchianATREngine
    from indicators import atr as atr_func
    return UNIVERSE, TF, LOOKBACK, BinanceUSDM, DonchianATREngine, atr_func


@app.tool()
def get_status() -> dict:
    try:
        UNIVERSE, TF, LOOKBACK, BinanceUSDM, _, _ = _load_runtime()
        b = BinanceUSDM(); b.load_markets()
        return {
            "universe": list(UNIVERSE),
            "timeframe": TF,
            "lookback": LOOKBACK,
            "equity_usdt": b.get_equity_usdt(),
            "exchange": "binance_usdm",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.tool()
def scan_signals(universe: Optional[List[str]] = None, timeframe: Optional[str] = None, lookback: Optional[int] = None) -> dict:
    try:
        from config import TF, LOOKBACK
        from broker_binance import BinanceUSDM
        from strategy import DonchianATREngine
        import pandas as pd
        syms = universe or []
        if not syms:
            from config import UNIVERSE
            syms = list(UNIVERSE)
        tf = timeframe or TF
        lb = int(lookback or LOOKBACK)
        b = BinanceUSDM(); b.load_markets()
        out: Dict[str, Any] = {}
        for symbol in syms:
            try:
                ohlcv = b.fetch_ohlcv(symbol, tf, limit=lb)
                df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
                eng = DonchianATREngine(persist_state=False, initial_state={})
                out[symbol] = eng.detect_signal(df)
            except Exception as e:
                out[symbol] = {"error": str(e)}
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.tool()
def calc_entry(symbol: str, side: str, price: Optional[float] = None) -> dict:
    try:
        from config import TF, LOOKBACK, ATR_LEN, ATR_STOP_K, RISK_PCT, LEVERAGE
        from broker_binance import BinanceUSDM
        from strategy import DonchianATREngine
        from indicators import atr as atr_func
        import pandas as pd
        ohlcv = b.fetch_ohlcv(symbol, TF, limit=LOOKBACK)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        last_close = float(df["close"].iloc[-1])
        px = float(price or last_close)
        atr_abs = float(atr_func(df, n=ATR_LEN)) if hasattr(atr_func, "__call__") else 0.0

        eng = DonchianATREngine()
        equity = b.get_equity_usdt()
        # Risk multiplier from signal/candle position
        sig = eng.detect_signal(df)
        risk_mult = sig.get("risk_multiplier", {}).get("long" if side == "long" else "short", 1.0)
        adjusted_risk_pct = RISK_PCT * float(risk_mult)
        qty = eng.calc_qty_by_risk_adjusted(equity, px, atr_abs, LEVERAGE, symbol, adjusted_risk_pct)
        stop_price = px - max(ATR_STOP_K * atr_abs, px * 0.004) if side == "long" else px + max(ATR_STOP_K * atr_abs, px * 0.004)
        risk_amount = abs(qty * (px - stop_price))
        return {
            "symbol": symbol,
            "side": side,
            "price": px,
            "qty": qty,
            "stop_price": stop_price,
            "risk_amount": risk_amount,
            "atr": atr_abs,
            "risk_multiplier": risk_mult,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.tool()
def sync_state() -> dict:
    """Reconcile state.json with actual positions for all symbols. Returns changes."""
    try:
        UNIVERSE, _, _, BinanceUSDM, DonchianATREngine, _ = _load_runtime()
        b = BinanceUSDM(); b.load_markets()
        eng = DonchianATREngine()
        changes = {}
        for sym in UNIVERSE:
            res = eng.sync_position_state(b, sym)
            changes[sym] = res
        return {"ok": True, "changes": changes}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _guarded() -> Optional[str]:
    if not ALLOW_MCP_TRADING:
        return "Trading actions disabled. Set ALLOW_MCP_TRADING=true to enable."
    return None


@app.tool()
def start_bot() -> dict:
    """Spawn the trading loop in a separate process. Returns PID. Guarded by ALLOW_MCP_TRADING."""
    msg = _guarded()
    if msg:
        return {"ok": False, "error": msg}
    try:
        py = sys.executable or "python"
        # Detach process on Windows/Linux
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x00000008  # CREATE_NEW_CONSOLE
        p = subprocess.Popen([py, "main.py"], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, creationflags=creationflags)
        return {"ok": True, "pid": p.pid}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.tool()
def stop_bot(pid: int) -> dict:
    """Terminate a previously started bot process by PID. Guarded by ALLOW_MCP_TRADING."""
    msg = _guarded()
    if msg:
        return {"ok": False, "error": msg}
    try:
        if os.name == "nt":
            subprocess.check_call(["taskkill", "/PID", str(pid), "/F"])
        else:
            os.kill(pid, 15)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.tool()
def emergency_close_all(confirm: bool = False) -> dict:
    """Cancel all orders and close open positions for configured UNIVERSE. Guarded.

    Requires confirm=true and ALLOW_MCP_TRADING=true.
    """
    msg = _guarded()
    if msg:
        return {"ok": False, "error": msg}
    if not confirm:
        return {"ok": False, "error": "confirm=true required"}
    try:
        UNIVERSE, _, _, BinanceUSDM, DonchianATREngine, _ = _load_runtime()
        b = BinanceUSDM(); b.load_markets()
        eng = DonchianATREngine()
        results = {}
        for symbol in UNIVERSE:
            try:
                b.cancel_all(symbol)
                pos = b.position_for(symbol)
                if pos:
                    contracts = float(pos.get('contracts') or pos.get('positionAmt') or 0)
                    if abs(contracts) > 0:
                        side = "sell" if contracts > 0 else "buy"
                        b.exchange.create_market_order(symbol, side, abs(contracts), None, {"reduceOnly": True})
                        eng.clear_position_state(symbol)
                        results[symbol] = "closed"
                    else:
                        results[symbol] = "no_position"
                else:
                    results[symbol] = "no_position"
            except Exception as e:
                results[symbol] = f"error: {e}"
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    import traceback, time, asyncio
    try:
        print("DEBUG: ROOT =", ROOT)
        import sys
        print("DEBUG: sys.path[0] =", sys.path[0])

        # app.list_tools() is async -> run it synchronously here to inspect actual tools
        try:
            coro = None
            if hasattr(app, "list_tools"):
                coro = app.list_tools()
            elif hasattr(app, "_tool_manager"):
                # fallback: try to access internal tool manager sync info later
                pass

            if coro is not None:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    tools_list = loop.run_until_complete(coro)
                    print("DEBUG: app.list_tools() ->", tools_list)
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
            else:
                print("DEBUG: no app.list_tools coroutine available; attempting alternate introspect")
                tm = getattr(app, "_tool_manager", getattr(app, "tools", None))
                print("DEBUG: tool manager object type:", type(tm))
        except Exception as e:
            print("DEBUG: exception while enumerating tools:", e)
            traceback.print_exc()

        print("DEBUG: starting FastMCP app.run(); waiting for client (stdio)")
        app.run()
    except KeyboardInterrupt:
        print("DEBUG: KeyboardInterrupt received; shutting down cleanly")
    except Exception as e:
        print("DEBUG: unexpected error in __main__:", e)
        traceback.print_exc()
    finally:
        time.sleep(0.05)
        print("DEBUG: exiting")

