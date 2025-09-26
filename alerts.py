from __future__ import annotations

import os
import json
from typing import Optional, List, Dict, Any

import glob
import csv
import requests


class SlackNotifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = (webhook_url or "").strip()

    @classmethod
    def from_env(cls) -> "SlackNotifier":
        return cls(os.getenv("SLACK_WEBHOOK_URL"))

    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send(self, text: str) -> bool:
        if not self.enabled():
            return False
        try:
            payload = {"text": str(text)}
            resp = requests.post(self.webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=5)
            return 200 <= resp.status_code < 300
        except Exception as e:
            # Do not raise; caller must not crash bot on alert failure
            try:
                print(f"[WARN] Slack notify failed: {e}")
            except Exception:
                pass
            return False


_SLACK = None  # lazy singleton


def _get_notifier() -> SlackNotifier:
    global _SLACK
    if _SLACK is None:
        _SLACK = SlackNotifier.from_env()
    return _SLACK


def slack_notify_safely(message: str) -> bool:
    """Send a Slack message if configured. Never raises; returns True on success."""
    n = _get_notifier()
    return n.send(message)


def _collect_closed_trades(base_dir: str, env: str) -> List[Dict[str, Any]]:
    """Collect closed trades across data/<env>/trades_*.csv into simple dict rows.

    Each row includes side, entry (float), exit (float). Ignores incomplete rows.
    """
    root = os.path.join(base_dir, env)
    patterns = [os.path.join(root, "trades_*.csv")]
    files: List[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    out: List[Dict[str, Any]] = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8", newline="") as fh:
                r = csv.DictReader(fh)
                for row in r:
                    if (row.get("status") or "").upper() != "CLOSED":
                        continue
                    try:
                        entry = float(row.get("entry_price") or 0.0)
                        exitp = float(row.get("exit_price") or 0.0)
                    except Exception:
                        continue
                    if entry <= 0 or exitp <= 0:
                        continue
                    side = (row.get("side") or "").strip().lower()
                    try:
                        r_mult = row.get("R_multiple")
                        r_mult_val = float(r_mult) if r_mult not in ("", None) else None
                    except Exception:
                        r_mult_val = None
                    out.append({"side": side, "entry": entry, "exit": exitp, "R_multiple": r_mult_val})
        except Exception:
            continue
    return out


def _winrate_and_expectancy_from_logs(base_dir: str, env: str) -> Dict[str, float]:
    try:
        rows = _collect_closed_trades(base_dir, env)
        if not rows:
            return {"win_rate_pct": 0.0, "avg_r": 0.0}
        # Use reporter to compute metrics from trades
        try:
            import pandas as pd  # lazy import
            from reporter import generate_report
            df = pd.DataFrame(rows)
            rep = generate_report(df).iloc[0]
            win_rate_pct = float(rep.get("win_rate", 0.0)) * 100.0
            # Prefer true Avg R if provided; fallback to expectancy
            avg_r = float(rep.get("avg_r", 0.0))
            return {"win_rate_pct": win_rate_pct, "avg_r": avg_r}
        except Exception:
            return {"win_rate_pct": 0.0, "avg_r": 0.0}
    except Exception:
        return {"win_rate_pct": 0.0, "avg_r": 0.0}


def slack_notify_exit(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    pnl_usdt: float,
    pnl_pct: float,
    equity_usdt: float,
) -> bool:
    """Send a richly formatted Slack message about a position exit.

    Pulls cumulative win rate and average R (expectancy) from recent trade logs.
    Returns True on successful post, False otherwise. Never raises.
    """
    try:
        import config as cfg
        env = "testnet" if getattr(cfg, "TESTNET", True) else "live"
        base_dir = getattr(cfg, "DATA_BASE_DIR", "data")
    except Exception:
        env = "testnet"
        base_dir = "data"

    metrics = _winrate_and_expectancy_from_logs(base_dir, env)
    win_rate_pct = metrics.get("win_rate_pct", 0.0)
    avg_r = metrics.get("avg_r", 0.0)

    # Compose message
    side_disp = (side or "").upper()
    msg = (
        f"💥 EXIT {symbol} {side_disp}\n"
        f"Entry: {entry_price:.4f} → Exit: {exit_price:.4f}\n"
        f"PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%)\n"
        f"Win rate: {win_rate_pct:.2f}% | Avg R: {avg_r:.3f}\n"
        f"Equity: ${equity_usdt:,.2f}"
    )
    return slack_notify_safely(msg)
