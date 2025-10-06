#!/usr/bin/env python3
"""Lightweight replay helpers for regenerating analytics artifacts.

This module intentionally avoids any live trading side-effects. It simply
re-runs the reporting pipeline so that trade logs and summaries reflect the
latest persisted CSV data (useful after structural logging fixes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from reporting import report_metrics


def replay_equity(env: str = "testnet", *, out_dir: Optional[Path] = None, min_trades: int = 1) -> None:
    """Re-run the reporting pipeline for the requested environment.

    Parameters
    ----------
    env:
        Environment name (e.g. "testnet", "live").
    out_dir:
        Root directory where reporting artifacts should be written. Defaults to
        ``Path("reporting") / "out"`` which mirrors the CLI convention.
    min_trades:
        Minimum closed trades required for the report to be considered
        decision-ready. Defaults to 1 so that fresh logs produce output even
        with small sample sizes.
    """

    target_root = Path(out_dir) if out_dir is not None else Path("reporting") / "out"
    target_root.mkdir(parents=True, exist_ok=True)

    argv = ["--env", env, "--out-dir", str(target_root), "--min-trades", str(min_trades)]
    exit_code = report_metrics.main(argv)
    if exit_code not in (0, None):  # pragma: no cover - informational only
        raise SystemExit(exit_code)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay equity summaries for trade logs")
    parser.add_argument("--env", default="testnet", help="Environment to rebuild (default: testnet)")
    parser.add_argument(
        "--out-dir",
        default=str(Path("reporting") / "out"),
        help="Output directory root (default: reporting/out)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum trades required for summary readiness (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    replay_equity(env=args.env, out_dir=Path(args.out_dir), min_trades=args.min_trades)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
