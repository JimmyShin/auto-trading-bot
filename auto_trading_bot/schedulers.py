from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from auto_trading_bot import alerts, reporter

logger = logging.getLogger(__name__)


def daily_job(env: str = "testnet") -> None:
    summary = reporter.load_latest_summary(env)
    metrics = reporter.collect_runtime_metrics(env)
    if not summary:
        logger.info("Daily snapshot summary missing for env=%s; using defaults", env)
    alerts.send_daily_snapshot(env, summary, metrics)


def weekly_job(env: str = "testnet") -> None:
    stats = reporter.compile_weekly_summary(env)
    if not stats:
        logger.info("Weekly summary stats missing for env=%s; using defaults", env)
    alerts.send_weekly_summary(env, stats)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trading bot reporting schedulers")
    parser.add_argument("job", choices=["daily", "weekly"], help="Scheduler job to execute")
    parser.add_argument("--env", default="testnet", help="Environment label (default: testnet)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    if args.job == "daily":
        daily_job(args.env)
    else:
        weekly_job(args.env)


if __name__ == "__main__":
    main()
