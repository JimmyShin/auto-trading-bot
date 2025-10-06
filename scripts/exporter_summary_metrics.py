#!/usr/bin/env python
"""Prometheus exporter for latest summary metrics.

Loads reporting/out/<env>/latest-summary.json and exposes selected stats.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from prometheus_client import Gauge, start_http_server

LOGGER = logging.getLogger("summary_exporter")
DEFAULT_PORT = 9102
DEFAULT_INTERVAL = 10.0

METRIC_KEYS = {
    "trades_total": ("trades_total", 0.0),
    "wins_total": ("wins", 0.0),
    "pnl_sum": ("pnl_total", 0.0),
    "fees_sum": ("fees_total", 0.0),
    "equity": ("equity", 0.0),
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_summary(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to decode JSON at {path}: {exc}") from exc


def _resolve_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    stats = data.get("stats") or {}
    if not isinstance(stats, dict):
        return {}
    return stats


def _extract_value(stats: Dict[str, Any], key: str, default: float) -> float:
    value = stats.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - guard
        return default


def _update_metrics(
    gauges: Dict[str, Gauge],
    stats: Dict[str, Any],
    env: str,
) -> None:
    for metric_name, (stats_key, default) in METRIC_KEYS.items():
        gauge = gauges[metric_name]
        value = _extract_value(stats, stats_key, default)
        gauge.labels(env=env).set(value)


def run_exporter(
    *,
    env: str,
    summary_path: Path,
    port: int,
    interval: float,
    once: bool = False,
) -> None:
    gauges = {
        name: Gauge(name, f"{name.replace('_', ' ')} from latest summary", ["env"])
        for name in METRIC_KEYS
    }
    start_http_server(port)
    LOGGER.info("Exporter listening on :%s", port)
    LOGGER.info("Watching summary at %s", summary_path)

    while True:
        try:
            data = _load_summary(summary_path)
            stats = _resolve_stats(data)
            _update_metrics(gauges, stats, env)
            LOGGER.debug("Metrics updated: %s", {name: stats.get(key) for name, (key, _) in METRIC_KEYS.items()})
        except FileNotFoundError:
            LOGGER.warning("Summary not found at %s; emitting zeros", summary_path)
            for metric_name, gauge in gauges.items():
                gauge.labels(env=env).set(0.0)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to update metrics: %s", exc)
        if once:
            break
        time.sleep(interval)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latest summary metrics to Prometheus")
    parser.add_argument("--env", default=os.getenv("ENV", "testnet"), help="Environment label (default: %(default)s)")
    parser.add_argument(
        "--summary-root",
        default=os.getenv("SUMMARY_ROOT", "reporting/out"),
        help="Base directory containing <env>/latest-summary.json (default: %(default)s)",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("EXPORTER_PORT", DEFAULT_PORT)), help="Exporter port")
    parser.add_argument(
        "--interval", type=float, default=float(os.getenv("EXPORTER_INTERVAL", DEFAULT_INTERVAL)), help="Refresh interval seconds"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--once", action="store_true", help="Run a single refresh then exit (for tests)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    summary_path = Path(args.summary_root) / args.env / "latest-summary.json"
    LOGGER.info(
        "Starting summary exporter: env=%s port=%s interval=%ss summary=%s",
        args.env,
        args.port,
        args.interval,
        summary_path,
    )
    try:
        run_exporter(
            env=args.env,
            summary_path=summary_path,
            port=args.port,
            interval=args.interval,
            once=args.once,
        )
    except KeyboardInterrupt:
        LOGGER.info("Exporter interrupted; shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())