#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request


def _fetch_metrics(port: int) -> str:
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError):
        return ""
    return data


def _heartbeat_fresh(metrics_text: str, max_age_sec: int) -> bool:
    if not metrics_text:
        return False
    for line in metrics_text.splitlines():
        if line.startswith("bot_heartbeat_ts"):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[-1])
            except ValueError:
                continue
            return (time.time() - ts) <= max_age_sec
    return False


def main() -> int:
    port = int(os.getenv("METRICS_PORT", os.getenv("OBS_METRICS_PORT", "9108")))
    metrics_text = _fetch_metrics(port)
    if not _heartbeat_fresh(metrics_text, max_age_sec=900):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
