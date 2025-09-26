#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path


def main() -> int:
    # Healthy if bot.log updated in last 3 minutes
    path = Path("bot.log")
    if not path.exists():
        return 1
    try:
        mtime = path.stat().st_mtime
        if (time.time() - mtime) < 180:
            return 0
    except Exception:
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())

