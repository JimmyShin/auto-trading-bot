import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def set_ohlcv_dir_for_ci():
    # On CI, use bundled sample OHLCV fixtures for baseline regeneration
    ci = os.getenv("CI", "").lower() == "true"
    prev_allow = os.getenv("ALLOW_SYNTHETIC_BASELINE")

    if ci:
        os.environ.setdefault("ALLOW_SYNTHETIC_BASELINE", "1")

    yield

    if ci:
        if prev_allow is None:
            os.environ.pop("ALLOW_SYNTHETIC_BASELINE", None)
        else:
            os.environ["ALLOW_SYNTHETIC_BASELINE"] = prev_allow
