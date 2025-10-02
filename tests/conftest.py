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
    prev_dir = os.getenv("BASELINE_OHLCV_DIR")
    prev_allow = os.getenv("ALLOW_SYNTHETIC_BASELINE")

    if ci:
        sample_dir = Path(__file__).parent / "data" / "ohlcv"
        os.environ["BASELINE_OHLCV_DIR"] = str(sample_dir)
        os.environ.setdefault("ALLOW_SYNTHETIC_BASELINE", "1")

    yield

    if ci:
        if prev_dir is None:
            os.environ.pop("BASELINE_OHLCV_DIR", None)
        else:
            os.environ["BASELINE_OHLCV_DIR"] = prev_dir

        if prev_allow is None:
            os.environ.pop("ALLOW_SYNTHETIC_BASELINE", None)
        else:
            os.environ["ALLOW_SYNTHETIC_BASELINE"] = prev_allow
