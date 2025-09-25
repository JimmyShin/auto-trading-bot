import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_ohlcv_dir_for_ci():
    # On CI, use bundled sample OHLCV fixtures for baseline regeneration
    if os.getenv("CI", "").lower() == "true":
        sample_dir = Path(__file__).parent / "data" / "ohlcv"
        os.environ["BASELINE_OHLCV_DIR"] = str(sample_dir)
