import json
import os
from pathlib import Path

import pytest

from baseline import DEFAULT_BASELINE_PATH, generate_baseline

# Full baseline regeneration check is regression-level (slow)
pytestmark = pytest.mark.regression


@pytest.mark.parametrize("baseline_path", [DEFAULT_BASELINE_PATH])
def test_baseline_matches_regenerated(tmp_path, baseline_path: Path) -> None:
    os.environ.setdefault("ALLOW_SYNTHETIC_BASELINE", "1")
    if not baseline_path.exists():
        pytest.skip(f"Baseline file missing: {baseline_path}")

    stored = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert stored.get("symbols"), "Baseline symbols are empty"

    regen_path = tmp_path / "baseline.json"
    regenerated = generate_baseline(
        symbols=stored["symbols"].keys(),
        timeframe=stored["timeframe"],
        bars=int(stored["bars"]),
        equity=float(stored["equity"]),
        output_path=regen_path,
    )

    from baseline import looks_like_fallback

    if looks_like_fallback(regenerated.get("symbols", {})):
        pytest.skip("Synthetic baseline fixtures detected; skipping baseline match.")

    assert regenerated["symbols"] == stored["symbols"], "Strategy output diverged from baseline"
