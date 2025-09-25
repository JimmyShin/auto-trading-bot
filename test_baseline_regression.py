import json
from pathlib import Path

import pytest

from baseline import DEFAULT_BASELINE_PATH, generate_baseline


@pytest.mark.parametrize("baseline_path", [DEFAULT_BASELINE_PATH])
def test_baseline_matches_regenerated(tmp_path, baseline_path: Path) -> None:
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

    assert regenerated["symbols"] == stored["symbols"], "Strategy output diverged from baseline"
