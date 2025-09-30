from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "reporting" / "samples"
CONFIG_PATH = ROOT / "reporting" / "good_to_go_config.yaml"


def run_report_metrics(env: str, out_dir: Path, sample: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "reporting.report_metrics",
        "--env",
        env,
        "--out-dir",
        str(out_dir),
        "--csv-glob",
        str(sample),
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True, cwd=ROOT)
    return out_dir / env / "latest-summary.json"


def run_good_to_go(report: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "reporting.good_to_go_check",
        "--report",
        str(report),
        "--config",
        str(CONFIG_PATH),
    ]
    return subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)


def load_artifact(report: Path) -> dict:
    with (report.parent / "good_to_go_result.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_good_to_go_passes_for_healthy_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    report_path = run_report_metrics("ok", out_dir, SAMPLES_DIR / "sample_ok.csv")
    result = run_good_to_go(report_path)
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout

    payload = load_artifact(report_path)
    assert payload["status"] == "pass"


def test_good_to_go_fails_when_not_enough_trades(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    report_path = run_report_metrics(
        "limited", out_dir, SAMPLES_DIR / "sample_not_enough_trades.csv"
    )
    result = run_good_to_go(report_path)
    assert result.returncode == 1
    payload = load_artifact(report_path)
    gate = next(item for item in payload["results"] if item["key"] == "min_trades_rolling_30")
    assert gate["passed"] is False
    assert gate["status"] == "fail"


def test_good_to_go_flags_fee_ratio(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    report_path = run_report_metrics("fees", out_dir, SAMPLES_DIR / "sample_fees_killprofit.csv")
    result = run_good_to_go(report_path)
    assert result.returncode == 1
    payload = load_artifact(report_path)
    fee_gate = next(item for item in payload["results"] if item["key"] == "max_fee_pct_of_pnl")
    assert fee_gate["passed"] is False
    assert fee_gate["status"] == "fail"
