from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import auto_trading_bot.reporter as reporter

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "reporting" / "samples"


def run_report_metrics(
    env: str, out_dir: Path, csv_glob: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "reporting.report_metrics",
        "--env",
        env,
        "--out-dir",
        str(out_dir),
        "--csv-glob",
        str(csv_glob),
    ]
    env_vars = os.environ.copy()
    if extra_env:
        env_vars.update(extra_env)
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT, env=env_vars)


def test_report_metrics_generates_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    result = run_report_metrics("testnet", out_dir, SAMPLES_DIR / "sample_ok.csv")
    assert "rolling_30" in result.stdout or "rolling_30" in result.stderr

    report_path = out_dir / "testnet" / "latest-summary.json"
    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    for key in ("rolling_windows", "distribution", "fees", "drawdown"):
        assert key in report

    rolling_30 = report["rolling_windows"]["30"]
    assert rolling_30["n"] >= 30
    assert math.isfinite(float(rolling_30["avg_r_atr"]))

    normalized_csv = out_dir / "testnet" / "latest.csv"
    df = pd.read_csv(normalized_csv)
    assert "R_atr_expost" in df.columns


def test_report_metrics_computes_r_without_column(tmp_path: Path) -> None:
    df = pd.read_csv(SAMPLES_DIR / "sample_ok.csv")
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("r_")])
    csv_path = tmp_path / "missing_r.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "reports"
    run_report_metrics(
        "fallback",
        out_dir,
        csv_path,
        extra_env={"REPORTING_FORCE_FALLBACK": "1"},
    )

    normalized_csv = out_dir / "fallback" / "latest.csv"
    norm = pd.read_csv(normalized_csv)
    assert "R_atr_expost" in norm.columns
    assert norm["R_atr_expost"].notna().any(), "R_atr_expost should be computed when missing"


def test_generate_report_logs_metrics(monkeypatch, caplog):
    monkeypatch.setenv("OBS_DEBUG_ALERTS", "1")
    df = pd.DataFrame(
        {
            "return": [0.6, -0.3, 0.4],
            "R_atr_expost": [1.2, -0.5, 0.9],
            "R_usd_expost": [40.0, -20.0, 25.0],
            "pnl_quote_expost": [40.0, -20.0, 25.0],
            "entry_ts": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "exit_ts": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "side": ["long", "short", "long"],
        }
    )

    with caplog.at_level(logging.INFO, logger="auto_trading_bot.reporter"):
        report = reporter.generate_report(df)

    logs = [rec.message for rec in caplog.records if rec.message.startswith("REPORTER_METRICS ")]
    assert logs, "Expected REPORTER_METRICS log entry"
    payload = json.loads(logs[0].split(" ", 1)[1])
    assert payload["avg_r_atr_30"] == pytest.approx(report.iloc[0]["avg_r_atr"])
    assert payload["window_trades"] == 3


def test_generate_report_empty_window(monkeypatch):
    monkeypatch.setenv("OBS_DEBUG_ALERTS", "1")
    empty_df = pd.DataFrame(
        columns=[
            "return",
            "R_atr_expost",
            "R_usd_expost",
            "pnl_quote_expost",
            "entry_ts",
            "exit_ts",
            "side",
        ]
    )
    report = reporter.generate_report(empty_df)
    assert math.isnan(report.iloc[0]["avg_r_atr"])
