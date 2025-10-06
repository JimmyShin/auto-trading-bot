from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Optional dependency; Markdown table fallback implemented below.
    from tabulate import tabulate  # type: ignore
except Exception:  # pragma: no cover - runtime import guard
    tabulate = None  # type: ignore

logger = logging.getLogger("reporting.report_metrics")

LEGACY_REPORTER: Optional[Any] = None
LEGACY_COERCE: Optional[Any] = None
LEGACY_MAX_DD: Optional[Any] = None

if os.environ.get("REPORTING_FORCE_FALLBACK", "").lower() not in {"1", "true", "yes"}:
    try:  # pragma: no cover - import guard is exercised indirectly in tests
        from auto_trading_bot import reporter as _legacy_reporter  # type: ignore
    except Exception:  # pragma: no cover - absence path covered elsewhere
        _legacy_reporter = None
    else:
        LEGACY_REPORTER = _legacy_reporter
        LEGACY_COERCE = getattr(_legacy_reporter, "_coerce_trades", None)
        LEGACY_MAX_DD = getattr(_legacy_reporter, "_max_drawdown_from_returns", None)
        logger.debug("Using legacy reporter helpers from auto_trading_bot.reporter")
else:
    logger.debug("REPORTING_FORCE_FALLBACK requested; skipping legacy reporter imports")

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "entry_price": ("entry_price", "entry", "entryPrice"),
    "exit_price": ("exit_price", "exit", "exitPrice"),
    "qty": ("qty", "quantity", "size"),
    "side": ("side", "direction", "position_side"),
    "entry_ts_utc": ("entry_ts_utc", "entry_timestamp", "entry_time", "timestamp"),
    "exit_ts_utc": ("exit_ts_utc", "exit_timestamp", "exit_time", "close_time"),
    "entry_atr_abs": ("entry_atr_abs", "atr_abs", "atr"),
    "stop_k": ("stop_k", "stop_multiplier", "stopk"),
    "stop_distance": ("stop_distance", "stop_distance_quote", "stop_dist"),
    "risk_usdt_planned": ("risk_usdt_planned", "risk_usdt", "risk_quote"),
    "pnl_quote_expost": (
        "pnl_quote_expost",
        "pnl_quote",
        "pnl_realized_quote",
        "pnl",
    ),
    "fees_quote_actual": ("fees_quote_actual", "fees_quote", "commission_paid"),
    "R_atr_expost": ("R_atr_expost", "r_atr_expost", "r_multiple_atr"),
    "R_usd_expost": ("R_usd_expost", "r_usd_expost", "r_multiple_usd"),
    "order_errors": ("order_errors", "order_error_count", "errors"),
    "signals_emitted": ("signals_emitted", "signals", "signal_count"),
    "mae": ("mae", "mae_quote", "mae_pct"),
    "mfe": ("mfe", "mfe_quote", "mfe_pct"),
}

DATE_COLUMNS = ("entry_ts_utc", "exit_ts_utc")
NUMERIC_COLUMNS = (
    "entry_price",
    "exit_price",
    "qty",
    "entry_atr_abs",
    "stop_k",
    "stop_distance",
    "risk_usdt_planned",
    "pnl_quote_expost",
    "fees_quote_actual",
    "R_atr_expost",
    "R_usd_expost",
)

REQUIRED_FOR_R = ("entry_price", "exit_price", "side", "entry_atr_abs", "stop_k")
DEFAULT_WINDOWS = (10, 30, 100)
BOOTSTRAP_SAMPLES = 1000
REPORT_SCHEMA_VERSION = 1


@dataclass
class TradeMetrics:
    n: int
    wins: int
    losses: int
    win_rate: float
    avg_r_atr: float
    avg_r_usd: float
    expectancy_usd: float
    profit_factor: float
    payoff_ratio: float
    fees_total: float
    fees_per_trade: float
    sum_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_r_atr": self.avg_r_atr,
            "avg_r_usd": self.avg_r_usd,
            "expectancy_usd": self.expectancy_usd,
            "profit_factor": self.profit_factor,
            "payoff_ratio": self.payoff_ratio,
            "fees_total": self.fees_total,
            "fees_per_trade": self.fees_per_trade,
            "sum_pnl": self.sum_pnl,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate trade performance reports from CSV logs.")
    parser.add_argument("--env", required=True, help="Environment name (e.g. testnet, live)")
    parser.add_argument(
        "--csv-glob",
        help="Optional glob override for trade CSVs (defaults to data/<env>/trades_*.csv)",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write reports into")
    parser.add_argument(
        "--windows",
        default="10,30,100",
        help="Comma separated list of rolling window sizes",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=30,
        help="Minimum trades required to consider report decision-ready",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def discover_csvs(env: str, csv_glob: Optional[str]) -> List[Path]:
    pattern = csv_glob if csv_glob else os.path.join("data", env, "trades_*.csv")
    if os.path.isabs(pattern):
        paths = sorted(Path(p) for p in glob.glob(pattern))
    else:
        paths = sorted(Path().glob(pattern))
    return [p for p in paths if Path(p).is_file()]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    lower_to_original = {col.lower(): col for col in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
            lower = alias.lower()
            if lower in lower_to_original:
                rename_map[lower_to_original[lower]] = canonical
                break
    return df.rename(columns=rename_map)


def _coerce_via_legacy(trades: pd.DataFrame) -> Optional[pd.DataFrame]:
    if LEGACY_COERCE is None:
        return None
    try:
        coerced = LEGACY_COERCE(trades)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Legacy coercion failed: %s", exc)
        return None
    logger.info("Using legacy reporter._coerce_trades() for normalization")
    return coerced


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _compute_returns(df: pd.DataFrame) -> pd.Series:
    if LEGACY_COERCE is not None and "return" in df.columns:
        return df["return"].astype(float)

    entry = _safe_numeric(df.get("entry_price", pd.Series(dtype=float)))
    exitp = _safe_numeric(df.get("exit_price", pd.Series(dtype=float)))
    side = df.get("side", pd.Series(dtype=str)).astype(str).str.lower()

    returns = pd.Series(np.nan, index=df.index, dtype="float64")
    long_mask = side == "long"
    short_mask = side == "short"
    if not entry.empty and not exitp.empty:
        returns.loc[long_mask] = (exitp - entry)[long_mask] / entry[long_mask]
        returns.loc[short_mask] = (entry - exitp)[short_mask] / entry[short_mask]
    pnl_pct = _safe_numeric(df.get("pnl_pct", pd.Series(dtype=float)))
    returns = returns.fillna(pd.NA)
    if returns.isna().all() and not pnl_pct.empty:
        scale = 100.0 if pnl_pct.abs().median() > 1 else 1.0
        returns = (pnl_pct / scale).astype(float)
    return returns.fillna(0.0)


def normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    original_columns = trades.columns.tolist()
    trades = _rename_columns(trades)
    trades.columns = [c.strip() for c in trades.columns]
    trades = trades.replace({"": np.nan, " ": np.nan})

    if LEGACY_COERCE is not None:
        coerced = _coerce_via_legacy(trades)
        if coerced is not None:
            # Ensure we keep original extras by aligning on index
            for column in trades.columns:
                if column not in coerced.columns:
                    coerced[column] = trades[column]
            trades = coerced

    trades = _parse_dates(trades)

    for col in NUMERIC_COLUMNS:
        if col in trades.columns:
            trades[col] = _safe_numeric(trades[col])
        else:
            trades[col] = np.nan

    if "side" in trades.columns:
        trades["side"] = trades["side"].astype(str).str.lower()

    # Compose timestamps for filtering and ordering
    if "exit_ts_utc" not in trades.columns:
        trades["exit_ts_utc"] = pd.NaT
    if "entry_ts_utc" not in trades.columns:
        trades["entry_ts_utc"] = pd.NaT

    # Filter to closed trades: exit timestamp or exit price present
    closed_mask = trades["exit_ts_utc"].notna() | trades["exit_price"].notna()
    closed = trades.loc[closed_mask].copy()
    closed["exit_ts_utc"] = closed["exit_ts_utc"].fillna(closed["entry_ts_utc"])

    closed["return"] = _compute_returns(closed)
    closed = _compute_r_values(closed)
    closed = _ensure_usd_metrics(closed)
    closed = _attach_duration(closed)
    closed = _attach_positional_flags(closed)

    logger.debug("Normalized trades with columns: %s -> %s", original_columns, closed.columns.tolist())
    return closed


def _attach_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_ts_utc" in df.columns and "exit_ts_utc" in df.columns:
        duration = df["exit_ts_utc"] - df["entry_ts_utc"]
        df["duration_sec"] = duration.dt.total_seconds()
    else:
        df["duration_sec"] = np.nan
    return df


def _attach_positional_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_win"] = _safe_numeric(df.get("pnl_quote_expost", pd.Series(dtype=float))) > 0
    if df["is_win"].sum() == 0 and "R_atr_expost" in df.columns:
        df["is_win"] = df["R_atr_expost"].fillna(0.0) > 0
    df["is_loss"] = ~df["is_win"]
    return df


def _ensure_usd_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "R_usd_expost" not in df.columns:
        df["R_usd_expost"] = np.nan

    risk = df.get("risk_usdt_planned", pd.Series(dtype=float))
    pnl_usd = df.get("pnl_quote_expost", pd.Series(dtype=float))
    fees = df.get("fees_quote_actual", pd.Series(dtype=float))

    need_r_usd = df["R_usd_expost"].isna() & risk.notna() & risk.ne(0)
    if need_r_usd.any() and not pnl_usd.empty:
        df.loc[need_r_usd, "R_usd_expost"] = (pnl_usd - fees.fillna(0))[need_r_usd] / risk[need_r_usd]

    return df


def _compute_r_values(df: pd.DataFrame) -> pd.DataFrame:
    if "R_atr_expost" not in df.columns:
        df["R_atr_expost"] = np.nan

    missing = df["R_atr_expost"].isna() | ~np.isfinite(df["R_atr_expost"])
    if missing.any():
        atr = df.get("entry_atr_abs", pd.Series(dtype=float))
        stop_k = df.get("stop_k", pd.Series(dtype=float))
        entry = df.get("entry_price", pd.Series(dtype=float))
        exitp = df.get("exit_price", pd.Series(dtype=float))
        side = df.get("side", pd.Series(dtype=str)).astype(str).str.lower()
        denom = stop_k * atr
        price_distance = pd.Series(np.nan, index=df.index, dtype="float64")
        price_distance.loc[side == "long"] = (exitp - entry)[side == "long"]
        price_distance.loc[side == "short"] = (entry - exitp)[side == "short"]

        viable = missing & denom.notna() & denom.ne(0) & price_distance.notna()
        df.loc[viable, "R_atr_expost"] = price_distance[viable] / denom[viable]

        # fallback using USD metrics if still missing
        still_missing = df["R_atr_expost"].isna()
        usd = df.get("R_usd_expost", pd.Series(dtype=float))
        if usd is not None and not usd.empty:
            df.loc[still_missing, "R_atr_expost"] = usd[still_missing]

    return df


def _calc_trade_metrics(df: pd.DataFrame) -> TradeMetrics:
    if df.empty:
        return TradeMetrics(0, 0, 0, 0.0, math.nan, math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnl = _safe_numeric(df.get("pnl_quote_expost", pd.Series(dtype=float)))
    fees = _safe_numeric(df.get("fees_quote_actual", pd.Series(dtype=float)))
    pnl = pnl.fillna(0.0)
    fees = fees.fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    n = int(len(df))
    win_rate = wins / n if n else 0.0

    avg_win = float(pnl[pnl > 0].mean()) if wins else 0.0
    avg_loss = float(abs(pnl[pnl <= 0].mean())) if losses else 0.0
    expectancy_usd = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    gross_wins = float(pnl[pnl > 0].sum())
    gross_losses = float(abs(pnl[pnl <= 0].sum()))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (math.inf if gross_wins > 0 else 0.0)
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else (math.inf if avg_win > 0 else 0.0)

    avg_r_atr = float(df["R_atr_expost"].dropna().mean()) if "R_atr_expost" in df.columns else math.nan
    avg_r_usd = float(df["R_usd_expost"].dropna().mean()) if "R_usd_expost" in df.columns else math.nan

    fees_total = float(fees.sum())
    fees_per_trade = fees_total / n if n else 0.0

    return TradeMetrics(
        n=n,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_r_atr=avg_r_atr,
        avg_r_usd=avg_r_usd,
        expectancy_usd=expectancy_usd,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        fees_total=fees_total,
        fees_per_trade=fees_per_trade,
        sum_pnl=float(pnl.sum()),
    )


def _distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
    series = df.get("R_atr_expost", pd.Series(dtype=float)).dropna()
    if series.empty:
        return {key: math.nan for key in ["median", "p25", "p75", "p95", "skew", "kurtosis"]}
    return {
        "median": float(series.median()),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurt()),
    }


def _bootstrap_ci(series: pd.Series, samples: int = BOOTSTRAP_SAMPLES) -> Dict[str, Optional[float]]:
    clean = series.dropna().astype(float)
    if len(clean) < 30:
        return {"lower": math.nan, "upper": math.nan}
    rng = np.random.default_rng(seed=42)
    means = []
    values = clean.to_numpy()
    for _ in range(samples):
        draw = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(draw)))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def _drawdown_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    pnl = _safe_numeric(df.get("pnl_quote_expost", pd.Series(dtype=float))).fillna(0.0)
    cumulative = pnl.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    max_dd = float(drawdown.max()) if not drawdown.empty else 0.0
    max_peak = float(peak.max()) if not peak.empty else 0.0
    dd_pct = (max_dd / max_peak) if max_peak > 0 else (max_dd if max_dd else 0.0)

    returns = df.get("return", pd.Series(dtype=float))
    ratio_dd = float("nan")
    if LEGACY_MAX_DD is not None and not returns.empty:
        try:
            ratio_dd = float(LEGACY_MAX_DD(returns))
        except Exception:  # pragma: no cover - defensive guard
            ratio_dd = float("nan")
    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": float(dd_pct),
        "return_based_drawdown": ratio_dd,
    }


def _fees_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    pnl = _safe_numeric(df.get("pnl_quote_expost", pd.Series(dtype=float))).fillna(0.0)
    fees = _safe_numeric(df.get("fees_quote_actual", pd.Series(dtype=float))).fillna(0.0)
    gross = float(pnl.sum())
    total_fees = float(fees.sum())
    ratio = total_fees / abs(gross) if abs(gross) > 1e-9 else float("nan")
    return {
        "total_fees": total_fees,
        "fees_to_pnl_ratio": ratio,
        "avg_fee": float(fees.mean()) if len(fees) else 0.0,
    }


def _error_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if "order_errors" in df.columns:
        total_errors = float(_safe_numeric(df["order_errors"]).sum())
        rate = (total_errors / len(df)) * 100.0 if len(df) else 0.0
    else:
        total_errors = 0.0
        rate = math.nan
    return {"total_errors": total_errors, "error_rate_pct": rate}


def _signal_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if "signals_emitted" in df.columns and "symbol" in df.columns:
        signals = float(_safe_numeric(df["signals_emitted"]).sum())
        trades = len(df)
        rate = trades / signals if signals else math.nan
    else:
        signals = math.nan
        rate = math.nan
    return {"signals_emitted": signals, "conversion_rate": rate}


def _data_quality(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    required = ["entry_price", "exit_price", "side", "pnl_quote_expost"]
    for col in required:
        if col not in df.columns:
            issues.append(f"missing required column: {col}")
            continue
        missing_ratio = df[col].isna().mean() if len(df) else 1.0
        if missing_ratio > 0.1:
            issues.append(f"{col} missing for {missing_ratio:.0%} of trades")
    return issues


def _rolling_metrics(df: pd.DataFrame, windows: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    rolling: Dict[str, Dict[str, Any]] = {}
    ordered = df.sort_values(by=["exit_ts_utc", "entry_ts_utc"]).reset_index(drop=True)
    for window in windows:
        subset = ordered.tail(window)
        metrics = _calc_trade_metrics(subset)
        entry = metrics.to_dict()
        entry["window"] = window
        entry["avg_fee"] = float(_safe_numeric(subset.get("fees_quote_actual", pd.Series(dtype=float))).mean()) if len(subset) else 0.0
        entry["expectancy_usd"] = metrics.expectancy_usd
        entry["avg_r_atr_ci"] = _bootstrap_ci(subset.get("R_atr_expost", pd.Series(dtype=float)))
        rolling[str(window)] = entry
    return rolling


def _select_anomalies(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    sort_series = _safe_numeric(df.get("R_atr_expost", pd.Series(dtype=float))).fillna(0.0)
    worst = df.assign(_r=sort_series).sort_values(by="_r").head(limit)
    cols = [c for c in ["entry_ts_utc", "symbol", "side", "R_atr_expost", "pnl_quote_expost", "fees_quote_actual"] if c in worst.columns]
    records: List[Dict[str, Any]] = []
    for _, row in worst.iterrows():
        record: Dict[str, Any] = {}
        for col in cols:
            value = row[col]
            if isinstance(value, pd.Timestamp):
                record[col] = value.isoformat()
            elif isinstance(value, (np.generic,)):
                record[col] = float(value)
            else:
                record[col] = value if pd.notna(value) else None
        records.append(record)
    return records


def _tabulate_windows(rolling: Mapping[str, Mapping[str, Any]]) -> str:
    headers = ["window", "n", "win_rate", "avg_r_atr", "expectancy_usd", "profit_factor"]
    rows = []
    for key in sorted(rolling.keys(), key=lambda k: int(k)):
        entry = rolling[key]
        rows.append(
            [
                key,
                int(entry.get("n", 0)),
                f"{entry.get('win_rate', 0.0):.2%}" if not math.isnan(entry.get("win_rate", 0.0)) else "n/a",
                _fmt(entry.get("avg_r_atr")),
                _fmt(entry.get("expectancy_usd")),
                _fmt(entry.get("profit_factor")),
            ]
        )
    if tabulate:
        return tabulate(rows, headers=headers, tablefmt="github")
    # Manual Markdown table fallback
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def build_report(df: pd.DataFrame, env: str, windows: Sequence[int], min_trades: int, csv_paths: Sequence[Path]) -> Dict[str, Any]:
    metrics = _calc_trade_metrics(df)
    rolling = _rolling_metrics(df, windows)
    distribution = _distribution_stats(df)
    drawdown = _drawdown_metrics(df)
    fees = _fees_metrics(df)
    errors = _error_metrics(df)
    signals = _signal_metrics(df)
    bootstrap = _bootstrap_ci(df.get("R_atr_expost", pd.Series(dtype=float)))
    anomalies = _select_anomalies(df)
    data_quality = _data_quality(df)

    decision_ready = metrics.n >= min_trades
    header_summary = (
        f"{env} trades={metrics.n} avg_r_atr={_fmt(metrics.avg_r_atr)} "
        f"win_rate={metrics.win_rate:.1%} expectancy=${metrics.expectancy_usd:.2f}"
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "environment": env,
        "generated_at": datetime.utcnow().replace(tzinfo=None).isoformat() + "Z",
        "source_files": [str(p) for p in csv_paths],
        "trade_count": metrics.n,
        "min_trades_required": min_trades,
        "decision_ready": decision_ready,
        "header_summary": header_summary,
        "overall": metrics.to_dict(),
        "rolling_windows": rolling,
        "distribution": distribution,
        "bootstrap_ci_avg_r_atr": bootstrap,
        "fees": fees,
        "drawdown": drawdown,
        "errors": errors,
        "signals": signals,
        "anomalies": anomalies,
        "data_quality_issues": data_quality,
    }
    return report


def write_outputs(report: Dict[str, Any], df: pd.DataFrame, env: str, out_dir: Path) -> Dict[str, Path]:
    out_root = out_dir
    if out_root.name != env:
        out_root = out_root / env
    out_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    json_path = out_root / f"{timestamp}-summary.json"
    markdown_path = out_root / f"{timestamp}-summary.md"
    csv_path = out_root / "latest.csv"

    clean_report = _clean_for_json(report)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(clean_report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    latest_summary = out_root / "latest-summary.json"
    latest_summary.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    df.to_csv(csv_path, index=False)

    markdown = _render_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_markdown = out_root / "latest-summary.md"
    latest_markdown.write_text(markdown, encoding="utf-8")

    return {
        "json": json_path,
        "latest_json": latest_summary,
        "markdown": markdown_path,
        "latest_md": latest_markdown,
        "csv": csv_path,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    rolling_md = _tabulate_windows(report.get("rolling_windows", {}))
    anomalies = report.get("anomalies", [])
    anomaly_lines = ["| entry_ts | symbol | side | R_atr | pnl_quote | fees |", "| --- | --- | --- | --- | --- | --- |"]
    for row in anomalies:
        anomaly_lines.append(
            "| {entry_ts} | {symbol} | {side} | {r} | {pnl} | {fees} |".format(
                entry_ts=row.get("entry_ts_utc", ""),
                symbol=row.get("symbol", ""),
                side=row.get("side", ""),
                r=_fmt(row.get("R_atr_expost")),
                pnl=_fmt(row.get("pnl_quote_expost")),
                fees=_fmt(row.get("fees_quote_actual")),
            )
        )
    md = [
        f"# Trade Summary — {report.get('environment', '')}",
        "",
        report.get("header_summary", ""),
        "",
        "## Rolling Windows",
        rolling_md,
        "",
        "## Top 5 Anomalies",
        "\n".join(anomaly_lines),
        "",
        "## Data Quality",
    ]
    issues = report.get("data_quality_issues", [])
    if issues:
        md.extend(["- " + issue for issue in issues])
    else:
        md.append("- No material data quality issues detected.")
    return "\n".join(md) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    csv_paths = discover_csvs(args.env, args.csv_glob)
    if not csv_paths:
        logger.error("No trade CSV files found for pattern")
        return 2

    frames = []
    for path in csv_paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    if not frames:
        logger.error("No readable trade CSV files found")
        return 2

    raw_df = pd.concat(frames, ignore_index=True)
    normalized = normalize_trades(raw_df)
    if normalized.empty:
        logger.error("No closed trades available for analysis")
        return 1

    report = build_report(normalized, args.env, windows, args.min_trades, csv_paths)
    outputs = write_outputs(report, normalized, args.env, Path(args.out_dir))

    rolling_30 = report.get("rolling_windows", {}).get("30", {})
    summary_line = (
        f"env={args.env} trades={report['trade_count']} "
        f"rolling_30.n={rolling_30.get('n', 'n/a')} "
        f"rolling_30.avg_r_atr={_fmt(rolling_30.get('avg_r_atr'))} "
        f"expectancy_usd={_fmt(rolling_30.get('expectancy_usd'))}"
    )
    print(summary_line)

    logger.info("Report written: %s", outputs["json"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
