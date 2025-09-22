import os
import math
import pandas as pd
from datetime import timezone


def timeframe_to_ms(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith('ms'):
        return int(tf[:-2])
    units = {
        's': 1000,
        'm': 60_000,
        'h': 3_600_000,
        'd': 86_400_000,
    }
    for u, mult in units.items():
        if tf.endswith(u):
            return int(float(tf[:-1]) * mult)
    raise ValueError(f"Unknown timeframe: {tf}")


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if 'ts' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
        else:
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize('UTC')
            else:
                df['ts'] = df['ts'].dt.tz_convert('UTC')
    return df


def dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['ts']).sort_values('ts').reset_index(drop=True)
    return df


def check_gaps(df: pd.DataFrame, tf: str) -> dict:
    step = timeframe_to_ms(tf)
    ts = df['ts'].astype('int64') // 1_000_000  # ms
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > step]
    gap_count = int((gaps > step).sum())
    max_gap = int(gaps.max()) if not gaps.empty else 0
    return {
        'rows': len(df),
        'gap_count': gap_count,
        'max_gap_ms': max_gap,
        'expected_step_ms': step,
    }


def out_path(base_dir: str, timeframe: str, symbol: str) -> str:
    sym = symbol.replace('/', '_')
    sub = os.path.join(base_dir, 'backtest', 'ohlcv', timeframe)
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{sym}.parquet")


def to_parquet_or_csv(df: pd.DataFrame, path: str) -> str:
    """Save DataFrame to parquet if engine available; otherwise CSV.

    Returns the actual file path written.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        # Missing pyarrow/fastparquet or other parquet engine issue
        base, _ = os.path.splitext(path)
        csv_path = base + '.csv'
        df.to_csv(csv_path, index=False)
        return csv_path


def read_parquet_or_csv(path: str) -> pd.DataFrame:
    """Read parquet if available; otherwise CSV sibling with same basename.
    Ensures 'ts' is UTC datetime if present.
    """
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            return ensure_utc(df)
        except Exception:
            pass
    base, _ = os.path.splitext(path)
    csv_path = base + '.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return ensure_utc(df)
    raise FileNotFoundError(f"Neither parquet nor csv found for {path}")
