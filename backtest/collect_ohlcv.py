import argparse
import time
from typing import List
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone

from backtest.utils import out_path, ensure_utc, dedup_sort, timeframe_to_ms, to_parquet_or_csv


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    all_rows = []
    step = timeframe_to_ms(timeframe) * 500  # batches of 500 candles
    cursor = since_ms
    while cursor < until_ms:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=500)
        except Exception as e:
            time.sleep(1)
            continue
        if not data:
            break
        all_rows.extend(data)
        last_ms = data[-1][0]
        # advance one step beyond last to avoid duplicates
        cursor = max(cursor + step, last_ms + timeframe_to_ms(timeframe))
        # be nice to rate limits
        time.sleep(0.1)
    if not all_rows:
        return pd.DataFrame(columns=['ts','open','high','low','close','volume'])
    df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='Comma-separated symbols e.g. BTC/USDT,ETH/USDT')
    ap.add_argument('--timeframe', default='15m')
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--base_dir', default='data')
    args = ap.parse_args()

    symbols: List[str] = [s.strip() for s in args.symbols.split(',') if s.strip()]
    tf = args.timeframe
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)

    ex = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': { 'defaultType': 'future' },
        'timeout': 30_000,
    })

    for sym in symbols:
        print(f"Collecting {sym} {tf} for {args.days}d...")
        df = fetch_ohlcv(ex, sym, tf, since_ms, until_ms)
        if df.empty:
            print(f"WARN no data for {sym}")
            continue
        df = ensure_utc(df)
        df = dedup_sort(df)
        out = out_path(args.base_dir, tf, sym)
        actual = to_parquet_or_csv(df, out)
        print(f"Saved {len(df)} rows -> {actual}")

if __name__ == '__main__':
    main()
