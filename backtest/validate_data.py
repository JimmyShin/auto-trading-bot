import argparse
import pandas as pd
from backtest.utils import out_path, ensure_utc, dedup_sort, check_gaps, read_parquet_or_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True)
    ap.add_argument('--timeframe', default='15m')
    ap.add_argument('--base_dir', default='data')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    tf = args.timeframe
    for sym in symbols:
        p = out_path(args.base_dir, tf, sym)
        df = read_parquet_or_csv(p)
        df = ensure_utc(dedup_sort(df))
        report = check_gaps(df, tf)
        print(f"{sym} {tf}: rows={report['rows']} gaps={report['gap_count']} max_gap_ms={report['max_gap_ms']} step={report['expected_step_ms']}")

if __name__ == '__main__':
    main()
