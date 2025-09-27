import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")


def iter_order_csv_files(base: Path) -> list[Path]:
    return sorted(base.glob("**/orders_*.csv"))


def summarize_orders(files: list[Path]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for file in files:
        try:
            with file.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    order_type = (row.get("type") or "unknown").lower()
                    status = (row.get("status") or "unknown").lower()
                    counter[(order_type, status)] += 1
        except FileNotFoundError:
            continue
    return counter


def main() -> int:
    files = iter_order_csv_files(DATA_DIR)
    if not files:
        print("No order CSV files found under data/.")
        return 0

    counter = summarize_orders(files)
    if not counter:
        print("No order rows found in CSV files.")
        return 0

    print("type,status,count")
    for (order_type, status), count in counter.most_common(30):
        print(f"{order_type},{status},{count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
