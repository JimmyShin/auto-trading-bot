import csv
from pathlib import Path
from typing import List

import pytest

from scripts.orders_summary import iter_order_csv_files, main, summarize_orders


def test_iter_order_csv_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_a = data_dir / "orders_one.csv"
    file_a.write_text("type,status\nlimit,filled\n", encoding="utf-8")
    file_b = data_dir / "nested" / "orders_two.csv"
    file_b.parent.mkdir()
    file_b.write_text("type,status\nmarket,cancelled\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    files = iter_order_csv_files(Path("data"))
    assert len(files) == 2
    assert {f.name for f in files} == {"orders_one.csv", "orders_two.csv"}


def create_order_csv(path: Path, rows: List[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["type", "status"])
        for row in rows:
            writer.writerow(row)


def test_summarize_orders_counts(tmp_path: Path) -> None:
    files = []
    for idx, rows in enumerate(
        [
            [("limit", "filled"), ("limit", "filled")],
            [("limit", "cancelled")],
            [("market", "filled"), ("market", "filled"), ("market", "filled")],
        ]
    ):
        file_path = tmp_path / f"orders_{idx}.csv"
        create_order_csv(file_path, rows)
        files.append(file_path)

    counter = summarize_orders(files)
    assert counter[("limit", "filled")] == 2
    assert counter[("limit", "cancelled")] == 1
    assert counter[("market", "filled")] == 3


def test_main_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    assert main() == 0
    output = capsys.readouterr().out.strip()
    assert output == "No order CSV files found under data/."


def test_main_with_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    nested = data_dir / "nested"
    nested.mkdir(parents=True)
    create_order_csv(data_dir / "orders_one.csv", [("limit", "filled"), ("limit", "filled")])
    create_order_csv(nested / "orders_two.csv", [("market", "cancelled")])

    monkeypatch.chdir(tmp_path)

    assert main() == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0] == "type,status,count"
    assert "limit,filled,2" in output[1:]
    assert "market,cancelled,1" in output[1:]
