from __future__ import annotations

import statistics
from pathlib import Path


ROOT = Path(r"C:\Users\MAHATVA GOEL\Desktop\Gait Analysis")


def read_rows(path: Path, width: int):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            tokens = [token.strip() for token in line.strip("\r\n").split("\t") if token.strip()]
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                continue
            if len(values) == width:
                rows.append(values)
    return rows


def rising_edges(rows):
    return [index for index in range(1, len(rows)) if rows[index - 1][1] == 0 and rows[index][1] == 1]


def describe(path_text: str, width: int, channels: slice | None = None):
    path = ROOT / path_text
    rows = read_rows(path, width)
    edges = rising_edges(rows)
    print(f"\n{path_text}: {len(rows)} valid rows; edges={[(i, rows[i][0]) for i in edges[:15]]}")
    if not rows:
        return
    if channels is not None:
        for anchor in [0, edges[0] if edges else 0, min(len(rows) - 200, (edges[0] if edges else 0) + 100)]:
            start = max(0, anchor)
            end = min(len(rows), start + 100)
            values = [row[channels] for row in rows[start:end]]
            energy = [sum((value - statistics.median(col)) ** 2 for value, col in zip(row, zip(*values))) for row in values]
            print(f"  rows {start}:{end} time {rows[start][0]:.2f}-{rows[end-1][0]:.2f}; median energy {statistics.median(energy):.1f}")


for fmg, left, right in [
    ("Sub06_H/Rishabh_2", "Sub06_H/Rishabh_2L", "Sub06_H/Rishabh_2R"),
    ("Sub01_H/abh_1", "Sub01_H/abh_1L", "Sub01_H/abh_1R"),
    ("Sub02_H/Alif_1", "Sub02_H/Alif_1L", "Sub02_H/Alif_1R"),
]:
    describe(fmg, 18, slice(2, 10))
    describe(left, 4)
    describe(right, 4)
