"""Provisional Study 1 labels: walking direction of every trial and the class of every gait cycle.

Walking direction. The protocol alternates short-to-long and long-to-short walks. A trial's direction
is read from the CoP excursion of its last five steps against its first five (larger = longer steps;
both limbs pooled; ratio > 1.05 or < 0.95). A trial without clear evidence takes the opposite direction
of an adjacent walk. How often adjacent walks alternate is reported as a check of the evidence.
This is provisional until the protocol sheet is confirmed.

Classes of one limb graph (step k runs from valley v(k-1) to v(k)):
    GI          rise out of standing to v1 (gait initiation and the first step)
    steps 2-5   short (short-to-long walk) or long (long-to-short walk)
    step 6      short-to-long or long-to-short transition
    steps 7-11  the other step length
    GT          v11 to the point of rest (at least one median step long)
Normal-step walking is not in the segmented recordings yet.

Run:  python -m common.labels   ->  datasets/labels_provisional.csv
"""
from __future__ import annotations

import collections
import csv
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets"
CLASSES = ["GI", "short", "long", "short_to_long", "long_to_short", "GT"]


def walk_order(graphs: list[dict]) -> dict[tuple, tuple]:
    """(block, walk number) of every trial. Sets with one trial per record number their walks by the record suffix."""
    trials = collections.defaultdict(set)
    for g in graphs:
        trials[(g["subject"], g["record"])].add(g["trial"])
    order = {}
    for (subject, record), numbers in trials.items():
        for trial in numbers:
            if len(numbers) > 2:
                order[(subject, record, trial)] = ((subject, record), trial)
            else:
                order[(subject, record, trial)] = ((subject, "single"), int(re.findall(r"\d+", record)[-1]))
    return order


def excursion_ratio(graph: dict) -> float:
    ex = graph["cop_excursion"]
    return float(np.median(ex[6:11]) / max(np.median(ex[1:5]), 1e-9))


def evidence(ratio: float) -> str:
    return "short_to_long" if ratio > 1.05 else "long_to_short" if ratio < 0.95 else "unclear"


def directions(graphs: list[dict]) -> tuple[dict, list[dict]]:
    """Direction of every trial: its own CoP evidence (both limbs pooled); an unclear trial takes the
    opposite of an adjacent walk of the same block, because the protocol alternates the direction."""
    ratios = collections.defaultdict(list)
    for g in graphs:
        ratios[(g["subject"], g["record"], g["trial"])].append(excursion_ratio(g))
    order = walk_order(graphs)
    own = {key: evidence(float(np.median(r))) for key, r in ratios.items()}
    by_walk = {order[key]: key for key in own}
    flip = {"short_to_long": "long_to_short", "long_to_short": "short_to_long"}
    result, report = {}, []
    for key, mine in own.items():
        block, walk = order[key]
        direction, source = mine, "own CoP evidence"
        if mine == "unclear":
            for neighbour in (walk - 1, walk + 1):
                other = by_walk.get((block, neighbour))
                if other and own[other] != "unclear":
                    direction, source = flip[own[other]], f"opposite of walk {neighbour}"
                    break
        alternates = [own[by_walk[(block, n)]] == flip.get(mine) for n in (walk - 1, walk + 1)
                      if (block, n) in by_walk and own[by_walk[(block, n)]] != "unclear" and mine != "unclear"]
        result[key] = direction
        report.append({"subject": key[0], "record": key[1], "trial": key[2], "block": "/".join(block), "walk": walk,
                       "direction": direction, "source": source, "cop_excursion_ratio": round(float(np.median(ratios[key])), 3),
                       "alternates_with_neighbours": f"{sum(alternates)}/{len(alternates)}"})
    return result, report


def cycle_windows(graph: dict, direction: str) -> list[tuple[str, int, int, str]]:
    """(class, start, end, position) of every labelled window of one limb graph."""
    v = graph["valleys"]
    first, second = ("short", "long") if direction == "short_to_long" else ("long", "short")
    windows = [("GI", graph["rise"], v[1], "GI")]
    for k in range(2, 12):
        label = first if k <= 5 else direction if k == 6 else second
        windows.append((label, v[k - 1], v[k], f"step{k}"))
    median_step = int(np.median(np.diff(v)))
    end = min(graph["n"] - 1, max(graph["rest"], v[11] + median_step))
    windows.append(("GT", v[11], end, "GT"))
    return windows


if __name__ == "__main__":
    with (OUT / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, report = directions(graphs)
    with (OUT / "labels_provisional.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(report, key=lambda r: (r["subject"], r["record"], r["trial"])))
    c = collections.Counter(r["direction"] for r in report)
    pairs = [tuple(map(int, r["alternates_with_neighbours"].split("/"))) for r in report]
    print(f"trials: {len(report)}  directions: {dict(c)}  (from a neighbour: {sum(r['source'] != 'own CoP evidence' for r in report)})")
    print(f"adjacent walks with clear evidence walk in opposite directions in {sum(a for a, _ in pairs)}/{sum(b for _, b in pairs)} comparisons")
    graphs_by_dir = collections.Counter(dirs[(g['subject'], g['record'], g['trial'])] for g in graphs)
    print("limb graphs by direction:", dict(graphs_by_dir))
