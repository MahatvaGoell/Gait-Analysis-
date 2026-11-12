"""Study 1, real-time track: classify trailing FMG windows every 20 ms.

Every sample from 1.5 s before the rise to the end of the GT window gets the class of the labelled
window it lies in (common.labels.cycle_windows); samples before the rise or after the GT window are
"standing". A causal window of 200 or 300 ms ending at the sample is described by 6 statistics per
channel (48 features; FMG minus the trial's quiet-standing baseline), standardised with the subject's
calibration walks (as in study1/cycle/run_cycle_context.py) and classified with LDA or gradient boosting.
Calibration walks are never scored. Windows that contain estimated FMG samples are skipped.

Event timing: for the rise (GI), the transition step (v5) and v11 (GT), the delay is the time from the
event to the first of three consecutive windows predicted as the new class, searched up to the end
of that class window.

Run:  python -m study1.realtime.run_realtime   ->  results/realtime_*.csv
"""
from __future__ import annotations

import collections
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores  # noqa: E402
from common.labels import CLASSES, cycle_windows, directions, walk_order  # noqa: E402

RESULTS = ROOT / "results"
LABELS = ["standing"] + CLASSES
STRIDE = 2              # 20 ms
TRAIN_STRIDE = 6        # training windows every 60 ms (neighbouring windows are near copies)
PRE = 150               # samples of standing before the rise


def window_features(x: np.ndarray) -> np.ndarray:
    """x: (windows, length, 8) -> (windows, 48)."""
    t = np.linspace(-1, 1, x.shape[1])[None, :, None]
    slope = (x * t).sum(1) / (t ** 2).sum()
    return np.concatenate([x.mean(1), x.std(1), np.sqrt((x ** 2).mean(1)),
                           np.percentile(x, 90, axis=1) - np.percentile(x, 10, axis=1),
                           np.abs(np.diff(x, axis=1)).mean(1), slope], axis=1)


def build(graphs, dirs, length):
    order = walk_order(graphs)
    walks = collections.defaultdict(list)
    for key, (block, walk) in order.items():
        walks[(key[0], dirs[key])].append(((block[1], walk), key))
    calibration = {key for lst in walks.values() for _, key in sorted(lst)[:2]}
    data = []
    for gi, g in enumerate(graphs):
        key = (g["subject"], g["record"], g["trial"])
        wins = cycle_windows(g, dirs[key])
        start, end = max(length, g["rise"] - PRE), min(g["n"] - 1, wins[-1][2] + 100)
        label = np.full(g["n"], "standing", dtype=object)
        for name, a, b, _ in wins:
            label[a:b] = name
        fmg = g["fmg"] - g["baseline"]
        ends = np.arange(start, end, STRIDE)
        idx = ends[:, None] - np.arange(length)[::-1][None, :]
        bad = g["estimated"][idx].any(axis=(1, 2)) | ~np.isfinite(fmg[idx]).all(axis=(1, 2))
        ends, idx = ends[~bad], idx[~bad]
        if not len(ends):
            continue
        events = {"GI": g["rise"], "transition": g["valleys"][5], "GT": g["valleys"][11]}
        data.append({"graph": gi, "subject": g["subject"], "band": f"{g['subject']}/{g['side']}", "calibration": key in calibration,
                     "X": window_features(fmg[idx]), "y": label[ends], "t": ends, "events": events,
                     "event_ends": {"GI": g["valleys"][1], "transition": g["valleys"][6], "GT": wins[-1][2]},
                     "transition_class": dirs[key]})
    return data


def delays(pred, t, d):
    out = {}
    for event, target in (("GI", "GI"), ("transition", d["transition_class"]), ("GT", "GT")):
        e, stop = d["events"][event], d["event_ends"][event]
        hit = (pred == target) & (t >= e) & (t <= stop)
        run = np.convolve(hit.astype(int), np.ones(3, int), "valid") == 3
        first = np.flatnonzero(run)
        out[event] = (t[first[0] + 2] - e) / 100.0 if len(first) else np.nan
    return out


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    summary = []
    for length, models in ((30, ("LDA", "HGB")), (20, ("LDA",))):
        data = build(graphs, dirs, length)
        X = np.concatenate([d["X"] for d in data]); y = np.concatenate([d["y"] for d in data])
        bands = np.concatenate([[d["band"]] * len(d["y"]) for d in data])
        is_cal = np.concatenate([[d["calibration"]] * len(d["y"]) for d in data])
        subjects = np.concatenate([[d["subject"]] * len(d["y"]) for d in data])
        sub_stride = np.concatenate([np.arange(len(d["y"])) % (TRAIN_STRIDE // STRIDE) == 0 for d in data])
        for band in set(bands):   # calibration standardisation per subject and band
            m = bands == band; ref = X[m & is_cal] if (m & is_cal).sum() > 50 else X[m]
            X[m] = (X[m] - ref.mean(0)) / (ref.std(0) + 1e-6)
        print(f"window {length * 10} ms: {len(y)} windows; per class:", {c: int((y == c).sum()) for c in LABELS}, flush=True)
        for model_name in models:
            t0 = time.time()
            pred = np.empty(len(y), dtype=object)
            for _, train, test in loso_splits(subjects):
                train = train[sub_stride[train]]
                model = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") if model_name == "LDA" else
                         HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=0))
                model.fit(X[train], y[train])
                pred[test] = model.predict(X[test])
            scored = ~is_cal
            per = {s: scores(y[scored & (subjects == s)], pred[scored & (subjects == s)], LABELS) for s in sorted(set(subjects))}
            ba = np.array([v["balanced_accuracy"] for v in per.values()]); f1 = np.array([v["macro_f1"] for v in per.values()])
            # event delays per graph
            pos, ev = 0, collections.defaultdict(list)
            for d in data:
                k = len(d["y"])
                if not d["calibration"]:
                    for name, value in delays(pred[pos:pos + k], d["t"], d).items():
                        ev[name].append(value)
                pos += k
            row = {"window_ms": length * 10, "model": model_name,
                   "balanced_accuracy_mean": round(ba.mean() * 100, 1), "balanced_accuracy_sd": round(ba.std() * 100, 1),
                   "macro_f1_mean": round(f1.mean() * 100, 1), "macro_f1_sd": round(f1.std() * 100, 1),
                   **{k: round(v * 100, 1) for k, v in scores(y[scored], pred[scored], LABELS).items() if k.startswith("recall_")}}
            for name, vals in ev.items():
                vals = np.array(vals, dtype=float)
                row[f"{name}_detected_pct"] = round(np.isfinite(vals).mean() * 100, 1)
                row[f"{name}_delay_median_s"] = round(float(np.nanmedian(vals)), 2) if np.isfinite(vals).any() else ""
            summary.append(row)
            print(row, f"({time.time() - t0:.0f}s)", flush=True)
            np.savetxt(RESULTS / f"realtime_{length * 10}ms_{model_name}_confusion.csv", confusion(y[scored], pred[scored], LABELS),
                       fmt="%d", delimiter=",", header=",".join(LABELS))
            with (RESULTS / f"realtime_{length * 10}ms_{model_name}_per_subject.csv").open("w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["subject", "balanced_accuracy", "macro_f1"])
                for s, v in per.items():
                    w.writerow([s, round(v["balanced_accuracy"] * 100, 1), round(v["macro_f1"] * 100, 1)])
    with (RESULTS / "realtime_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)), lineterminator="\n")
        w.writeheader(); w.writerows(summary)
