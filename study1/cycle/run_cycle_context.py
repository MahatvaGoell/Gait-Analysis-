"""Study 1, per-gait-cycle track: subject calibration and previous-cycle context.

Two additions to run_cycle.py, both available to a device at run time:

* Calibration: the first two walks in each direction of every subject (both limbs) serve as an
  unlabelled calibration recording. Every feature is standardised with the mean and SD of that
  subject's (and band's) calibration windows. Calibration walks are never scored.
* Context: each window also gets the change of its features from the window before it in the same
  walk (for GI, the 1 s of quiet standing before the rise). Only past data are used.

All configurations are scored on the same test windows (calibration walks removed everywhere).

Run:  python -m study1.cycle.run_cycle_context   ->  results/cycle_context_*.csv
"""
from __future__ import annotations

import collections
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores, within_subject_splits  # noqa: E402
from common.labels import CLASSES, cycle_windows, directions, walk_order  # noqa: E402
from study1.cycle.run_cycle import MODELS, features, resample  # noqa: E402

RESULTS = ROOT / "results"
CALIBRATION_WALKS = 2   # per direction


def build(graphs, dirs):
    order = walk_order(graphs)
    # calibration walks: the earliest walks of each direction, per subject
    walks = collections.defaultdict(list)
    for key, (block, walk) in order.items():
        walks[(key[0], dirs[key])].append(((block[1], walk), key))
    calibration = {key for lst in walks.values() for _, key in sorted(lst)[:CALIBRATION_WALKS]}
    rows = []
    for g in graphs:
        key = (g["subject"], g["record"], g["trial"])
        windows = cycle_windows(g, dirs[key])
        previous = (max(0, g["rise"] - 100), g["rise"])
        for label, a, b, position in windows:
            seg = g["fmg"][a:b + 1] - g["baseline"]
            prev = g["fmg"][previous[0]:previous[1] + 1] - g["baseline"]
            measured = not g["estimated"][a:b + 1].any()
            rows.append({"subject": g["subject"], "band": f"{g['subject']}/{g['side']}", "trial": f"{key[0]}/{key[1]}#{key[2]}",
                         "label": label, "calibration": key in calibration, "measured": measured,
                         "f": features(resample(seg), True), "f_prev": features(resample(prev), True)})
            previous = (a, b)
    return rows


def calibrate(F: np.ndarray, bands: np.ndarray, is_cal: np.ndarray) -> np.ndarray:
    out = np.empty_like(F)
    for band in set(bands):
        m = bands == band
        ref = F[m & is_cal] if (m & is_cal).sum() >= 10 else F[m]
        out[m] = (F[m] - ref.mean(0)) / (ref.std(0) + 1e-6)
    return out


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    rows = build(graphs, dirs)
    keep = np.array([r["measured"] for r in rows])
    rows = [r for r, k in zip(rows, keep) if k]
    subjects = np.array([r["subject"] for r in rows]); bands = np.array([r["band"] for r in rows])
    trials = np.array([r["trial"] for r in rows]); y = np.array([r["label"] for r in rows])
    is_cal = np.array([r["calibration"] for r in rows])
    F = np.array([r["f"] for r in rows]); P = np.array([r["f_prev"] for r in rows])
    Fc, Pc = calibrate(F, bands, is_cal), calibrate(P, bands, is_cal)
    print(f"{len(y)} measured windows; {is_cal.sum()} in calibration walks (never scored); test windows per class:",
          {c: int(((y == c) & ~is_cal).sum()) for c in CLASSES})
    configs = {
        "baseline": F,
        "+calibration": Fc,
        "+context": np.column_stack([F, F - P]),
        "+calibration+context": np.column_stack([Fc, Fc - Pc]),
    }
    summary = []
    for cfg, X in configs.items():
        for model_name in ("LDA", "SVM", "HGB"):
            if model_name == "HGB" and cfg not in ("baseline", "+calibration+context"):
                continue
            for protocol in ("LOSO", "within-subject"):
                if protocol == "within-subject" and cfg not in ("baseline", "+calibration+context"):
                    continue
                t0 = time.time()
                splits = loso_splits(subjects) if protocol == "LOSO" else within_subject_splits(subjects, trials)
                preds = np.empty(len(y), dtype=object)
                for _, train, test in splits:
                    model = MODELS[model_name]()
                    model.fit(X[train], y[train])
                    preds[test] = model.predict(X[test])
                scored = ~is_cal
                per = {s: scores(y[scored & (subjects == s)], preds[scored & (subjects == s)], CLASSES) for s in sorted(set(subjects))}
                ba = np.array([v["balanced_accuracy"] for v in per.values()]); f1 = np.array([v["macro_f1"] for v in per.values()])
                pooled = scores(y[scored], preds[scored], CLASSES)
                row = {"config": cfg, "model": model_name, "protocol": protocol,
                       "balanced_accuracy_mean": round(ba.mean() * 100, 1), "balanced_accuracy_sd": round(ba.std() * 100, 1),
                       "macro_f1_mean": round(f1.mean() * 100, 1), "macro_f1_sd": round(f1.std() * 100, 1),
                       **{k: round(v * 100, 1) for k, v in pooled.items() if k.startswith("recall_")}}
                summary.append(row)
                print(f"{cfg:22s} {model_name:4s} {protocol:15s} BA {row['balanced_accuracy_mean']:5.1f} ± {row['balanced_accuracy_sd']:4.1f}  "
                      f"F1 {row['macro_f1_mean']:5.1f} ± {row['macro_f1_sd']:4.1f}  recall " +
                      " ".join(f"{c[:5]}={row['recall_' + c]:.0f}" for c in CLASSES) + f"  ({time.time() - t0:.0f}s)", flush=True)
                tag = f"{cfg}_{model_name}_{protocol}".replace("+", "plus_")
                with (RESULTS / f"cycle_context_{tag}_per_subject.csv").open("w", newline="") as f:
                    w = csv.writer(f, lineterminator="\n")
                    w.writerow(["subject", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES])
                    for s, v in per.items():
                        w.writerow([s, round(v["balanced_accuracy"] * 100, 1), round(v["macro_f1"] * 100, 1)] + [round(v[f"recall_{c}"] * 100, 1) for c in CLASSES])
                np.savetxt(RESULTS / f"cycle_context_{tag}_confusion.csv", confusion(y[scored], preds[scored], CLASSES),
                           fmt="%d", delimiter=",", header=",".join(CLASSES))
    with (RESULTS / "cycle_context_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader(); w.writerows(summary)
