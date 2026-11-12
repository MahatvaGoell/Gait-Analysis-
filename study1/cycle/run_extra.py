"""Study 1, per-gait-cycle track: further results of the SVM (128 FMG features).

* per-class precision, recall and F1, accuracy and Cohen's kappa (LOSO and within-subject)
* training and prediction time
* thigh compartments (sensor placement S1..S8 as in the FMG band): each compartment alone and
  the band without it; and every single channel alone

Run:  python -m study1.cycle.run_extra   ->  results/extra_*.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores, within_subject_splits  # noqa: E402
from common.labels import CLASSES, directions  # noqa: E402
from study1.cycle.run_cycle import MODELS, examples, features  # noqa: E402

RESULTS = ROOT / "results"
COMPARTMENTS = {"anterior (S1, S2, S8)": [0, 1, 7], "medial (S3, S4)": [2, 3],
                "posterior (S5, S6)": [4, 5], "lateral (S7)": [6]}


def evaluate(X, y, splits):
    pred = np.empty(len(y), dtype=object)
    fit_s, predict_s = [], []
    for _, train, test in splits:
        model = MODELS["SVM"]()
        t0 = time.perf_counter(); model.fit(X[train], y[train]); fit_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pred[test] = model.predict(X[test]); predict_s.append((time.perf_counter() - t0) / len(test))
    return pred, float(np.mean(fit_s)), float(np.mean(predict_s))


def per_subject_ba(y, pred, subjects):
    v = [scores(y[subjects == s], pred[subjects == s], CLASSES)["balanced_accuracy"] for s in sorted(set(subjects))]
    return np.mean(v) * 100, np.std(v) * 100, v


if __name__ == "__main__":
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    curves, meta, _ = examples(graphs, dirs)
    subjects = np.array([m[0] for m in meta]); trials = np.array([f"{m[0]}/{m[1]}" for m in meta]); y = np.array([m[3] for m in meta])
    X = np.array([features(c, True) for c in curves])

    rows = []
    for protocol in ("LOSO", "within-subject"):
        splits = loso_splits(subjects) if protocol == "LOSO" else within_subject_splits(subjects, trials)
        pred, fit_s, pred_s = evaluate(X, y, splits)
        p, r, f, n = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
        ba_m, ba_s, per = per_subject_ba(y, pred, subjects)
        print(f"{protocol}: accuracy {np.mean(y == pred) * 100:.1f}  kappa {cohen_kappa_score(y, pred):.3f}  BA {ba_m:.1f} ± {ba_s:.1f}  "
              f"fit {fit_s:.2f} s/fold  predict {pred_s * 1e3:.3f} ms/example")
        print("  per-subject BA:", [round(v * 100, 1) for v in per])
        for c, pi, ri, fi, ni in zip(CLASSES, p, r, f, n):
            rows.append({"protocol": protocol, "class": c, "precision": round(pi * 100, 1), "recall": round(ri * 100, 1), "f1": round(fi * 100, 1), "support": int(ni)})
            print(f"  {c:14s} P {pi * 100:5.1f}  R {ri * 100:5.1f}  F1 {fi * 100:5.1f}  n {ni}")
        rows.append({"protocol": protocol, "class": "overall", "precision": "", "recall": "", "f1": "",
                     "support": len(y), "accuracy": round(np.mean(y == pred) * 100, 1), "kappa": round(cohen_kappa_score(y, pred), 3),
                     "fit_seconds_per_fold": round(fit_s, 2), "predict_ms_per_example": round(pred_s * 1e3, 3)})
        np.savetxt(RESULTS / f"extra_svm_{protocol}_confusion.csv", confusion(y, pred, CLASSES), fmt="%d", delimiter=",", header=",".join(CLASSES))
    with (RESULTS / "extra_svm_per_class.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    subsets = {"all 8 channels": list(range(8))}
    for name, ch in COMPARTMENTS.items():
        subsets[f"{name} only"] = ch
        subsets[f"without {name.split(' (')[0]}"] = [c for c in range(8) if c not in ch]
    for c in range(8):
        subsets[f"S{c + 1} only"] = [c]
    rows = []
    for name, ch in subsets.items():
        Xs = np.array([features(cv[:, ch], True) for cv in curves])
        out = {"channels": name, "n_channels": len(ch)}
        for protocol in ("LOSO", "within-subject"):
            splits = loso_splits(subjects) if protocol == "LOSO" else within_subject_splits(subjects, trials)
            pred, _, _ = evaluate(Xs, y, splits)
            m, s, _ = per_subject_ba(y, pred, subjects)
            out[f"{protocol}_BA_mean"] = round(m, 1); out[f"{protocol}_BA_sd"] = round(s, 1)
        rows.append(out)
        print(f"{name:32s} LOSO {out['LOSO_BA_mean']:5.1f} ± {out['LOSO_BA_sd']:4.1f}   within {out['within-subject_BA_mean']:5.1f} ± {out['within-subject_BA_sd']:4.1f}", flush=True)
    with (RESULTS / "extra_svm_channels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
