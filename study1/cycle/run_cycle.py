"""Study 1, per-gait-cycle track: classify each labelled window from the limb's 8 FMG channels only.

Input of one example: the FMG of one window (GI, one step, or GT), minus the trial's quiet-standing
baseline (median of the 0.5 s before the rise, available before walking starts), resampled to 100 points.
Features per channel: mean, SD, RMS, 10-90th percentile range, mean absolute successive difference,
linear slope (48), optionally plus the waveform at 10 time points (80 more, "shape").
Models use fixed settings chosen before testing; scaling is fitted on the training fold only.
Windows containing estimated FMG samples are left out (measured data only).

Run:  python -m study1.cycle.run_cycle    ->  results/cycle_*.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores, within_subject_splits  # noqa: E402
from common.labels import CLASSES, cycle_windows, directions  # noqa: E402

RESULTS = ROOT / "results"
POINTS = 100


def resample(segment: np.ndarray) -> np.ndarray:
    x = np.linspace(0, len(segment) - 1, POINTS)
    out = np.empty((POINTS, segment.shape[1]))
    for c in range(segment.shape[1]):
        col = segment[:, c]
        ok = np.isfinite(col)
        out[:, c] = np.interp(x, np.flatnonzero(ok), col[ok]) if ok.sum() > 1 else 0.0
    return out


def features(curve: np.ndarray, shape: bool) -> np.ndarray:
    t = np.linspace(0, 1, len(curve))
    slope = np.polyfit(t, curve, 1)[0]
    parts = [curve.mean(0), curve.std(0), np.sqrt((curve ** 2).mean(0)),
             np.percentile(curve, 90, axis=0) - np.percentile(curve, 10, axis=0),
             np.abs(np.diff(curve, axis=0)).mean(0), slope]
    if shape:
        parts.append(curve[np.linspace(5, POINTS - 5, 10).astype(int)].T.ravel())
    return np.concatenate(parts)


def examples(graphs: list[dict], dirs: dict):
    rows, curves, meta = [], [], []
    dropped = 0
    for g in graphs:
        direction = dirs[(g["subject"], g["record"], g["trial"])]
        for label, a, b, position in cycle_windows(g, direction):
            if g["estimated"][a:b + 1].any():
                dropped += 1
                continue
            seg = g["fmg"][a:b + 1] - g["baseline"]
            curves.append(resample(seg))
            meta.append((g["subject"], f"{g['record']}#{g['trial']}", g["side"], label, position, (b - a) / 100.0))
    return np.array(curves), meta, dropped


MODELS = {
    "LDA": lambda: make_pipeline(RobustScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "SVM": lambda: make_pipeline(RobustScaler(), StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")),
    "RF": lambda: make_pipeline(RobustScaler(), RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=8, random_state=0)),
    "HGB": lambda: make_pipeline(RobustScaler(), HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=0)),
}


def run(X, y, groups_iter, model_name):
    preds = np.empty(len(y), dtype=object)
    per_fold = []
    for name, train, test in groups_iter:
        model = MODELS[model_name]()
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test])
        per_fold.append((name, scores(y[test], preds[test], CLASSES)))
    return preds, per_fold


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    curves, meta, dropped = examples(graphs, dirs)
    subjects = np.array([m[0] for m in meta]); trials = np.array([f"{m[0]}/{m[1]}" for m in meta])
    y = np.array([m[3] for m in meta]); duration = np.array([m[5] for m in meta])
    print(f"{len(y)} measured windows ({dropped} with estimated FMG left out); per class:",
          {c: int((y == c).sum()) for c in CLASSES})
    feature_sets = {
        "stats48": np.array([features(c, False) for c in curves]),
        "stats+shape128": np.array([features(c, True) for c in curves]),
    }
    feature_sets["stats+shape128+duration"] = np.column_stack([feature_sets["stats+shape128"], duration])
    summary = []
    for fs, X in feature_sets.items():
        for model_name in MODELS:
            for protocol in ("LOSO", "within-subject"):
                if protocol == "within-subject" and fs != "stats+shape128":
                    continue
                t0 = time.time()
                splits = loso_splits(subjects) if protocol == "LOSO" else within_subject_splits(subjects, trials)
                preds, per_fold = run(X, y, splits, model_name)
                if protocol == "within-subject":   # summarise per subject, not per fold
                    per_subject = {}
                    for s in sorted(set(subjects)):
                        m = subjects == s
                        per_subject[s] = scores(y[m], preds[m], CLASSES)
                    per_fold = list(per_subject.items())
                ba = np.array([s["balanced_accuracy"] for _, s in per_fold]); f1 = np.array([s["macro_f1"] for _, s in per_fold])
                pooled = scores(y, preds, CLASSES)
                row = {"features": fs, "model": model_name, "protocol": protocol,
                       "balanced_accuracy_mean": round(ba.mean() * 100, 1), "balanced_accuracy_sd": round(ba.std() * 100, 1),
                       "macro_f1_mean": round(f1.mean() * 100, 1), "macro_f1_sd": round(f1.std() * 100, 1),
                       **{k: round(v * 100, 1) for k, v in pooled.items() if k.startswith("recall_")}}
                summary.append(row)
                print(f"{fs:26s} {model_name:4s} {protocol:15s} BA {row['balanced_accuracy_mean']:5.1f} ± {row['balanced_accuracy_sd']:4.1f}  "
                      f"F1 {row['macro_f1_mean']:5.1f} ± {row['macro_f1_sd']:4.1f}  ({time.time() - t0:.0f}s)", flush=True)
                if protocol == "LOSO":
                    with (RESULTS / f"cycle_loso_{fs}_{model_name}_per_subject.csv").open("w", newline="") as f:
                        w = csv.writer(f, lineterminator="\n")
                        w.writerow(["subject", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES])
                        for name, s in per_fold:
                            w.writerow([name, round(s["balanced_accuracy"] * 100, 1), round(s["macro_f1"] * 100, 1)] +
                                       [round(s[f"recall_{c}"] * 100, 1) for c in CLASSES])
                    np.savetxt(RESULTS / f"cycle_loso_{fs}_{model_name}_confusion.csv", confusion(y, preds, CLASSES),
                               fmt="%d", delimiter=",", header=",".join(CLASSES))
    with (RESULTS / "cycle_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader(); w.writerows(summary)
