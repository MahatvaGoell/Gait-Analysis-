"""Study 1, per-gait-cycle track: tuned baselines with nested LOSO (task O3 in TRAINING_PLAN.md).

Same windows and features as run_cycle.py (128 FMG features + window duration). For every held-out subject,
the settings are chosen by an inner leave-one-subject-out over the 7 training subjects (mean balanced
accuracy), refitted on all 7 and tested once on the held-out subject. The held-out subject is never used
for choosing. The fixed settings of run_cycle.py are scored on the same folds for comparison.

Run:  python -m study1.cycle.run_tuning   ->  results/cycle_tuning_*.csv
"""
from __future__ import annotations

import csv
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores  # noqa: E402
from common.labels import CLASSES, directions  # noqa: E402
from study1.cycle.run_cycle import examples, features  # noqa: E402

RESULTS = ROOT / "results"


def svm(C, gamma_factor):
    # after StandardScaler every feature has unit variance, so gamma="scale" equals 1 / n_features
    return make_pipeline(RobustScaler(), StandardScaler(),
                         SVC(C=C, gamma=gamma_factor / 129, class_weight="balanced"))


def hgb(learning_rate, max_depth):
    return make_pipeline(RobustScaler(), HistGradientBoostingClassifier(
        class_weight="balanced", max_iter=300, learning_rate=learning_rate, max_depth=max_depth, random_state=0))


GRIDS = {
    "SVM": (svm, {"C": [1, 3, 10, 30, 100], "gamma_factor": [0.25, 0.5, 1, 2, 4]}, {"C": 10, "gamma_factor": 1}),
    "HGB": (hgb, {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [None, 3, 6]}, {"learning_rate": 0.1, "max_depth": None}),
}


def inner_score(make, params, X, y, subjects):
    ba = []
    for _, train, test in loso_splits(subjects):
        model = make(**params).fit(X[train], y[train])
        ba.append(balanced_accuracy_score(y[test], model.predict(X[test])))
    return float(np.mean(ba))


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    curves, meta, _ = examples(graphs, dirs)
    subjects = np.array([m[0] for m in meta]); y = np.array([m[3] for m in meta])
    X = np.column_stack([np.array([features(c, True) for c in curves]), [m[5] for m in meta]])
    print(f"{len(y)} measured windows, {X.shape[1]} features", flush=True)

    summary, chosen_rows = [], []
    for model_name, (make, grid, fixed) in GRIDS.items():
        t0 = time.time()
        candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
        pred = {"fixed": np.empty(len(y), dtype=object), "tuned": np.empty(len(y), dtype=object)}
        for held_out, train, test in loso_splits(subjects):
            inner = Parallel(n_jobs=6)(delayed(inner_score)(make, p, X[train], y[train], subjects[train]) for p in candidates)
            best = candidates[int(np.argmax(inner))]
            chosen_rows.append({"model": model_name, "held_out": held_out, **{k: best[k] for k in grid},
                                "inner_balanced_accuracy": round(max(inner) * 100, 1)})
            for kind, params in (("fixed", fixed), ("tuned", best)):
                pred[kind][test] = make(**params).fit(X[train], y[train]).predict(X[test])
            print(f"{model_name} {held_out}: chose {best} (inner BA {max(inner) * 100:.1f})  ({time.time() - t0:.0f}s)", flush=True)
        for kind, p in pred.items():
            per = {s: scores(y[subjects == s], p[subjects == s], CLASSES) for s in sorted(set(subjects))}
            ba = np.array([v["balanced_accuracy"] for v in per.values()]); f1 = np.array([v["macro_f1"] for v in per.values()])
            row = {"model": model_name, "settings": kind,
                   "balanced_accuracy_mean": round(ba.mean() * 100, 1), "balanced_accuracy_sd": round(ba.std() * 100, 1),
                   "macro_f1_mean": round(f1.mean() * 100, 1), "macro_f1_sd": round(f1.std() * 100, 1),
                   **{k: round(v * 100, 1) for k, v in scores(y, p, CLASSES).items() if k.startswith("recall_")}}
            summary.append(row)
            print(f"{model_name} {kind:5s} BA {row['balanced_accuracy_mean']:5.1f} ± {row['balanced_accuracy_sd']:4.1f}  "
                  f"F1 {row['macro_f1_mean']:5.1f} ± {row['macro_f1_sd']:4.1f}", flush=True)
            np.savetxt(RESULTS / f"cycle_tuning_{model_name}_{kind}_confusion.csv", confusion(y, p, CLASSES),
                       fmt="%d", delimiter=",", header=",".join(CLASSES))
    with (RESULTS / "cycle_tuning_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader(); w.writerows(summary)
    with (RESULTS / "cycle_tuning_chosen.csv").open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in chosen_rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(chosen_rows)
