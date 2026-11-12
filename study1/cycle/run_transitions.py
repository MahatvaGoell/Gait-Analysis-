"""Study 1, per-gait-cycle track: methods aimed at the transition classes (task O2 in TRAINING_PLAN.md).

All methods use the windows and features of run_cycle.py (128 FMG features + window duration), LOSO,
and are scored on the same measured windows, so they compare directly with results/cycle_summary.csv.

* single window   one window at a time (the run_cycle.py result, repeated here as the reference)
* history         the window's features plus their change from the previous 1 and 2 windows of the same
                  limb graph (only past data; before the first step the 1 s of quiet standing before the
                  rise is used). A previous window that contains estimated FMG gives no change (zeros) and
                  sets a flag, so estimated data never enter the input.
* two-stage       first GI / steady step / transition / GT, then short vs long for steady steps and
                  short-to-long vs long-to-short for transitions; with single-window and history features.
* order filter    causal forward filtering of the history model's class probabilities with the order of
                  the walking protocol (GI -> steady steps -> one transition -> steady steps of the other
                  length -> GT). It uses only past windows, but it uses knowledge of the protocol, so it is
                  reported separately.

Run:  python -m study1.cycle.run_transitions   ->  results/cycle_transitions_*.csv
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.evaluate import confusion, loso_splits, scores  # noqa: E402
from common.labels import CLASSES, cycle_windows, directions  # noqa: E402
from study1.cycle.run_cycle import features, resample  # noqa: E402

RESULTS = ROOT / "results"
STEADY = {"short", "long"}
TRANSITION = {"short_to_long", "long_to_short"}


def stage(label: str) -> str:
    return "steady" if label in STEADY else "transition" if label in TRANSITION else label


MODELS = {
    "SVM": lambda: make_pipeline(RobustScaler(), StandardScaler(),
                                 SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=0)),
    "HGB": lambda: make_pipeline(RobustScaler(), HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=0)),
}


def build(graphs, dirs):
    """One row per labelled window, in walk order, with the features of the two windows before it."""
    rows = []
    for gi, g in enumerate(graphs):
        windows = cycle_windows(g, dirs[(g["subject"], g["record"], g["trial"])])
        standing = (max(0, g["rise"] - 100), g["rise"])
        f_stand = np.append(features(resample(g["fmg"][standing[0]:standing[1] + 1] - g["baseline"]), True), 1.0)
        stand_ok = not g["estimated"][standing[0]:standing[1] + 1].any()
        past = [(f_stand, stand_ok), (f_stand, stand_ok)]
        for k, (label, a, b, _) in enumerate(windows):
            measured = not g["estimated"][a:b + 1].any()
            f = np.append(features(resample(g["fmg"][a:b + 1] - g["baseline"]), True), (b - a) / 100.0)
            rows.append({"graph": gi, "position": k, "subject": g["subject"], "label": label, "measured": measured,
                         "f": f, "prev1": past[-1], "prev2": past[-2]})
            past.append((f, measured))
    return rows


def history(rows):
    out = []
    for r in rows:
        parts = [r["f"]]
        for prev, ok in (r["prev1"], r["prev2"]):
            parts.append(r["f"] - prev if ok and r["measured"] else np.zeros_like(r["f"]))
            parts.append([float(ok)])
        out.append(np.concatenate(parts))
    return np.array(out)


def fit_predict(model_name, X, y, train, test):
    model = MODELS[model_name]()
    model.fit(X[train], y[train])
    return model.predict(X[test]), model.predict_proba(X[test]), list(model.classes_)


# protocol order: hidden states and the classes they emit
STATES = ["GI", "short_b", "long_b", "short_to_long", "long_to_short", "short_a", "long_a", "GT"]
EMITS = {"GI": "GI", "short_b": "short", "long_b": "long", "short_to_long": "short_to_long",
         "long_to_short": "long_to_short", "short_a": "short", "long_a": "long", "GT": "GT"}
NEXT = {"GI": ["short_b", "long_b"], "short_b": ["short_b", "short_to_long"], "long_b": ["long_b", "long_to_short"],
        "short_to_long": ["long_a"], "long_to_short": ["short_a"], "short_a": ["short_a", "GT"],
        "long_a": ["long_a", "GT"], "GT": ["GT"]}


def order_filter(proba: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Causal forward filtering over one limb graph. proba: (windows, CLASSES); observed: windows with a
    measured window (others carry no evidence). Returns the class predicted at each window from the past only."""
    n = len(STATES)
    T = np.zeros((n, n))
    for i, s in enumerate(STATES):
        for t in NEXT[s]:
            T[i, STATES.index(t)] = 1.0 / len(NEXT[s])
    emit_index = [CLASSES.index(EMITS[s]) for s in STATES]
    belief = np.full(n, 1.0 / n)
    out = []
    for k in range(len(proba)):
        if k:
            belief = belief @ T
        if observed[k]:
            belief = belief * np.clip(proba[k, emit_index], 1e-6, None)
        belief /= belief.sum()
        class_belief = np.zeros(len(CLASSES))
        for i, s in enumerate(STATES):
            class_belief[CLASSES.index(EMITS[s])] += belief[i]
        out.append(CLASSES[int(class_belief.argmax())])
    return np.array(out, dtype=object)


def summarise(name, model_name, y, pred, subjects, scored):
    per = {s: scores(y[scored & (subjects == s)], pred[scored & (subjects == s)], CLASSES) for s in sorted(set(subjects))}
    ba = np.array([v["balanced_accuracy"] for v in per.values()]); f1 = np.array([v["macro_f1"] for v in per.values()])
    pooled = scores(y[scored], pred[scored], CLASSES)
    row = {"method": name, "model": model_name,
           "balanced_accuracy_mean": round(ba.mean() * 100, 1), "balanced_accuracy_sd": round(ba.std() * 100, 1),
           "macro_f1_mean": round(f1.mean() * 100, 1), "macro_f1_sd": round(f1.std() * 100, 1),
           **{k: round(v * 100, 1) for k, v in pooled.items() if k.startswith("recall_")}}
    tag = f"{name}_{model_name}".replace(" ", "_")
    with (RESULTS / f"cycle_transitions_{tag}_per_subject.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subject", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES])
        for s, v in per.items():
            w.writerow([s, round(v["balanced_accuracy"] * 100, 1), round(v["macro_f1"] * 100, 1)] + [round(v[f"recall_{c}"] * 100, 1) for c in CLASSES])
    np.savetxt(RESULTS / f"cycle_transitions_{tag}_confusion.csv", confusion(y[scored], pred[scored], CLASSES),
               fmt="%d", delimiter=",", header=",".join(CLASSES))
    print(f"{name:22s} {model_name:4s} BA {row['balanced_accuracy_mean']:5.1f} ± {row['balanced_accuracy_sd']:4.1f}  "
          f"F1 {row['macro_f1_mean']:5.1f}  recall S->L {row['recall_short_to_long']:5.1f}  L->S {row['recall_long_to_short']:5.1f}", flush=True)
    return row


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    dirs, _ = directions(graphs)
    rows = build(graphs, dirs)
    subjects = np.array([r["subject"] for r in rows]); y = np.array([r["label"] for r in rows], dtype=object)
    graph_id = np.array([r["graph"] for r in rows]); scored = np.array([r["measured"] for r in rows])
    X_single = np.array([r["f"] for r in rows]); X_hist = history(rows)
    y_stage = np.array([stage(c) for c in y], dtype=object)
    print(f"{scored.sum()} measured windows are scored (of {len(y)}); per class:", {c: int(((y == c) & scored).sum()) for c in CLASSES})

    summary = []
    for model_name in MODELS:
        t0 = time.time()
        preds = {k: np.empty(len(y), dtype=object) for k in ("single window", "history", "two-stage", "two-stage+history", "order filter")}
        proba_hist = np.zeros((len(y), len(CLASSES)))
        for _, train, test in loso_splits(subjects):
            train = train[scored[train]]                      # train on measured windows only
            preds["single window"][test] = fit_predict(model_name, X_single, y, train, test)[0]
            p, proba, classes = fit_predict(model_name, X_hist, y, train, test)
            preds["history"][test] = p
            proba_hist[test] = proba[:, [classes.index(c) for c in CLASSES]]
            for key, X in (("two-stage", X_single), ("two-stage+history", X_hist)):
                first = fit_predict(model_name, X, y_stage, train, test)[0]
                out = first.copy()
                for group in (STEADY, TRANSITION):
                    tr = train[np.isin(y[train], list(group))]
                    te = test[np.isin(first, ["steady" if group is STEADY else "transition"])]
                    if len(te):
                        out[np.isin(test, te)] = fit_predict(model_name, X, y, tr, te)[0]
                preds[key][test] = out
        for gid in np.unique(graph_id):
            m = np.flatnonzero(graph_id == gid)
            preds["order filter"][m] = order_filter(proba_hist[m], scored[m])
        for name, pred in preds.items():
            summary.append(summarise(name, model_name, y, pred, subjects, scored))
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    with (RESULTS / "cycle_transitions_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader(); w.writerows(summary)
