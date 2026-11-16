"""M2 matched-window ablations. Run with python -m study1.realtime.run_ablations."""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common.evaluate import confusion, loso_splits, scores, within_subject_splits
from common.labels import directions
from study1.cycle.run_extra import COMPARTMENTS
from study1.realtime.run_realtime import LABELS, RESULTS, ROOT, TRAIN_STRIDE, build, delays, false_switches

LENGTHS = (10, 20, 30, 40)
FULL_BAND = tuple(range(8))
LABEL_CODES = np.arange(len(LABELS))
HGB_ITERATIONS = 24  # Compact supplementary sweep, not the M1 HGB configuration.
HGB_TRAIN_STRIDE = 12
EVENTS = ("GI", "transition", "GT")


def channel_sets():
    """Share the cycle track's channel map, including leave-one-compartment-out."""
    groups = {"full_band": FULL_BAND}
    for name, members in COMPARTMENTS.items():
        region = name.split()[0]
        groups[f"{region}_only"] = tuple(members)
        groups[f"without_{region}"] = tuple(c for c in FULL_BAND if c not in members)
    return groups


def feature_columns(channels):
    # The M1 matrix has six successive blocks, each ordered S1..S8.
    return np.array([stat * len(FULL_BAND) + ch for stat in range(6) for ch in channels])


def held_out_predictions(X, y, stride_mask, splits, model_name):
    pred = np.full(len(y), -1, dtype=int)
    for name, train, test in splits:
        if not len(test):
            continue
        train = train[stride_mask[train]]
        if not len(train) or np.any(pred[test] != -1):
            raise ValueError(f"Invalid fold {name}")
        estimator = (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") if model_name == "LDA"
            else HistGradientBoostingClassifier(class_weight="balanced", max_iter=HGB_ITERATIONS,
                                               early_stopping=False, max_bins=63, max_leaf_nodes=15,
                                               random_state=0)
        )
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    if np.any(pred < 0):
        raise ValueError("Some windows did not receive a held-out prediction")
    return pred


def named_scores(true, pred):
    values = scores(true, pred, LABEL_CODES)
    return {key: float(value) for key, value in values.items() if not key.startswith("recall_")} | {
        f"recall_{label}": float(values[f"recall_{i}"]) for i, label in enumerate(LABELS)
    }


def summarize(data, subjects, y, pred, config):
    per_subject = []
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        per_subject.append({**config, "subject": subject, "scored_windows": int(mask.sum()),
                            **{k: round(v * 100, 4) for k, v in named_scores(y[mask], pred[mask]).items()}})

    summary = {**config, "scored_subjects": len(per_subject), "scored_limb_trials": len(data),
               "scored_windows": len(y)}
    for key in ("balanced_accuracy", "macro_f1"):
        values = np.array([r[key] for r in per_subject])
        summary[f"{key}_mean"] = round(float(values.mean()), 2)
        summary[f"{key}_sd"] = round(float(values.std()), 2)
    summary.update({k: round(v * 100, 2) for k, v in named_scores(y, pred).items() if k.startswith("recall_")})

    labels = np.asarray(LABELS)[pred]
    pos, switches = 0, []
    event_delays = {event: [] for event in EVENTS}
    for d in data:
        end = pos + len(d["t"])
        for event, delay in delays(labels[pos:end], d["t"], d).items():
            event_delays[event].append(delay)
        switches.append(false_switches(pred[pos:end], y[pos:end], d["t"]))
        pos = end
    summary["false_switches_per_trial_median"] = float(np.median(switches))
    for event, values in event_delays.items():
        detected = np.asarray(values)[np.isfinite(values)]
        summary[f"{event}_detected_count"] = len(detected)
        summary[f"{event}_detected_pct"] = round(len(detected) / len(data) * 100, 2)
        summary[f"{event}_delay_median_ms"] = round(float(np.median(detected)) * 1000, 1) if len(detected) else ""

    matrix = confusion(y, pred, LABEL_CODES)
    confusion_rows = [{**config, "true_class": label, **dict(zip(LABELS, matrix[i]))}
                      for i, label in enumerate(LABELS)]
    return summary, per_subject, confusion_rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(graphs, model_name, output):
    dirs, _ = directions(graphs)
    groups = channel_sets() if model_name == "LDA" else {"full_band": FULL_BAND}
    summaries, subject_rows, confusion_rows = [], [], []
    canonical = None
    for length in LENGTHS:
        data = build(graphs, dirs, length, match_length=max(LENGTHS))
        signature = [(d["graph"], d["t"].tobytes()) for d in data]
        if canonical is None:
            canonical = signature
        elif signature != canonical:
            raise ValueError("Window-length comparison does not share an identical scoring cohort")
        X = np.concatenate([d["X"] for d in data])
        y = np.concatenate([d["y"] for d in data])
        # Integer codes reduce repeated string work in the shared metric functions.
        y = np.array([LABELS.index(label) for label in y])
        subjects = np.repeat([d["subject"] for d in data], [len(d["y"]) for d in data])
        trials = np.repeat(["/".join(map(str, d["trial_key"])) for d in data], [len(d["y"]) for d in data])
        train_stride = TRAIN_STRIDE if model_name == "LDA" else HGB_TRAIN_STRIDE
        stride_mask = np.concatenate([(d["t"] - d["t"][0]) % train_stride == 0 for d in data])
        splits = {"LOSO": list(loso_splits(subjects))}
        if model_name == "LDA":
            splits["within-subject"] = list(within_subject_splits(subjects, trials))
        for group, channels in groups.items():
            inputs = np.ascontiguousarray(X[:, feature_columns(channels)])
            for protocol, folds in splits.items():
                start = perf_counter()
                config = {"model": model_name, "window_ms": length * 10, "channel_set": group,
                          "channels": " ".join(f"S{i + 1}" for i in channels), "protocol": protocol}
                pred = held_out_predictions(inputs, y, stride_mask, folds, model_name)
                summary, per, matrix = summarize(data, subjects, y, pred, config)
                summaries.append(summary)
                subject_rows.extend(per)
                confusion_rows.extend(matrix)
                print(f"{model_name:3} {length * 10:3} ms {group:18} {protocol:14} "
                      f"BA {summary['balanced_accuracy_mean']:5.2f}% | "
                      f"transition {summary['transition_detected_pct']:5.2f}% | "
                      f"{perf_counter() - start:.1f} s", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"realtime_m2_{model_name.lower()}"
    for name, rows in (("summary", summaries), ("per_subject", subject_rows), ("confusion", confusion_rows)):
        write_csv(output / f"{prefix}_{name}.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("LDA", "HGB"), default="LDA",
                        help="LDA runs the complete channel grid; HGB checks full-band window lengths under LOSO.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with (ROOT / "datasets" / "healthy_graphs.pkl").open("rb") as f:
        graphs = pickle.load(f)
    with threadpool_limits(limits=args.threads):
        run(graphs, args.model, args.output_dir)


if __name__ == "__main__":
    main()
