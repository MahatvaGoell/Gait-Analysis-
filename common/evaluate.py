"""Splits and metrics shared by both Study 1 tracks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def loso_splits(subjects: np.ndarray):
    """Leave-one-subject-out: every recording and both limbs of one subject form the test fold."""
    for held_out in sorted(set(subjects)):
        yield held_out, np.flatnonzero(subjects != held_out), np.flatnonzero(subjects == held_out)


def within_subject_splits(subjects: np.ndarray, trials: np.ndarray, folds: int = 5, seed: int = 0):
    """Within one subject: whole trials (both limbs, all their cycles) are held out together."""
    rng = np.random.default_rng(seed)
    for subject in sorted(set(subjects)):
        idx = np.flatnonzero(subjects == subject)
        unique = np.array(sorted(set(trials[idx])))
        rng.shuffle(unique)
        for k, chunk in enumerate(np.array_split(unique, folds)):
            test = idx[np.isin(trials[idx], chunk)]
            train = idx[~np.isin(trials[idx], chunk)]
            yield f"{subject}#{k}", train, test


def scores(y_true, y_pred, labels) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        **{f"recall_{c}": r for c, r in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))},
    }


def confusion(y_true, y_pred, labels) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
