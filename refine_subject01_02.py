"""Source-preserving CoP graph refinement, scoped to Sub01_H and Sub02_H.

Reuse the reviewed noise-aware engine used for subjects 03/04. Timing is
detected from each trial, never copied from another subject or the reference.
The registry below only changes in memory. No other subject is regenerated.
"""
from pathlib import Path

import refine_subject03_04 as engine

SUBJECTS = ("Sub01_H", "Sub02_H")
old = engine.old
for subject in SUBJECTS:
    root = Path(__file__).resolve().parent / "tempgraphs" / subject.lower()
    old.SUBJECTS[subject] = (root / "segmented_data", root)
engine.SUBJECTS = SUBJECTS

clean_signals = engine.clean_signals
cadence = engine.cadence
load_train = engine.load_train
_base_find_stances = engine.find_stances


def find_stances(z, flags, first, last, stride, main_window=None):
    """Exclude a weak preparatory CoP shift lacking a normal load excursion.

    Uses this recording's own amplitudes and cadence, not fixed event times.
    The extra check was prompted by the native/full-trace review of abh_1/13 L.
    It never inserts an event or changes the measured signal.
    """
    stances, span = _base_find_stances(z, flags, first, last, stride, main_window)
    if len(stances) >= 4:
        np = engine.np
        v, p, _ = stances[0]
        typical = float(np.median([z[b, 8] - z[a, 8] for a, b, _ in stances[1:]]))
        load_span = max(float(np.percentile(z[:, 9], 95) - np.percentile(z[:, 9], 1)), 1e-6)
        load_excursion = float(np.ptp(z[max(0, v-10):min(len(z), p+11), 9]))
        if z[p, 8]-z[v, 8] < .60*typical and load_excursion < .30*load_span:
            stances = stances[1:]
    return stances, span


engine.find_stances = find_stances
segment = engine.segment
draw = engine.draw


def process(subject, backup_root):
    if subject not in SUBJECTS:
        raise ValueError("Only Sub01_H and Sub02_H are in this run's scope")
    source, root = old.SUBJECTS[subject]
    if not source.is_dir() or not (root / "recording_windows.csv").is_file():
        raise FileNotFoundError(f"Missing source exports or trial windows for {subject}")
    return engine.process(subject, backup_root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", nargs="+", choices=SUBJECTS)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()
    for subject in args.subjects:
        process(subject, args.backup_root)
