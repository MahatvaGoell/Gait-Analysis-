"""Build the Study 1 data set of the healthy subjects from the CoP phase cuts.

For every cut limb graph of Sub01_H..Sub08_H this reads the 100 Hz export, phase_cuts.csv and
stance_segments.csv, and stores:

* the limb's 8 FMG channels for the whole recording, with their estimated-sample flags,
* the event samples: rise out of standing, the twelve CoP valleys v0..v11 and the rest point,
* the quiet-standing FMG baseline (median of the 0.5 s before the rise; available before walking starts).

Nothing is re-detected here: every boundary comes from refine_sub08_h_valleys.py.
CoP and vGRF are not stored; they are never a model input.

Run:  python -m common.build_dataset        ->  datasets/healthy_graphs.pkl
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refine_sub08_h_valleys import SUBJECTS, parse_csv  # noqa: E402

OUT = ROOT / "datasets"
HEALTHY = [name for name in sorted(SUBJECTS) if name.endswith("_H")]
STEPS = ["S1", "S2", "S3", "S4", "S5", "T", "L1", "L2", "L3", "L4", "L5"]
BASELINE_SAMPLES = 50


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build() -> list[dict]:
    graphs = []
    for subject in HEALTHY:
        source_root, graphs_root = SUBJECTS[subject]
        valleys: dict[tuple, dict] = {}
        for row in read_rows(graphs_root / "stance_segments.csv"):
            valleys.setdefault((row["record"], int(row["trial"]), row["side"]), {})[row["step"]] = row
        for cut in read_rows(graphs_root / "phase_cuts.csv"):
            if cut["status"] != "cop_valley_cut":
                continue
            key = (cut["record"], int(cut["trial"]), cut["side"])
            steps = valleys[key]
            assert list(steps) == STEPS or sorted(steps) == sorted(STEPS), key
            times, values, estimated = parse_csv(source_root / key[0] / f"trial_{key[1]:02d}_{key[2].lower()}_segmented.csv")
            v = [int(steps[name]["cop_valley_sample"]) for name in STEPS] + [int(steps["L5"]["next_valley_sample"])]
            peaks = [int(steps[name]["cop_peak_sample"]) for name in STEPS]
            rise = int(cut["GI_start_sample"])
            rest = int(np.argmin(np.abs(times - float(cut["rest_s"]))))
            fmg = values[:, :8].astype(np.float32)
            graphs.append({
                "subject": subject, "record": key[0], "trial": key[1], "side": key[2],
                "fmg": fmg, "estimated": estimated[:, :8].copy(),
                "rise": rise, "valleys": v, "peaks": peaks, "rest": rest, "n": len(times),
                "baseline": np.nanmedian(fmg[max(0, rise - BASELINE_SAMPLES):max(rise, 1)], axis=0),
                # CoP excursion of every step (valley to peak), used only to check the walking direction
                "cop_excursion": [abs(float(steps[name]["cop_at_peak"]) - float(steps[name]["cop_at_valley"])) for name in STEPS],
                "review_required": cut.get("review_required") == "1",
                "GT_start_kind": cut.get("GT_start_kind", ""),
            })
    return graphs


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    graphs = build()
    with (OUT / "healthy_graphs.pkl").open("wb") as f:
        pickle.dump(graphs, f)
    subjects = sorted({g["subject"] for g in graphs})
    print(f"{len(graphs)} cut limb graphs from {len(subjects)} healthy subjects -> datasets/healthy_graphs.pkl")
    for s in subjects:
        print(f"  {s}: {sum(g['subject'] == s for g in graphs)} graphs")
