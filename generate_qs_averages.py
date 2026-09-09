"""Create mean QS plots for every source recording from the real trial data.

Each mean is calculated separately for the left and right limb.  Individual
QS windows are resampled to the same 0–100% QS phase before their pointwise
mean is calculated; no PNG files are used as input.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np

from generate_qs_plots import (
    DEFAULT_OUTPUT,
    DATASET_ROOT,
    ROOT,
    SUBJECT_RE,
    detect_qs_end,
    load_stream,
    map_qs_to_insole,
    natural_key,
    output_folder_name,
    render_plot,
    split_source_files,
    trigger_sync,
    unit_trial_starts,
)


NORMALIZED_SAMPLES = 300


def resample_qs(values: np.ndarray, target_samples: int = NORMALIZED_SAMPLES) -> np.ndarray:
    """Interpolate one QS window onto a common 0–100% QS timeline."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot resample an empty QS window.")
    if len(values) == 1:
        return np.repeat(values, target_samples, axis=0)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, target_samples)
    return np.column_stack([np.interp(new_x, old_x, values[:, channel]) for channel in range(values.shape[1])])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_averages(output_dir: Path, subjects: set[str] | None = None) -> dict[str, int]:
    avg_dir = output_dir / "avg"
    avg_dir.mkdir(parents=True, exist_ok=True)
    per_trial_avg_dir = avg_dir / "per trialavg"
    per_trial_avg_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    subject_dirs = sorted(
        (path for path in DATASET_ROOT.iterdir() if path.is_dir() and SUBJECT_RE.match(path.name)),
        key=natural_key,
    )
    if subjects:
        subject_dirs = [path for path in subject_dirs if path.name in subjects]
    for subject_dir in subject_dirs:
        fmg_files, insole_files = split_source_files(subject_dir)
        for base, fmg_path in sorted(fmg_files.items(), key=lambda item: natural_key(item[1])):
            fmg = load_stream(fmg_path, 18, "FMG")
            trials = unit_trial_starts(fmg)
            if not trials:
                continue

            for side in ("L", "R"):
                insole_path = insole_files.get((base, side))
                insole = load_stream(insole_path, 4, "insole", side) if insole_path else None
                sync_status, _ = trigger_sync(fmg, insole)
                if insole is None or not insole.rising_edges:
                    rows.append({
                        "subject": subject_dir.name,
                        "source_data_file": fmg_path.name,
                        "side": side.lower(),
                        "status": "not created: missing insole trigger",
                        "trials_averaged": 0,
                        "png": "",
                    })
                    continue

                channel_start = 2 if side == "L" else 10
                channels = fmg.values[:, channel_start : channel_start + 8]
                fmg_windows: list[np.ndarray] = []
                cop_windows: list[np.ndarray] = []
                vgrf_windows: list[np.ndarray] = []
                for fmg_start, fmg_trial_end, local_trial in trials:
                    edge = 2 * (local_trial - 1)
                    if edge >= len(insole.rising_edges):
                        continue
                    insole_start = insole.rising_edges[edge]
                    insole_trial_end = insole.rising_edges[edge + 1] if edge + 1 < len(insole.rising_edges) else None
                    scan_stop = fmg_trial_end if fmg_trial_end is not None else min(
                        len(fmg.values), fmg_start + int(round(15 / fmg.median_dt))
                    )
                    fmg_end, _ = detect_qs_end(channels, fmg_start, scan_stop, fmg.median_dt)
                    if fmg_end <= fmg_start + 5:
                        continue
                    insole_qs_start, insole_qs_end, _ = map_qs_to_insole(
                        fmg,
                        insole,
                        fmg_start,
                        fmg_end,
                        fmg_trial_end,
                        insole_start,
                        insole_trial_end,
                    )
                    fmg_windows.append(resample_qs(channels[fmg_start:fmg_end]))
                    cop_windows.append(resample_qs(insole.values[insole_qs_start:insole_qs_end, 2:3]))
                    vgrf_windows.append(resample_qs(insole.values[insole_qs_start:insole_qs_end, 3:4]))

                if not fmg_windows:
                    rows.append({
                        "subject": subject_dir.name,
                        "source_data_file": fmg_path.name,
                        "side": side.lower(),
                        "status": "not created: no usable QS trial",
                        "trials_averaged": 0,
                        "png": "",
                    })
                    continue

                n_trials = len(fmg_windows)
                mean_fmg = np.mean(np.stack(fmg_windows), axis=0)
                mean_cop = np.mean(np.stack(cop_windows), axis=0)[:, 0]
                mean_vgrf = np.mean(np.stack(vgrf_windows), axis=0)[:, 0]
                file_name = f"{subject_dir.name}__{output_folder_name(base)}__{side.lower()}_avg.png"
                destination = per_trial_avg_dir / file_name
                render_plot(
                    destination=destination,
                    subject=subject_dir.name,
                    record=base,
                    trial_number=0,
                    side="left" if side == "L" else "right",
                    fmg_values=mean_fmg,
                    fmg_dt=0.01,
                    cop=mean_cop,
                    vgrf=mean_vgrf,
                    insole_dt=0.01,
                    sync_status=f"mean of {n_trials} trigger-aligned QS trial(s); {sync_status}",
                    onset_status="each trial cut before detected movement onset",
                    average_trials=n_trials,
                    normalized_qs_phase=True,
                )
                rows.append({
                    "subject": subject_dir.name,
                    "source_data_file": fmg_path.name,
                    "side": side.lower(),
                    "status": "created",
                    "trials_averaged": n_trials,
                    "png": file_name,
                })

    write_csv(avg_dir / "average_manifest.csv", rows)
    summary = {
        "average_graphs_created": sum(1 for row in rows if row["status"] == "created"),
        "left_limb_averages": sum(1 for row in rows if row["status"] == "created" and row["side"] == "l"),
        "right_limb_averages": sum(1 for row in rows if row["status"] == "created" and row["side"] == "r"),
        "method": "Mean of real QS trial data within each subject/source-data-file/limb. Each trial is resampled to a common 0–100% QS phase before averaging.",
    }
    (avg_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize_existing(output_dir: Path) -> dict[str, int]:
    """Write a complete index for averages already rendered in small batches."""
    avg_dir = output_dir / "avg"
    rows: list[dict[str, object]] = []
    graph_dir = avg_dir / "per trialavg"
    for path in sorted(graph_dir.glob("*_avg.png"), key=natural_key):
        parts = path.stem.split("__")
        if len(parts) != 3:
            continue
        subject, source_file, side_part = parts
        side = "left" if side_part == "l_avg" else "right" if side_part == "r_avg" else side_part
        rows.append({
            "subject": subject,
            "source_data_file": source_file,
            "side": side,
            "status": "created",
            "png": path.name,
        })
    write_csv(avg_dir / "average_manifest.csv", rows)
    summary = {
        "average_graphs_created": len(rows),
        "left_limb_averages": sum(1 for row in rows if row["side"] == "left"),
        "right_limb_averages": sum(1 for row in rows if row["side"] == "right"),
        "method": "Each image is the mean of real QS trials from one subject/source-data-file/limb. Trials are resampled to a common 0–100% QS phase before averaging.",
    }
    (avg_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", action="append", help="Create averages for this subject only. May be supplied more than once.")
    parser.add_argument("--summarize-existing", action="store_true", help="Rebuild the average-folder index without rendering images.")
    args = parser.parse_args()
    if args.summarize_existing:
        print(json.dumps(summarize_existing(DEFAULT_OUTPUT.resolve()), indent=2))
    else:
        print(json.dumps(generate_averages(DEFAULT_OUTPUT.resolve(), set(args.subject) if args.subject else None), indent=2))
