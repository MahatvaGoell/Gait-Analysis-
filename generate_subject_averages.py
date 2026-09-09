"""Create one whole-subject QS mean graph from all real QS trials per subject."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generate_qs_averages import resample_qs, write_csv
from generate_qs_plots import (
    DATASET_ROOT,
    DEFAULT_OUTPUT,
    SUBJECT_RE,
    detect_qs_end,
    load_stream,
    map_qs_to_insole,
    natural_key,
    render_plot,
    split_source_files,
    trigger_sync,
    unit_trial_starts,
)


def generate_subject_averages(output_dir: Path) -> dict[str, int]:
    target_dir = output_dir / "avg" / "per subject"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    subject_dirs = sorted(
        (path for path in DATASET_ROOT.iterdir() if path.is_dir() and SUBJECT_RE.match(path.name)),
        key=natural_key,
    )
    for subject_dir in subject_dirs:
        fmg_windows: list[np.ndarray] = []
        cop_windows: list[np.ndarray] = []
        vgrf_windows: list[np.ndarray] = []
        contributing_records: set[str] = set()
        review_pairs = 0

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
                    continue
                if not (sync_status.startswith("trigger sequence") or sync_status.startswith("single")):
                    review_pairs += 1

                channel_start = 2 if side == "L" else 10
                channels = fmg.values[:, channel_start : channel_start + 8]
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
                    contributing_records.add(base)

        if not fmg_windows:
            rows.append({
                "subject": subject_dir.name,
                "status": "not created: no usable QS trial",
                "trials_averaged": 0,
                "source_recordings": 0,
                "png": "",
            })
            continue

        mean_fmg = np.mean(np.stack(fmg_windows), axis=0)
        mean_cop = np.mean(np.stack(cop_windows), axis=0)[:, 0]
        mean_vgrf = np.mean(np.stack(vgrf_windows), axis=0)[:, 0]
        trial_count = len(fmg_windows)
        file_name = f"{subject_dir.name}_subject_average.png"
        render_plot(
            destination=target_dir / file_name,
            subject=subject_dir.name,
            record="all recordings",
            trial_number=0,
            side="combined left + right",
            fmg_values=mean_fmg,
            fmg_dt=0.01,
            cop=mean_cop,
            vgrf=mean_vgrf,
            insole_dt=0.01,
            sync_status=(
                f"mean of {trial_count} QS trials from {len(contributing_records)} recording(s), both limbs combined"
                + (f"; {review_pairs} trigger pairing(s) flagged for review" if review_pairs else "")
            ),
            onset_status="each trial cut before detected movement onset",
            average_trials=trial_count,
            normalized_qs_phase=True,
            custom_title=(
                f"Subject Mean Quiet Standing (QS) — {subject_dir.name}  |  all recordings, both limbs  |  n = {trial_count} trials"
            ),
        )
        rows.append({
            "subject": subject_dir.name,
            "status": "created",
            "trials_averaged": trial_count,
            "source_recordings": len(contributing_records),
            "trigger_pairs_flagged_for_review": review_pairs,
            "png": file_name,
        })

    write_csv(target_dir / "subject_average_manifest.csv", rows)
    summary = {
        "subject_average_graphs_created": sum(1 for row in rows if row["status"] == "created"),
        "method": "For each subject, all valid left and right QS trials from every source recording are resampled to a common 0–100% QS phase, then averaged point by point.",
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_subject_averages(DEFAULT_OUTPUT.resolve()), indent=2))
