"""Generate gait-initiation (GI) plots and averages from the raw FMG/insole data.

The Study 1 protocol places quiet standing in the first three seconds after a
trial-start trigger.  GI is therefore the fixed 3.0--4.0 s interval after that
same trigger.  It excludes both QS and later steady-walking samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_qs_averages import resample_qs, write_csv
from generate_qs_plots import (
    DATASET_ROOT,
    ROOT,
    SUBJECT_RE,
    Stream,
    natural_key,
    output_folder_name,
    render_plot,
    load_stream,
    split_source_files,
    trigger_sync,
    unit_trial_starts,
)


GI_OUTPUT = ROOT / "output" / "gioutput"
GI_START_SECONDS = 3.0
GI_END_SECONDS = 4.0
QS_CONTEXT_SECONDS = 0.5
NORMALIZED_SAMPLES = 450


@dataclass
class GiWindow:
    subject: str
    record: str
    side: str
    trial: int
    fmg: np.ndarray
    cop: np.ndarray
    vgrf: np.ndarray
    fmg_dt: float
    insole_dt: float
    sync_status: str
    detection_status: str
    gi_start_fmg_sample: int
    gi_start_insole_sample: int


def map_fixed_interval_to_insole(
    fmg: Stream,
    insole: Stream,
    fmg_interval_start: int,
    fmg_interval_end: int,
    fmg_trial_start: int,
    insole_trial_start: int,
    insole_trial_end: int | None,
) -> tuple[int | None, int | None, str]:
    """Map a fixed elapsed FMG interval to the same elapsed insole interval.

    The calculation intentionally uses the separately recorded sample periods,
    rather than rescaling to the trial end: GI is defined by its elapsed time
    after the matched start trigger.
    """
    start_seconds = (fmg_interval_start - fmg_trial_start) * fmg.median_dt
    end_seconds = (fmg_interval_end - fmg_trial_start) * fmg.median_dt
    mapped_start = insole_trial_start + int(round(start_seconds / insole.median_dt))
    mapped_end = insole_trial_start + int(round(end_seconds / insole.median_dt))
    available_end = min(len(insole.values), insole_trial_end) if insole_trial_end is not None else len(insole.values)
    if mapped_start < 0 or mapped_end > available_end or mapped_end - mapped_start < 20:
        return None, None, "insole trial does not contain the full fixed 3.0-4.0 s GI interval"
    return mapped_start, mapped_end, "matched fixed elapsed time after trial-start trigger"


def render_gi(destination: Path, window: GiWindow, average_trials: int | None = None, custom_title: str | None = None, normalized: bool = False) -> None:
    render_plot(
        destination=destination,
        subject=window.subject,
        record=window.record,
        trial_number=window.trial,
        side="left" if window.side == "L" else "right" if window.side == "R" else "combined left + right",
        fmg_values=window.fmg,
        fmg_dt=window.fmg_dt,
        cop=window.cop,
        vgrf=window.vgrf,
        insole_dt=window.insole_dt,
        sync_status=window.sync_status,
        onset_status=window.detection_status,
        average_trials=average_trials,
        normalized_qs_phase=normalized,
        custom_title=custom_title,
        phase_title="Gait Initiation (GI)",
        phase_short="GI",
        phase_window_label="0.5 s QS reference + fixed GI 3.0–4.0 s interval after trial start",
        phase_note="The dashed boundary marks GI start. The left 0.5 s is the recorded QS reference; the right 1.0 s is the fixed GI interval. A 0.25 s moving average reduces sensor quantisation noise.",
        normalized_phase_label="Normalised QS-to-GI transition (%)",
        baseline_samples=window.gi_start_fmg_sample,
        phase_boundary_sample=window.gi_start_fmg_sample,
        pre_phase_label="QS reference",
    )


def make_gi_window(
    subject: str,
    record: str,
    side: str,
    trial: int,
    fmg: Stream,
    insole: Stream,
    channels: np.ndarray,
    fmg_trial_start: int,
    fmg_trial_end: int | None,
    insole_trial_start: int,
    insole_trial_end: int | None,
    sync_status: str,
) -> tuple[GiWindow | None, str, str]:
    scan_stop = fmg_trial_end if fmg_trial_end is not None else min(
        len(fmg.values), fmg_trial_start + int(round(15 / fmg.median_dt))
    )
    gi_start = fmg_trial_start + int(round(GI_START_SECONDS / fmg.median_dt))
    gi_end = fmg_trial_start + int(round(GI_END_SECONDS / fmg.median_dt))
    plot_start = gi_start - int(round(QS_CONTEXT_SECONDS / fmg.median_dt))
    if gi_end > scan_stop:
        return None, "trial does not contain the full fixed 3.0-4.0 s GI interval", ""
    insole_start, insole_end, alignment_method = map_fixed_interval_to_insole(
        fmg,
        insole,
        plot_start,
        gi_end,
        fmg_trial_start,
        insole_trial_start,
        insole_trial_end,
    )
    if insole_start is None or insole_end is None:
        return None, alignment_method, ""
    detection_status = "fixed protocol GI window: 3.0-4.0 s after trial-start trigger"
    return (
        GiWindow(
            subject=subject,
            record=record,
            side=side,
            trial=trial,
            fmg=channels[plot_start:gi_end],
            cop=insole.values[insole_start:insole_end, 2],
            vgrf=insole.values[insole_start:insole_end, 3],
            fmg_dt=fmg.median_dt,
            insole_dt=insole.median_dt,
            sync_status=sync_status,
            detection_status=detection_status,
            gi_start_fmg_sample=gi_start - plot_start,
            gi_start_insole_sample=int(round(QS_CONTEXT_SECONDS / insole.median_dt)),
        ),
        detection_status,
        alignment_method,
    )


def average_windows(windows: list[GiWindow], subject: str, record: str, side: str, sync_note: str) -> GiWindow:
    fmg = np.mean(np.stack([resample_qs(window.fmg, NORMALIZED_SAMPLES) for window in windows]), axis=0)
    cop = np.mean(np.stack([resample_qs(window.cop[:, None], NORMALIZED_SAMPLES) for window in windows]), axis=0)[:, 0]
    vgrf = np.mean(np.stack([resample_qs(window.vgrf[:, None], NORMALIZED_SAMPLES) for window in windows]), axis=0)[:, 0]
    return GiWindow(
        subject=subject,
        record=record,
        side=side,
        trial=0,
        fmg=fmg,
        cop=cop,
        vgrf=vgrf,
        fmg_dt=0.01,
        insole_dt=0.01,
        sync_status=sync_note,
        detection_status="each trial uses the fixed 3.0-4.0 s GI interval after its trial-start trigger",
        gi_start_fmg_sample=int(round(NORMALIZED_SAMPLES * QS_CONTEXT_SECONDS / (QS_CONTEXT_SECONDS + GI_END_SECONDS - GI_START_SECONDS))),
        gi_start_insole_sample=int(round(NORMALIZED_SAMPLES * QS_CONTEXT_SECONDS / (QS_CONTEXT_SECONDS + GI_END_SECONDS - GI_START_SECONDS))),
    )


def generate(output_dir: Path, subjects: set[str] | None = None, overwrite: bool = False) -> dict[str, int]:
    per_trial_dir = output_dir / "per trial"
    per_record_avg_dir = output_dir / "avg" / "per trialavg"
    per_subject_avg_dir = output_dir / "avg" / "per subject"
    for directory in (per_trial_dir, per_record_avg_dir, per_subject_avg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []
    subject_dirs = sorted(
        (path for path in DATASET_ROOT.iterdir() if path.is_dir() and SUBJECT_RE.match(path.name)),
        key=natural_key,
    )
    if subjects:
        subject_dirs = [path for path in subject_dirs if path.name in subjects]

    for subject_dir in subject_dirs:
        subject_windows: list[GiWindow] = []
        subject_records: set[str] = set()
        fmg_files, insole_files = split_source_files(subject_dir)
        for base, fmg_path in sorted(fmg_files.items(), key=lambda item: natural_key(item[1])):
            fmg = load_stream(fmg_path, 18, "FMG")
            trials = unit_trial_starts(fmg)
            if not trials:
                sync_rows.append({"subject": subject_dir.name, "record": base, "side": "both", "status": "no FMG trigger start"})
                continue
            for side in ("L", "R"):
                insole_path = insole_files.get((base, side))
                insole = load_stream(insole_path, 4, "insole", side) if insole_path else None
                sync_status, p95_error = trigger_sync(fmg, insole)
                sync_rows.append({
                    "subject": subject_dir.name,
                    "record": base,
                    "side": side,
                    "sync_status": sync_status,
                    "p95_relative_trigger_error_seconds": "" if p95_error is None else round(p95_error, 4),
                    "fmg_trigger_rising_edges": len(fmg.rising_edges),
                    "insole_trigger_rising_edges": len(insole.rising_edges) if insole else 0,
                })
                if insole is None or not insole.rising_edges:
                    continue
                channel_start = 2 if side == "L" else 10
                channels = fmg.values[:, channel_start : channel_start + 8]
                record_windows: list[GiWindow] = []
                for fmg_trial_start, fmg_trial_end, local_trial in trials:
                    file_name = f"trial_{local_trial:02d}_{side.lower()}.png"
                    destination = per_trial_dir / subject_dir.name / output_folder_name(base) / file_name
                    edge = 2 * (local_trial - 1)
                    if edge >= len(insole.rising_edges):
                        if overwrite and destination.is_file():
                            destination.unlink()
                        trial_rows.append({"subject": subject_dir.name, "record": base, "side": side, "local_trial": local_trial, "status": "skipped: no corresponding insole trigger", "png": ""})
                        continue
                    insole_trial_start = insole.rising_edges[edge]
                    insole_trial_end = insole.rising_edges[edge + 1] if edge + 1 < len(insole.rising_edges) else None
                    window, detection_status, alignment_method = make_gi_window(
                        subject_dir.name,
                        base,
                        side,
                        local_trial,
                        fmg,
                        insole,
                        channels,
                        fmg_trial_start,
                        fmg_trial_end,
                        insole_trial_start,
                        insole_trial_end,
                        sync_status,
                    )
                    if window is None:
                        # An old onset-detected GI image must not remain when
                        # the fixed protocol interval is unavailable.
                        if overwrite and destination.is_file():
                            destination.unlink()
                        trial_rows.append({"subject": subject_dir.name, "record": base, "side": side, "local_trial": local_trial, "status": f"skipped: {detection_status}", "png": ""})
                        continue
                    if overwrite or not destination.exists() or destination.stat().st_size == 0:
                        render_gi(destination, window)
                        state = "created"
                    else:
                        state = "existing valid PNG preserved"
                    relative_png = destination.relative_to(per_trial_dir).as_posix()
                    trial_rows.append({
                        "subject": subject_dir.name,
                        "record": base,
                        "side": side,
                        "local_trial": local_trial,
                        "status": state,
                        "sync_status": sync_status,
                        "detection_status": detection_status,
                        "alignment_method": alignment_method,
                        "fmg_qs_reference_samples": window.gi_start_fmg_sample,
                        "fmg_gi_samples": len(window.fmg) - window.gi_start_fmg_sample,
                        "fmg_gi_seconds": round((len(window.fmg) - window.gi_start_fmg_sample) * window.fmg_dt, 3),
                        "fmg_display_seconds": round(len(window.fmg) * window.fmg_dt, 3),
                        "insole_qs_reference_samples": window.gi_start_insole_sample,
                        "insole_gi_samples": len(window.cop) - window.gi_start_insole_sample,
                        "insole_gi_seconds": round((len(window.cop) - window.gi_start_insole_sample) * window.insole_dt, 3),
                        "insole_display_seconds": round(len(window.cop) * window.insole_dt, 3),
                        "png": relative_png,
                    })
                    record_windows.append(window)
                    subject_windows.append(window)
                    subject_records.add(base)

                average_name = f"{subject_dir.name}__{output_folder_name(base)}__{side.lower()}_gi_avg.png"
                average_path = per_record_avg_dir / average_name
                if record_windows:
                    average = average_windows(
                        record_windows,
                        subject_dir.name,
                        base,
                        side,
                        f"mean of {len(record_windows)} trigger-aligned GI trial(s); {sync_status}",
                    )
                    if overwrite or not average_path.exists() or average_path.stat().st_size == 0:
                        render_gi(average_path, average, average_trials=len(record_windows), normalized=True)
                    record_rows.append({"subject": subject_dir.name, "source_data_file": base, "side": side.lower(), "trials_averaged": len(record_windows), "status": "created", "png": average_name})
                elif overwrite and average_path.is_file():
                    average_path.unlink()

        subject_name = f"{subject_dir.name}_subject_gi_average.png"
        subject_path = per_subject_avg_dir / subject_name
        if subject_windows:
            subject_average = average_windows(
                subject_windows,
                subject_dir.name,
                "all recordings",
                "both",
                f"mean of {len(subject_windows)} GI trials from {len(subject_records)} recording(s), both limbs combined",
            )
            if overwrite or not subject_path.exists() or subject_path.stat().st_size == 0:
                render_gi(
                    subject_path,
                    subject_average,
                    average_trials=len(subject_windows),
                    normalized=True,
                    custom_title=f"Subject Mean Gait Initiation (GI) — {subject_dir.name}  |  all recordings, both limbs  |  n = {len(subject_windows)} trials",
                )
            subject_rows.append({"subject": subject_dir.name, "trials_averaged": len(subject_windows), "source_recordings": len(subject_records), "status": "created", "png": subject_name})
        else:
            if overwrite and subject_path.is_file():
                subject_path.unlink()
            subject_rows.append({"subject": subject_dir.name, "trials_averaged": 0, "source_recordings": 0, "status": "not created: no usable GI trials", "png": ""})

    write_csv(per_trial_dir / "gi_plot_manifest.csv", trial_rows)
    write_csv(per_trial_dir / "synchronization_report.csv", sync_rows)
    write_csv(output_dir / "avg" / "average_manifest.csv", record_rows)
    write_csv(per_subject_avg_dir / "subject_average_manifest.csv", subject_rows)
    summary = {
        "gi_per_trial_graphs": sum(1 for row in trial_rows if row.get("png")),
        "gi_per_record_average_graphs": len(record_rows),
        "gi_per_subject_average_graphs": sum(1 for row in subject_rows if row["status"] == "created"),
        "method": "Each figure shows a 0.5 second QS reference followed by the fixed 3.0-4.0 second GI interval after each matched trial-start trigger. The dashed boundary marks GI start.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", action="append", help="Process only one subject. May be supplied more than once.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing GI PNGs.")
    args = parser.parse_args()
    print(json.dumps(generate(GI_OUTPUT, set(args.subject) if args.subject else None, args.overwrite), indent=2))
