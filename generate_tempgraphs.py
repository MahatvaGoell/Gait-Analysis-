"""Create complete trigger-to-trigger graphs for Subject 08 without altering QS/GI outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_qs_plots import (
    DATASET_ROOT,
    ROOT,
    display_lane,
    font,
    load_stream,
    line_points,
    moving_mean,
    natural_key,
    output_folder_name,
    split_source_files,
    trigger_sync,
    unit_trial_starts,
)


SUBJECT = "Sub08_H"
OUTPUT = ROOT / "tempgraphs"

# These timings reproduce the study-protocol sequence in the supplied full
# reference figure. They are measured from the trigger-defined trial start.
PHASES = (
    (0.0, "QS"),
    (3.0, "GI"),
    (4.0, "Steady state short step (SSSW)"),
    (10.0, "SLT"),
    (11.0, "Steady state long step (SSLW)"),
    (17.0, "GT"),
)
DISPLAY_SECONDS = 22.0
REFERENCE_SAMPLES = 2200


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, top: int, bottom: int) -> None:
    for y in range(top, bottom, 18):
        draw.line((x, y, x, min(y + 10, bottom)), fill=(20, 20, 20), width=3)


def render_reference_style(
    destination: Path,
    fmg: np.ndarray,
    fmg_dt: float,
    cop: np.ndarray,
    vgrf: np.ndarray,
    insole_dt: float,
    phase_starts: list[tuple[int, str]],
) -> None:
    """Draw the compact all-phase layout used in the supplied reference image."""
    width, height = 2048, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_font = font(18)
    phase_font = font(19, bold=True)
    legend_font = font(16)
    small_font = font(15)
    left, right, top, bottom = 90, 1770, 80, 500
    sample_count = len(fmg)
    x_max = REFERENCE_SAMPLES

    # Reference-style axes: fixed digital display scale and sample ticks.
    axis = (45, 45, 45)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    for value in range(0, 501, 100):
        y = bottom - value / 500 * (bottom - top)
        draw.line((left - 12, y, left, y), fill=axis, width=2)
        box = draw.textbbox((0, 0), str(value), font=small_font)
        draw.text((left - 22 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), str(value), fill=axis, font=small_font)
        if value not in (0, 500):
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
    tick_step = 200
    for value in range(0, x_max + 1, tick_step):
        x = left + value / x_max * (right - left)
        draw.line((x, bottom, x, bottom + 12), fill=axis, width=2)
        text = str(value)
        box = draw.textbbox((0, 0), text, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 16), text, fill=axis, font=small_font)

    y_label = "FMG amplitude (digital value)"
    box = draw.textbbox((0, 0), y_label, font=label_font)
    label_image = Image.new("RGBA", (box[2] - box[0] + 6, box[3] - box[1] + 6), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((3, 3), y_label, fill=axis, font=label_font)
    label_image = label_image.rotate(90, expand=True)
    image.paste(label_image, (20, int((top + bottom - label_image.height) / 2)), label_image)
    x_label = "Samples @ 100Hz"
    box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 54), x_label, fill=axis, font=label_font)

    colors = [
        (78, 121, 167), (242, 142, 43), (225, 87, 89), (118, 183, 178),
        (89, 161, 79), (237, 201, 72), (176, 122, 161), (255, 157, 167),
    ]
    # Each channel is referenced to its own quiet-standing baseline.  The
    # display scale is bounded per channel so actual gait excursions remain
    # visible while a large excursion in one channel cannot flatten the rest.
    offsets = [220, 190, 160, 130, 105, 85, 70, 80]
    fmg_smoothing = max(5, int(round(0.25 / max(fmg_dt, 0.001))))
    qs_samples = min(sample_count, max(30, int(round(3.0 / fmg_dt))))
    fmg_x = np.arange(sample_count, dtype=float)
    chart = (left, top, right, bottom)
    for channel in range(8):
        smoothed = moving_mean(fmg[:, channel], fmg_smoothing)
        standing_baseline = float(np.median(smoothed[:qs_samples]))
        departure = smoothed - standing_baseline
        robust_excursion = max(
            1.0,
            abs(float(np.percentile(departure, 1))),
            abs(float(np.percentile(departure, 99))),
        )
        display_scale = min(1.2, 70.0 / robust_excursion)
        displayed = offsets[channel] + departure * display_scale
        displayed = np.clip(displayed, 0.0, 500.0)
        draw.line(line_points(fmg_x, displayed, chart, x_max, 0.0, 500.0), fill=colors[channel], width=2)

    insole_x = np.linspace(0.0, x_max, len(cop))
    insole_smoothing = max(5, int(round(0.25 / max(insole_dt, 0.001))))
    draw.line(line_points(insole_x, np.clip(display_lane(cop, 285.0, 2.50, insole_smoothing), 0, 500), chart, x_max, 0.0, 500.0), fill=(102, 102, 102), width=3)
    draw.line(line_points(insole_x, np.clip(display_lane(vgrf, 390.0, 0.08, insole_smoothing), 0, 500), chart, x_max, 0.0, 500.0), fill=(214, 148, 0), width=3)

    # Full protocol bar and phase boundaries match the reference notation.
    bar_y = 38
    draw.line((left, bar_y, right, bar_y), fill=(35, 35, 35), width=3)
    markers = phase_starts
    for index, (sample, label) in enumerate(markers):
        end_sample = markers[index + 1][0] if index + 1 < len(markers) else REFERENCE_SAMPLES
        start_x = left + sample / x_max * (right - left)
        end_x = left + end_sample / x_max * (right - left)
        label_box = draw.textbbox((0, 0), label, font=phase_font)
        label_x = max(left + 4, min((start_x + end_x - (label_box[2] - label_box[0])) / 2, right - (label_box[2] - label_box[0]) - 4))
        draw.rectangle((label_x - 8, 10, label_x + label_box[2] - label_box[0] + 8, 35), fill="white")
        draw.text((label_x, 8), label, fill=(0, 0, 0), font=phase_font)
        if index:
            draw_dashed_vertical(draw, start_x, top, bottom)

    legend_x, legend_y = 1810, 84
    for channel, color in enumerate(colors):
        y = legend_y + channel * 25
        draw.line((legend_x, y + 9, legend_x + 32, y + 9), fill=color, width=3)
        draw.text((legend_x + 40, y), f"Channel {channel + 1}", fill=(30, 30, 30), font=legend_font)
    for label, color, y in (("CoP", (102, 102, 102), legend_y + 208), ("vGRF", (214, 148, 0), legend_y + 233)):
        draw.line((legend_x, y + 9, legend_x + 32, y + 9), fill=color, width=3)
        draw.text((legend_x + 40, y), label, fill=(30, 30, 30), font=legend_font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def phase_markers(fmg_dt: float, samples: int, gi_start_sample: int | None = None) -> list[tuple[int, str]]:
    """Build phase markers, optionally aligning GI to the measured movement onset."""
    if gi_start_sample is None:
        return [
            (int(round(seconds / fmg_dt)), label)
            for seconds, label in PHASES
            if int(round(seconds / fmg_dt)) < samples
        ]
    sample_rate = 1.0 / fmg_dt
    labels = ("QS", "GI", "Steady state short step (SSSW)", "SLT", "Steady state long step (SSLW)", "GT")
    starts = (
        0,
        gi_start_sample,
        gi_start_sample + int(round(1.0 * sample_rate)),
        gi_start_sample + int(round(7.0 * sample_rate)),
        gi_start_sample + int(round(8.0 * sample_rate)),
        gi_start_sample + int(round(14.0 * sample_rate)),
    )
    return [(sample, label) for sample, label in zip(starts, labels) if sample < samples]


def detect_movement_onset(channels: np.ndarray, sample_dt: float) -> int:
    """Find the first sustained FMG departure from the initial 3-second QS baseline."""
    sample_rate = max(1.0, 1.0 / sample_dt)
    baseline_samples = min(len(channels) // 2, max(100, int(round(3.0 * sample_rate))))
    smooth_width = max(5, int(round(0.10 * sample_rate)))
    smoothed = np.column_stack([moving_mean(channels[:, channel], smooth_width) for channel in range(8)])
    baseline = smoothed[:baseline_samples]
    center = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - center), axis=0)
    scale = np.maximum(1.0, 1.4826 * mad)
    score = np.percentile(np.abs((smoothed - center) / scale), 75, axis=1)
    score = moving_mean(score, max(5, int(round(0.15 * sample_rate))))
    baseline_score = score[:baseline_samples]
    threshold = max(5.0, float(np.percentile(baseline_score, 99)) * 1.8)
    sustained = max(15, int(round(0.20 * sample_rate)))
    run = 0
    for index in range(max(50, baseline_samples // 2), len(score)):
        run = run + 1 if score[index] > threshold else 0
        if run >= sustained:
            return index - sustained + 1
    return int(round(3.0 * sample_rate))


def resample_to_length(values: np.ndarray, length: int) -> np.ndarray:
    """Map the matched insole recording to the FMG trial timeline without shifting it."""
    values = np.asarray(values, dtype=float)
    if len(values) == length:
        return values
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.column_stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])])


def normalised_motion_score(values: np.ndarray, sample_dt: float, percentile: float) -> np.ndarray:
    """Return a baseline-normalised activity score for one or more recorded signals."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    sample_rate = max(1.0, 1.0 / sample_dt)
    baseline_samples = min(len(values) // 2, max(100, int(round(3.0 * sample_rate))))
    smoothing = max(5, int(round(0.10 * sample_rate)))
    smoothed = np.column_stack([moving_mean(values[:, column], smoothing) for column in range(values.shape[1])])
    baseline = smoothed[:baseline_samples]
    center = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - center), axis=0)
    scale = np.maximum(1.0, 1.4826 * mad)
    score = np.percentile(np.abs((smoothed - center) / scale), percentile, axis=1)
    baseline_score = score[:baseline_samples]
    threshold = max(4.0, float(np.percentile(baseline_score, 99)) * 1.8)
    return moving_mean(score / threshold, max(5, int(round(0.15 * sample_rate))))


def local_peaks(values: np.ndarray, start: int, min_distance: int) -> list[int]:
    """Find separated, above-baseline activity peaks in one trial."""
    if start >= len(values) - 2:
        return []
    threshold = max(1.05, float(np.percentile(values[start:], 60)))
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] >= threshold)
    ) + 1
    candidates = [int(index) for index in candidates if index >= start]
    peaks: list[int] = []
    for candidate in candidates:
        if not peaks or candidate - peaks[-1] >= min_distance:
            peaks.append(candidate)
        elif values[candidate] > values[peaks[-1]]:
            peaks[-1] = candidate
    return peaks


def data_driven_phase_markers(
    channels: np.ndarray,
    cop: np.ndarray,
    vgrf: np.ndarray,
    sample_dt: float,
) -> tuple[list[tuple[int, str]], dict[str, object]]:
    """Estimate all trial phase starts from the individual FMG and insole pattern.

    The raw sample timeline is preserved.  Gait peaks and their recorded
    timing determine every boundary; no fixed phase duration is imposed.
    """
    insole = resample_to_length(np.column_stack((cop, vgrf)), len(channels))
    fmg_score = normalised_motion_score(channels, sample_dt, 75)
    insole_score = normalised_motion_score(insole, sample_dt, 100)
    joint_score = np.maximum(fmg_score, insole_score)
    sample_rate = max(1.0, 1.0 / sample_dt)
    onset = detect_movement_onset(channels, sample_dt)
    # Confirm a movement departure in the paired insole stream where possible.
    sustained = max(15, int(round(0.20 * sample_rate)))
    run = 0
    for index in range(max(50, onset - int(round(0.75 * sample_rate))), len(joint_score)):
        active = joint_score[index] >= 1.0 and (fmg_score[index] >= 0.70 or insole_score[index] >= 0.70)
        run = run + 1 if active else 0
        if run >= sustained:
            onset = index - sustained + 1
            break

    peaks = local_peaks(joint_score, onset, max(20, int(round(0.35 * sample_rate))))
    if len(peaks) < 6:
        markers = [(0, "QS"), (onset, "GI")]
        return markers, {
            "gi_start": onset,
            "sssw_start": "",
            "slt_start": "",
            "sslw_start": "",
            "gt_start": "",
            "phase_detection": "GI detected; insufficient gait peaks for later phase boundaries",
        }

    # GI ends when repeated, sustained gait peaks are present, rather than at a
    # universal one-second mark.
    sssw_start = peaks[min(2, len(peaks) - 1)]
    amplitudes = np.asarray([joint_score[index] for index in peaks])
    intervals = np.diff(peaks).astype(float)
    median_interval = max(1.0, float(np.median(intervals)))
    change_candidates: list[tuple[float, int]] = []
    for index in range(3, len(peaks) - 3):
        previous_period = float(np.median(intervals[index - 3:index]))
        following_period = float(np.median(intervals[index:index + 3]))
        previous_amplitude = float(np.median(amplitudes[index - 3:index]))
        following_amplitude = float(np.median(amplitudes[index:index + 3]))
        period_change = abs(following_period - previous_period) / median_interval
        amplitude_change = abs(following_amplitude - previous_amplitude) / max(0.25, previous_amplitude)
        change_candidates.append((period_change + amplitude_change, peaks[index]))

    # Select ordered, well-separated changes from the actual gait pattern.
    minimum_gap = max(50, int(round(0.75 * sample_rate)))
    selected: list[int] = []
    for _, position in sorted(change_candidates, reverse=True):
        if position <= sssw_start + minimum_gap or position >= peaks[-2] - minimum_gap:
            continue
        if all(abs(position - chosen) >= minimum_gap for chosen in selected):
            selected.append(position)
        if len(selected) == 2:
            break
    selected.sort()

    if len(selected) == 2:
        slt_start, sslw_start = selected
    elif len(selected) == 1:
        slt_start = selected[0]
        later = [peak for peak in peaks if peak > slt_start + minimum_gap]
        sslw_start = later[0] if later else peaks[-2]
    else:
        # No claim is made about an unobserved transition; the labels stop at
        # SSSW instead of fabricating phase durations.
        markers = [(0, "QS"), (onset, "GI"), (sssw_start, "Steady state short step (SSSW)")]
        return markers, {
            "gi_start": onset,
            "sssw_start": sssw_start,
            "slt_start": "",
            "sslw_start": "",
            "gt_start": "",
            "phase_detection": "GI and SSSW detected; later transitions not separable in recorded gait pattern",
        }

    # Termination starts at the last substantial gait-pattern change after the
    # long-step section; otherwise retain the last three observed gait peaks.
    later_changes = [position for _, position in change_candidates if position > sslw_start + minimum_gap]
    gt_start = min(later_changes) if later_changes else peaks[max(0, len(peaks) - 3)]
    gt_start = max(gt_start, sslw_start + 1)
    return [
        (0, "QS"),
        (onset, "GI"),
        (sssw_start, "Steady state short step (SSSW)"),
        (slt_start, "SLT"),
        (sslw_start, "Steady state long step (SSLW)"),
        (gt_start, "GT"),
    ], {
        "gi_start": onset,
        "sssw_start": sssw_start,
        "slt_start": slt_start,
        "sslw_start": sslw_start,
        "gt_start": gt_start,
        "phase_detection": "all phase boundaries estimated from paired FMG and insole gait-pattern changes",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(
    record_filter: str | None = None,
    trial_filter: int | None = None,
    overwrite: bool = False,
    side_filter: str | None = None,
) -> dict[str, int]:
    subject_dir = DATASET_ROOT / SUBJECT
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

    manifest: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []
    fmg_files, insole_files = split_source_files(subject_dir)
    for record, fmg_path in sorted(fmg_files.items(), key=lambda item: natural_key(item[1])):
        if record_filter and record != record_filter:
            continue
        fmg = load_stream(fmg_path, 18, "FMG")
        trials = unit_trial_starts(fmg)
        for side in ("L", "R"):
            if side_filter and side != side_filter:
                continue
            insole_path = insole_files.get((record, side))
            insole = load_stream(insole_path, 4, "insole", side) if insole_path else None
            sync_status, p95_error = trigger_sync(fmg, insole)
            sync_rows.append({
                "subject": SUBJECT,
                "record": record,
                "side": side,
                "sync_status": sync_status,
                "p95_relative_trigger_error_seconds": "" if p95_error is None else round(p95_error, 4),
                "fmg_trigger_rising_edges": len(fmg.rising_edges),
                "insole_trigger_rising_edges": len(insole.rising_edges) if insole else 0,
            })
            if insole is None:
                continue

            channels = fmg.values[:, 2:10] if side == "L" else fmg.values[:, 10:18]
            for fmg_start, fmg_end, local_trial in trials:
                if trial_filter is not None and local_trial != trial_filter:
                    continue
                file_name = f"trial_{local_trial:02d}_{side.lower()}_complete.png"
                destination = OUTPUT / SUBJECT / output_folder_name(record) / file_name
                edge = 2 * (local_trial - 1)
                if fmg_end is None or edge + 1 >= len(insole.rising_edges):
                    manifest.append({"subject": SUBJECT, "record": record, "side": side, "trial": local_trial, "status": "skipped: incomplete matched trigger pair", "png": ""})
                    continue
                insole_start = insole.rising_edges[edge]
                insole_end = insole.rising_edges[edge + 1]
                if fmg_end <= fmg_start + 5 or insole_end <= insole_start + 5:
                    manifest.append({"subject": SUBJECT, "record": record, "side": side, "trial": local_trial, "status": "skipped: empty trigger interval", "png": ""})
                    continue
                display_fmg_start = fmg_start
                display_fmg_end = min(fmg_end, fmg_start + int(round(DISPLAY_SECONDS / fmg.median_dt)))
                display_insole_start = insole_start
                display_insole_end = min(insole_end, insole_start + int(round((display_fmg_end - display_fmg_start) * fmg.median_dt / insole.median_dt)))
                display_seconds = (display_fmg_end - display_fmg_start) * fmg.median_dt
                displayed_channels = channels[display_fmg_start:display_fmg_end]
                displayed_cop = insole.values[display_insole_start:display_insole_end, 2]
                displayed_vgrf = insole.values[display_insole_start:display_insole_end, 3]
                starts, phase_info = data_driven_phase_markers(
                    displayed_channels,
                    displayed_cop,
                    displayed_vgrf,
                    fmg.median_dt,
                )
                if overwrite or not destination.is_file() or destination.stat().st_size == 0:
                    render_reference_style(
                        destination=destination,
                        fmg=displayed_channels,
                        fmg_dt=fmg.median_dt,
                        cop=displayed_cop,
                        vgrf=displayed_vgrf,
                        insole_dt=insole.median_dt,
                        phase_starts=starts,
                    )
                    status = "created"
                else:
                    status = "existing valid PNG preserved"
                manifest.append({
                    "subject": SUBJECT,
                    "record": record,
                    "side": side,
                    "trial": local_trial,
                    "status": status,
                    "sync_status": sync_status,
                    "fmg_samples": display_fmg_end - display_fmg_start,
                    "fmg_seconds": round(display_seconds, 3),
                    "insole_samples": display_insole_end - display_insole_start,
                    "insole_seconds": round((display_insole_end - display_insole_start) * insole.median_dt, 3),
                    "detected_gi_start_sample": phase_info["gi_start"],
                    "detected_gi_start_seconds": round(float(phase_info["gi_start"]) * fmg.median_dt, 3),
                    "detected_sssw_start_sample": phase_info["sssw_start"],
                    "detected_slt_start_sample": phase_info["slt_start"],
                    "detected_sslw_start_sample": phase_info["sslw_start"],
                    "detected_gt_start_sample": phase_info["gt_start"],
                    "phase_detection": phase_info["phase_detection"],
                    "png": destination.relative_to(OUTPUT).as_posix(),
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not record_filter and trial_filter is None:
        write_csv(OUTPUT / "complete_trial_manifest.csv", manifest)
        write_csv(OUTPUT / "synchronization_report.csv", sync_rows)
    return {
        "complete_graphs": sum(1 for row in manifest if row.get("png")),
        "skipped": sum(1 for row in manifest if not row.get("png")),
    }


if __name__ == "__main__":
    # Latest supported pipeline: full timestamps and explicit provisional labels.
    from subject08_segmentation import main
    main()
