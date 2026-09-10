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
    reference_aligned: bool = False,
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
                suffix = "reference_aligned" if reference_aligned else "complete"
                file_name = f"trial_{local_trial:02d}_{side.lower()}_{suffix}.png"
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
                full_trial_channels = channels[fmg_start:fmg_end]
                movement_onset = detect_movement_onset(full_trial_channels, fmg.median_dt)
                if reference_aligned:
                    reference_gi_start = int(round(3.0 / fmg.median_dt))
                    shift_samples = max(0, movement_onset - reference_gi_start)
                    display_fmg_start = fmg_start + shift_samples
                    display_fmg_end = min(fmg_end, display_fmg_start + REFERENCE_SAMPLES)
                    shift_seconds = shift_samples * fmg.median_dt
                    display_insole_start = insole_start + int(round(shift_seconds / insole.median_dt))
                    display_insole_end = min(insole_end, display_insole_start + int(round(REFERENCE_SAMPLES * fmg.median_dt / insole.median_dt)))
                    starts = phase_markers(fmg.median_dt, REFERENCE_SAMPLES + 1)
                else:
                    shift_samples = 0
                    display_fmg_start = fmg_start
                    display_fmg_end = min(fmg_end, fmg_start + int(round(DISPLAY_SECONDS / fmg.median_dt)))
                    display_insole_start = insole_start
                    starts = phase_markers(fmg.median_dt, REFERENCE_SAMPLES + 1, movement_onset)
                    display_insole_end = min(insole_end, insole_start + int(round((display_fmg_end - display_fmg_start) * fmg.median_dt / insole.median_dt)))
                display_seconds = (display_fmg_end - display_fmg_start) * fmg.median_dt
                displayed_channels = channels[display_fmg_start:display_fmg_end]
                if overwrite or not destination.is_file() or destination.stat().st_size == 0:
                    render_reference_style(
                        destination=destination,
                        fmg=displayed_channels,
                        fmg_dt=fmg.median_dt,
                        cop=insole.values[display_insole_start:display_insole_end, 2],
                        vgrf=insole.values[display_insole_start:display_insole_end, 3],
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
                    "detected_gi_start_sample": movement_onset,
                    "detected_gi_start_seconds": round(movement_onset * fmg.median_dt, 3),
                    "reference_alignment_shift_samples": shift_samples,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", help="Process only sir_1 or sir_21.")
    parser.add_argument("--trial", type=int, help="Process only a one-based trial number within the record.")
    parser.add_argument("--side", choices=("L", "R"), help="Process only one limb.")
    parser.add_argument("--reference-aligned", action="store_true", help="Shift the display window so the measured GI onset is at reference sample 300.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing complete-trial PNGs.")
    args = parser.parse_args()
    print(generate(args.record, args.trial, args.overwrite, args.side, args.reference_aligned))
