"""Generate trigger-aligned quiet-standing plots from the Study 01 FMG dataset.

The script keeps raw files unchanged. It skips malformed text rows, aligns each
FMG/insoles recording using its trigger-event sequence, and writes a separate
left- and right-limb PNG for every recoverable quiet-standing segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "output" / "qsoutput"
SUBJECT_RE = re.compile(r"^Sub\d+_[AH]$")
SIDE_RE = re.compile(r"_?([LR])$", re.IGNORECASE)
TRIAL_RE = re.compile(r"_(\d+)(?:_?[LR])?$")


@dataclass
class Stream:
    path: Path
    kind: str
    side: str | None
    values: np.ndarray
    invalid_rows: int
    rising_edges: list[int]
    median_dt: float


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def load_stream(path: Path, expected_columns: int, kind: str, side: str | None = None) -> Stream:
    good_rows: list[list[float]] = []
    invalid_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            tokens = [token.strip() for token in raw_line.strip("\r\n").split("\t") if token.strip()]
            try:
                row = [float(token) for token in tokens]
            except ValueError:
                invalid_rows += 1
                continue
            if len(row) != expected_columns or not all(math.isfinite(value) for value in row):
                invalid_rows += 1
                continue
            good_rows.append(row)

    values = np.asarray(good_rows, dtype=np.float64)
    if len(values) < 2:
        return Stream(path, kind, side, values, invalid_rows, [], 0.01)
    trigger = values[:, 1]
    edges = (np.flatnonzero((trigger[1:] == 1.0) & (trigger[:-1] == 0.0)) + 1).tolist()
    positive_dt = np.diff(values[:, 0])
    positive_dt = positive_dt[positive_dt > 0]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.01
    return Stream(path, kind, side, values, invalid_rows, edges, median_dt)


def split_source_files(subject_dir: Path) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    fmg: dict[str, Path] = {}
    insoles: dict[tuple[str, str], Path] = {}
    for path in sorted((item for item in subject_dir.iterdir() if item.is_file()), key=natural_key):
        side_match = SIDE_RE.search(path.name)
        if side_match:
            side = side_match.group(1).upper()
            base = path.name[: side_match.start()].rstrip("_")
            insoles[(base, side)] = path
        else:
            fmg[path.name] = path
    return fmg, insoles


def trigger_sync(fmg: Stream, insole: Stream | None) -> tuple[str, float | None]:
    if insole is None:
        return "missing insole", None
    if not fmg.rising_edges or not insole.rising_edges:
        return "missing trigger event", None
    if len(fmg.rising_edges) != len(insole.rising_edges):
        return f"trigger count mismatch {len(fmg.rising_edges)} vs {len(insole.rising_edges)}", None
    if len(fmg.rising_edges) == 1:
        return "single trigger matched", 0.0

    fmg_times = fmg.values[fmg.rising_edges, 0]
    insole_times = insole.values[insole.rising_edges, 0]
    mismatch = np.abs((fmg_times - fmg_times[0]) - (insole_times - insole_times[0]))
    error = float(np.percentile(mismatch, 95))
    if error <= 0.25:
        return "trigger sequence synchronized", error
    return "trigger timing mismatch", error


def moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def detect_qs_end(signal: np.ndarray, start: int, stop: int, sample_dt: float) -> tuple[int, str]:
    """Find the first sustained departure from the post-trigger standing baseline."""
    stop = min(stop, len(signal))
    length = stop - start
    if length < 80:
        return min(stop, start + max(20, length // 2)), "short recording fallback"

    sample_rate = max(1.0, 1.0 / sample_dt)
    baseline_len = min(length // 3, max(50, int(round(0.75 * sample_rate))))
    baseline_len = max(30, baseline_len)
    baseline = signal[start : start + baseline_len]
    center = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - center), axis=0)
    scale = np.maximum(1.0, 1.4826 * mad)
    # A movement can first appear in only a few FMG channels.  The upper
    # quartile detects that departure earlier than the previous median score.
    score = np.percentile(np.abs((signal[start:stop] - center) / scale), 75, axis=1)
    smooth_width = max(5, int(round(0.15 * sample_rate)))
    smooth_score = moving_mean(score, smooth_width)
    baseline_score = smooth_score[:baseline_len]
    baseline_center = float(np.median(baseline_score))
    baseline_mad = float(np.median(np.abs(baseline_score - baseline_center)))
    threshold = max(3.5, float(np.percentile(baseline_score, 99)) * 1.5, baseline_center + 8.0 * max(0.1, baseline_mad))
    sustained = max(10, int(round(0.20 * sample_rate)))
    minimum_onset = max(int(round(0.50 * sample_rate)), baseline_len)

    above = smooth_score > threshold
    run = 0
    for offset in range(minimum_onset, len(above)):
        run = run + 1 if above[offset] else 0
        if run >= sustained:
            detected_onset = start + offset - sustained + 1
            buffer = max(20, int(round(0.50 * sample_rate)))
            qs_end = max(start + baseline_len, detected_onset - buffer)
            qs_end = min(qs_end, start + int(round(3.0 * sample_rate)))
            return qs_end, "FMG movement onset detected with 0.5 s pre-onset buffer"

    # The study protocol samples at 100 Hz and shows QS as the initial three
    # seconds of each trial.  This cap prevents later gait samples from being
    # labelled as QS when a trigger end is absent or malformed.
    fallback = min(stop, start + max(int(round(3.0 * sample_rate)), baseline_len))
    return fallback, "stable QS cap (3.0 s)"


def map_qs_to_insole(
    fmg: Stream,
    insole: Stream,
    fmg_start: int,
    fmg_end: int,
    fmg_trial_end: int | None,
    insole_start: int,
    insole_trial_end: int | None,
) -> tuple[int, int, str]:
    """Map the FMG QS duration to the insole timeline after trigger alignment."""
    if fmg_trial_end is not None and insole_trial_end is not None and fmg_trial_end > fmg_start:
        fraction = (fmg_end - fmg_start) / (fmg_trial_end - fmg_start)
        end = insole_start + int(round(fraction * (insole_trial_end - insole_start)))
        method = "start and end trigger phase mapping"
    else:
        scale = fmg.median_dt / insole.median_dt if insole.median_dt > 0 else 1.0
        end = insole_start + int(round((fmg_end - fmg_start) * scale))
        method = "start trigger and nominal sample rate mapping"
    end = max(insole_start + 2, min(end, len(insole.values)))
    return insole_start, end, method


def font(size: int, bold: bool = False):
    names = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def line_points(x: np.ndarray, y: np.ndarray, rect: tuple[int, int, int, int], x_max: float, y_min: float, y_max: float) -> list[tuple[float, float]]:
    left, top, right, bottom = rect
    if len(x) > (right - left) * 2:
        select = np.linspace(0, len(x) - 1, (right - left) * 2).astype(int)
        x, y = x[select], y[select]
    x_scale = (right - left) / max(x_max, 1e-9)
    y_scale = (bottom - top) / max(y_max - y_min, 1e-9)
    return [(left + float(value) * x_scale, bottom - (float(level) - y_min) * y_scale) for value, level in zip(x, y)]


def padded_range(values: np.ndarray, floor_zero: bool = False) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        high = low + 1.0
    padding = max((high - low) * 0.08, 1.0)
    low = 0.0 if floor_zero else low - padding
    return low, high + padding


def draw_axis(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], x_max: float, y_min: float, y_max: float, y_label: str, x_label: str | None, label_font, small_font, right_axis: bool = False):
    left, top, right, bottom = rect
    axis_color = (55, 55, 55)
    grid_color = (225, 225, 225)
    draw.rectangle(rect, outline=axis_color, width=2)
    for i in range(5):
        y = top + (bottom - top) * i / 4
        value = y_max - (y_max - y_min) * i / 4
        draw.line((left, y, right, y), fill=grid_color, width=1)
        text = f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}"
        box = draw.textbbox((0, 0), text, font=small_font)
        if right_axis:
            draw.text((right + 10, y - (box[3] - box[1]) / 2), text, fill=axis_color, font=small_font)
        else:
            draw.text((left - 12 - (box[2] - box[0]), y - (box[3] - box[1]) / 2), text, fill=axis_color, font=small_font)
    for i in range(5):
        x = left + (right - left) * i / 4
        value = x_max * i / 4
        draw.line((x, bottom, x, bottom + 6), fill=axis_color, width=2)
        if x_label is not None:
            text = f"{value:.1f}"
            box = draw.textbbox((0, 0), text, font=small_font)
            draw.text((x - (box[2] - box[0]) / 2, bottom + 10), text, fill=axis_color, font=small_font)
    if y_label:
        label_box = draw.textbbox((0, 0), y_label, font=label_font)
        label_img = Image.new("RGBA", (label_box[2] - label_box[0] + 6, label_box[3] - label_box[1] + 6), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((3, 3), y_label, fill=axis_color, font=label_font)
        label_img = label_img.rotate(90, expand=True)
        x = right + 44 if right_axis else left - 60
        draw.bitmap((x, (top + bottom) / 2 - label_img.height / 2), label_img)
    if x_label is not None:
        box = draw.textbbox((0, 0), x_label, font=label_font)
        draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 42), x_label, fill=axis_color, font=label_font)


def display_lane(
    values: np.ndarray,
    anchor: float,
    gain: float,
    smoothing_width: int,
    baseline_samples: int | None = None,
) -> np.ndarray:
    """Use a fixed, non-normalising display scale around a QS baseline."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.full(len(values), anchor)
    smoothed = moving_mean(values, smoothing_width)
    default_count = max(20, smoothing_width * 3)
    baseline_count = min(len(smoothed), baseline_samples if baseline_samples is not None else default_count)
    baseline = float(np.median(smoothed[:baseline_count]))
    return anchor + (smoothed - baseline) * gain


def render_plot(
    destination: Path,
    subject: str,
    record: str,
    trial_number: int,
    side: str,
    fmg_values: np.ndarray,
    fmg_dt: float,
    cop: np.ndarray,
    vgrf: np.ndarray,
    insole_dt: float,
    sync_status: str,
    onset_status: str,
    average_trials: int | None = None,
    normalized_qs_phase: bool = False,
    custom_title: str | None = None,
    phase_title: str = "Quiet Standing (QS)",
    phase_short: str = "QS",
    phase_window_label: str = "QS-only window",
    phase_note: str = "QS data only. A 0.25 s moving average removes sensor quantisation noise; fixed lane offsets preserve each trace's recorded variation.",
    normalized_phase_label: str = "Normalised QS phase (%)",
    baseline_samples: int | None = None,
    phase_boundary_sample: int | None = None,
    pre_phase_label: str | None = None,
    phase_markers: list[tuple[int, str]] | None = None,
    fmg_gain: float = 0.65,
    channel_offsets_override: list[float] | None = None,
) -> None:
    """Draw one vertically offset ten-signal view for the selected gait phase."""
    width, height = 1920, 960
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(30, bold=True)
    label_font = font(19)
    small_font = font(16)
    legend_font = font(18)

    left, right = 150, 1560
    chart = (left, 175, right, 795)
    if normalized_qs_phase:
        fmg_x = np.linspace(0.0, 100.0, len(fmg_values))
        insole_x = np.linspace(0.0, 100.0, len(cop))
        x_max = 100.0
        x_label = normalized_phase_label
    else:
        fmg_x = np.arange(len(fmg_values), dtype=float)
        insole_x = np.linspace(0, max(len(fmg_values) - 1, 1), len(cop), dtype=float)
        x_max = max(float(len(fmg_values) - 1), 1.0)
        x_label = "Samples @ 100 Hz"

    if custom_title is not None:
        title = custom_title
    elif average_trials is None:
        title = f"{phase_title} — {subject}  |  Record {record}  |  Trial {trial_number:02d}  |  {side} limb"
    else:
        title = f"Average {phase_title} — {subject}  |  Data {record}  |  {side} limb  |  n = {average_trials} trials"
    draw.text((left, 28), title, fill=(0, 0, 0), font=title_font)
    detail = f"{phase_window_label} • recorded values displayed with 0.25 s smoothing • alignment: {sync_status}"
    draw.text((left, 76), detail, fill=(70, 70, 70), font=small_font)

    draw_axis(draw, chart, x_max, 0.0, 500.0, "Digitally offset amplitude", x_label, label_font, small_font)
    channel_colors = [
        (78, 121, 167), (242, 142, 43), (225, 87, 89), (118, 183, 178),
        (89, 161, 79), (237, 201, 72), (176, 122, 161), (255, 157, 167),
    ]
    channel_offsets = channel_offsets_override or [195, 170, 145, 120, 95, 70, 45, 20]
    fmg_smoothing = max(5, int(round(0.25 / max(fmg_dt, 0.001))))
    for index in range(8):
        displayed = display_lane(fmg_values[:, index], channel_offsets[index], fmg_gain, fmg_smoothing, baseline_samples)
        points = line_points(fmg_x, displayed, chart, x_max, 0.0, 500.0)
        draw.line(points, fill=channel_colors[index], width=2)

    # CoP and vGRF are intentionally positioned above all eight FMG lanes.
    insole_smoothing = max(5, int(round(0.25 / max(insole_dt, 0.001))))
    cop_points = line_points(insole_x, display_lane(cop, 285.0, 2.50, insole_smoothing, baseline_samples), chart, x_max, 0.0, 500.0)
    vgrf_points = line_points(insole_x, display_lane(vgrf, 390.0, 0.08, insole_smoothing, baseline_samples), chart, x_max, 0.0, 500.0)
    draw.line(cop_points, fill=(102, 102, 102), width=3)
    draw.line(vgrf_points, fill=(214, 148, 0), width=3)

    phase_bar_y = 138
    draw.line((left, phase_bar_y, right, phase_bar_y), fill=(40, 40, 40), width=4)
    if phase_markers:
        markers = sorted(
            ((max(0, min(int(sample), len(fmg_values) - 1)), label) for sample, label in phase_markers),
            key=lambda item: item[0],
        )
        for index, (sample, label) in enumerate(markers):
            next_sample = markers[index + 1][0] if index + 1 < len(markers) else len(fmg_values) - 1
            start_x = left + sample / max(len(fmg_values) - 1, 1) * (right - left)
            end_x = left + next_sample / max(len(fmg_values) - 1, 1) * (right - left)
            label_box = draw.textbbox((0, 0), label, font=label_font)
            label_x = max(left + 8, min((start_x + end_x - (label_box[2] - label_box[0])) / 2, right - (label_box[2] - label_box[0]) - 8))
            draw.rectangle((label_x - 8, phase_bar_y - 30, label_x + label_box[2] - label_box[0] + 8, phase_bar_y - 4), fill="white")
            draw.text((label_x, phase_bar_y - 31), label, fill=(0, 0, 0), font=label_font)
            if index:
                for y in range(chart[1], chart[3], 18):
                    draw.line((start_x, y, start_x, min(y + 10, chart[3])), fill=(35, 35, 35), width=3)
    else:
        first_phase_label = pre_phase_label if phase_boundary_sample is not None and pre_phase_label else phase_short
        phase_box = draw.textbbox((0, 0), first_phase_label, font=label_font)
        draw.rectangle((left + 20, phase_bar_y - 30, left + 20 + phase_box[2] - phase_box[0] + 16, phase_bar_y - 4), fill="white")
        draw.text((left + 28, phase_bar_y - 31), first_phase_label, fill=(0, 0, 0), font=label_font)

    if phase_boundary_sample is not None and not phase_markers:
        boundary_fraction = phase_boundary_sample / max(len(fmg_values) - 1, 1)
        boundary_x = left + boundary_fraction * (right - left)
        for y in range(chart[1], chart[3], 18):
            draw.line((boundary_x, y, boundary_x, min(y + 10, chart[3])), fill=(35, 35, 35), width=3)
        label_box = draw.textbbox((0, 0), phase_short, font=label_font)
        draw.rectangle((boundary_x + 14, phase_bar_y - 30, boundary_x + 30 + label_box[2] - label_box[0], phase_bar_y - 4), fill="white")
        draw.text((boundary_x + 22, phase_bar_y - 31), phase_short, fill=(0, 0, 0), font=label_font)

    legend_x, legend_y = 1610, 190
    draw.text((legend_x, legend_y - 42), "Signals", fill=(0, 0, 0), font=legend_font)
    for index, color in enumerate(channel_colors):
        y = legend_y + index * 32
        draw.line((legend_x, y + 12, legend_x + 40, y + 12), fill=color, width=3)
        draw.text((legend_x + 50, y), f"Channel {index + 1}", fill=(30, 30, 30), font=legend_font)
    legend_y = 480
    draw.line((legend_x, legend_y + 46, legend_x + 40, legend_y + 46), fill=(102, 102, 102), width=4)
    draw.text((legend_x + 50, legend_y + 31), "CoP", fill=(30, 30, 30), font=legend_font)
    draw.line((legend_x, legend_y + 82, legend_x + 40, legend_y + 82), fill=(214, 148, 0), width=4)
    draw.text((legend_x + 50, legend_y + 67), "vGRF", fill=(30, 30, 30), font=legend_font)

    draw.text((left, 870), phase_note, fill=(70, 70, 70), font=small_font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def unit_trial_starts(stream: Stream) -> list[tuple[int, int | None, int]]:
    """Return start edge, optional end edge, and one-based trial number inside a recording."""
    starts = []
    for local_trial, edge_index in enumerate(range(0, len(stream.rising_edges), 2), start=1):
        start = stream.rising_edges[edge_index]
        end = stream.rising_edges[edge_index + 1] if edge_index + 1 < len(stream.rising_edges) else None
        starts.append((start, end, local_trial))
    return starts


def record_label(name: str) -> str:
    match = TRIAL_RE.search(name)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9]+", "_", name)


def output_folder_name(name: str) -> str:
    """Create a safe folder name while retaining the source data file's identity."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "unnamed_recording"


def generate(output_dir: Path, overwrite: bool = False, subjects: set[str] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_trial_dir = output_dir / "per trial"
    per_trial_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []
    plot_count = 0

    subject_dirs = sorted((path for path in DATASET_ROOT.iterdir() if path.is_dir() and SUBJECT_RE.match(path.name)), key=natural_key)
    if subjects:
        subject_dirs = [path for path in subject_dirs if path.name in subjects]
    for subject_dir in subject_dirs:
        fmg_files, insole_files = split_source_files(subject_dir)
        for base, fmg_path in sorted(fmg_files.items(), key=lambda item: natural_key(item[1])):
            fmg = load_stream(fmg_path, 18, "FMG")
            trial_starts = unit_trial_starts(fmg)
            if not trial_starts:
                sync_rows.append({
                    "subject": subject_dir.name, "record": base, "side": "both", "status": "no FMG trigger start", "plots_created": 0,
                })
                continue

            loaded_insoles: dict[str, Stream | None] = {}
            for side in ("L", "R"):
                insole_path = insole_files.get((base, side))
                loaded_insoles[side] = load_stream(insole_path, 4, "insole", side) if insole_path else None

            for side in ("L", "R"):
                insole = loaded_insoles[side]
                status, p95_error = trigger_sync(fmg, insole)
                sync_rows.append({
                    "subject": subject_dir.name,
                    "record": base,
                    "side": side,
                    "fmg_trigger_rising_edges": len(fmg.rising_edges),
                    "insole_trigger_rising_edges": len(insole.rising_edges) if insole else 0,
                    "sync_status": status,
                    "p95_relative_trigger_error_seconds": "" if p95_error is None else round(p95_error, 4),
                    "fmg_invalid_rows_skipped": fmg.invalid_rows,
                    "insole_invalid_rows_skipped": insole.invalid_rows if insole else "",
                })

                if insole is None or not insole.rising_edges:
                    continue
                channel_start = 2 if side == "L" else 10
                fmg_channels = fmg.values[:, channel_start : channel_start + 8]
                for fmg_start, fmg_trial_end, local_trial in trial_starts:
                    source_edge_index = 2 * (local_trial - 1)
                    if source_edge_index >= len(insole.rising_edges):
                        manifest_rows.append({
                            "subject": subject_dir.name, "record": base, "local_trial": local_trial, "side": side,
                            "status": "skipped no corresponding insole trigger", "png": "",
                        })
                        continue
                    insole_start = insole.rising_edges[source_edge_index]
                    insole_trial_end = insole.rising_edges[source_edge_index + 1] if source_edge_index + 1 < len(insole.rising_edges) else None
                    scan_stop = fmg_trial_end if fmg_trial_end is not None else min(len(fmg.values), fmg_start + int(round(15 / fmg.median_dt)))
                    fmg_end, onset_status = detect_qs_end(fmg_channels, fmg_start, scan_stop, fmg.median_dt)
                    if fmg_end <= fmg_start + 5:
                        manifest_rows.append({
                            "subject": subject_dir.name, "record": base, "local_trial": local_trial, "side": side,
                            "status": "skipped QS window too short", "png": "",
                        })
                        continue
                    insole_qs_start, insole_qs_end, alignment_method = map_qs_to_insole(
                        fmg, insole, fmg_start, fmg_end, fmg_trial_end, insole_start, insole_trial_end
                    )
                    sync_label = status if status.startswith("trigger sequence") or status.startswith("single") else f"review {status}"
                    # Subject / source-data-file / QS figures keeps each recording's trials together.
                    file_name = f"trial_{local_trial:02d}_{side.lower()}.png"
                    destination = per_trial_dir / subject_dir.name / output_folder_name(fmg_path.name) / file_name
                    relative_png = destination.relative_to(per_trial_dir).as_posix()
                    if not overwrite and destination.exists() and destination.stat().st_size > 0:
                        plot_count += 1
                        manifest_rows.append({
                            "subject": subject_dir.name,
                            "record": base,
                            "local_trial": local_trial,
                            "side": side,
                            "status": "existing valid PNG preserved",
                            "sync_status": status,
                            "onset_status": onset_status,
                            "alignment_method": alignment_method,
                            "fmg_qs_samples": fmg_end - fmg_start,
                            "fmg_qs_seconds": round((fmg_end - fmg_start) * fmg.median_dt, 3),
                            "insole_qs_samples": insole_qs_end - insole_qs_start,
                            "insole_qs_seconds": round((insole_qs_end - insole_qs_start) * insole.median_dt, 3),
                            "png": relative_png,
                        })
                        continue
                    render_plot(
                        destination, subject_dir.name, record_label(base), local_trial, "left" if side == "L" else "right",
                        fmg_channels[fmg_start:fmg_end], fmg.median_dt,
                        insole.values[insole_qs_start:insole_qs_end, 2],
                        insole.values[insole_qs_start:insole_qs_end, 3], insole.median_dt,
                        sync_label, onset_status,
                    )
                    plot_count += 1
                    manifest_rows.append({
                        "subject": subject_dir.name,
                        "record": base,
                        "local_trial": local_trial,
                        "side": side,
                        "status": "created",
                        "sync_status": status,
                        "onset_status": onset_status,
                        "alignment_method": alignment_method,
                        "fmg_qs_samples": fmg_end - fmg_start,
                        "fmg_qs_seconds": round((fmg_end - fmg_start) * fmg.median_dt, 3),
                        "insole_qs_samples": insole_qs_end - insole_qs_start,
                        "insole_qs_seconds": round((insole_qs_end - insole_qs_start) * insole.median_dt, 3),
                        "png": relative_png,
                    })

    def write_csv(path: Path, rows: list[dict[str, object]]):
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(per_trial_dir / "qs_plot_manifest.csv", manifest_rows)
    write_csv(per_trial_dir / "synchronization_report.csv", sync_rows)
    rendered_entries = sum(1 for row in manifest_rows if row.get("status") == "created")
    preserved_entries = sum(1 for row in manifest_rows if row.get("status") == "existing valid PNG preserved")
    summary = {
        "plots_available": plot_count,
        "rendered_this_run": rendered_entries,
        "existing_pngs_preserved": preserved_entries,
        "unavailable_entries": len(manifest_rows) - rendered_entries - preserved_entries,
        "synchronized_pairs": sum(1 for row in sync_rows if str(row.get("sync_status", "")).startswith("trigger sequence") or str(row.get("sync_status", "")).startswith("single")),
        "pairs_needing_review": sum(1 for row in sync_rows if not (str(row.get("sync_status", "")).startswith("trigger sequence") or str(row.get("sync_status", "")).startswith("single"))),
        "method": "Each graph begins at a trial-start trigger and ends at the first sustained FMG departure from the standing baseline. Insole traces are aligned by matching trigger position and trial phase or nominal sampling rate.",
    }
    (per_trial_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing QS PNGs.")
    parser.add_argument("--subject", action="append", help="Process this subject folder only. May be supplied more than once.")
    args = parser.parse_args()
    summary = generate(args.output.resolve(), overwrite=args.overwrite, subjects=set(args.subject) if args.subject else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
