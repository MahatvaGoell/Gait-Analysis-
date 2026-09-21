"""Render the existing SSSW-labelled samples in the requested phase-band layout."""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_qs_plots import font
from subject08_review import COLORS, runs


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
TEMPGRAPHS = ROOT / "tempgraphs"
OUTPUT = ROOT / "SSSW" / "per trial"
SIGNALS = tuple([f"FMG_channel_{number}" for number in range(1, 9)] + ["CoP", "vGRF"])
ANCHORS = (200, 170, 145, 120, 100, 80, 60, 40, 400, 380)


def trial_number(path: Path) -> int:
    match = re.search(r"trial_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot determine trial number from {path.name}")
    return int(match.group(1))


def segmented_source(subject: str) -> Path | None:
    completed_subject = TEMPGRAPHS / subject.lower() / "segmented_data"
    original_subject = TEMPGRAPHS / "segmented_data" / subject
    if completed_subject.is_dir():
        return completed_subject
    if original_subject.is_dir():
        return original_subject
    return None


def read_trial(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[name]) if row[name] else np.nan for name in SIGNALS] for row in rows], dtype=float
    )
    phases = np.asarray([row["provisional_phase"] for row in rows])
    return times, values, phases


def display_lanes(times: np.ndarray, values: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Reuse each full trial's baseline, gain and lane offsets for its SSSW view."""
    displayed = np.full_like(values, np.nan)
    gi_rows = np.flatnonzero(phases == "GI")
    gi_start = float(times[gi_rows[0]]) if len(gi_rows) else 0.75
    baseline_rows = times < max(0.25, min(gi_start, 0.75))
    for index in range(values.shape[1]):
        signal = values[:, index]
        valid = np.isfinite(signal)
        if not valid.any():
            continue
        baseline_values = signal[baseline_rows]
        baseline = float(np.nanmedian(baseline_values)) if np.isfinite(baseline_values).any() else float(np.nanmedian(signal))
        delta = signal - baseline
        anchor = ANCHORS[index]
        lower, upper = (10, 340) if index < 8 else ((365, 435) if index == 8 else (345, 425))
        rise = max(0.0, float(np.nanmax(delta)))
        fall = max(0.0, float(-np.nanmin(delta)))
        cap = 1.0 if index < 8 else (2.5 if index == 8 else 0.08)
        gain = min(cap, (upper - anchor) / max(rise, 1e-9), (anchor - lower) / max(fall, 1e-9))
        displayed[:, index] = anchor + delta * gain
    return displayed


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, top: int, bottom: int) -> None:
    for y in range(top, bottom, 16):
        draw.line((x, y, x, min(y + 9, bottom)), fill="#222222", width=3)


def render(destination: Path, subject: str, source: Path, times: np.ndarray, values: np.ndarray, phases: np.ndarray) -> tuple[float, float]:
    sssw = phases == "SSSW"
    start, end = max((interval for interval in runs(sssw) if interval[1] - interval[0] >= 2), key=lambda interval: interval[1] - interval[0])
    phase_times = times[start:end]
    displayed = display_lanes(times, values, phases)[start:end]
    first_time, last_time = float(phase_times[0]), float(phase_times[-1])
    span = max(last_time - first_time, 0.01)

    # Match the compact phase-panel layout in the supplied reference image.
    width, height = 1200, 1120
    left, right, top, bottom = 110, 1090, 300, 890
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def x_position(time_value: float) -> float:
        return left + (time_value - first_time) / span * (right - left)

    def y_position(value: float) -> float:
        return bottom - value / 500 * (bottom - top)

    # Requested phase-band treatment: SSSW title above the interval with exact boundaries.
    phase_y = 235
    draw.line((left, phase_y, right, phase_y), fill="#222222", width=3)
    title = "SSSW*"
    title_box = draw.textbbox((0, 0), title, font=font(26, True))
    draw.rectangle((width / 2 - 90, phase_y - 44, width / 2 + 90, phase_y - 6), fill="white")
    draw.text(((width - (title_box[2] - title_box[0])) / 2, phase_y - 44), title, font=font(26, True), fill="black")
    draw.text((left - 3, phase_y + 12), f"{first_time:.1f}s", font=font(16), fill="#333333")
    end_label = f"{last_time:.1f}"
    end_box = draw.textbbox((0, 0), end_label, font=font(16))
    draw.text((right - (end_box[2] - end_box[0]) + 3, phase_y + 12), end_label, font=font(16), fill="#333333")

    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for level in range(0, 501, 100):
        y = y_position(level)
        draw.line((left, y, right, y), fill="#dedede", width=1)
        number = str(level)
        number_box = draw.textbbox((0, 0), number, font=font(17))
        draw.text((left - 18 - (number_box[2] - number_box[0]), y - 11), number, font=font(17), fill="black")
    # The sample reference retains the whole-trial clock: e.g. 6, 8, 10, 12,
    # rather than re-labelling the displayed SSSW interval to equal fractions.
    tick_start = int(math.ceil(first_time / 2.0) * 2)
    tick_end = int(math.floor(last_time / 2.0) * 2)
    for tick in range(tick_start, tick_end + 1, 2):
        x = x_position(float(tick))
        draw.line((x, bottom, x, bottom + 9), fill="#333333", width=2)
        label = str(tick)
        label_box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((x - (label_box[2] - label_box[0]) / 2, bottom + 15), label, font=font(17), fill="black")
    draw_dashed_vertical(draw, left, phase_y, bottom)
    draw_dashed_vertical(draw, right, phase_y, bottom)

    for index, signal in enumerate(SIGNALS):
        for run_start, run_end in runs(np.isfinite(displayed[:, index])):
            if run_end - run_start >= 2:
                points = [(x_position(float(phase_times[row])), y_position(float(displayed[row, index]))) for row in range(run_start, run_end)]
                draw.line(points, fill=COLORS[index], width=2)
    x_label = "Elapsed time from trial-start trigger (seconds)"
    x_box = draw.textbbox((0, 0), x_label, font=font(20))
    draw.text(((left + right - (x_box[2] - x_box[0])) / 2, 957), x_label, font=font(20), fill="black")
    y_label = Image.new("RGBA", (500, 30), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((0, 0), "Scaled display amplitude (offset per signal)", font=font(18), fill="black")
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (20, 380), y_label)
    # Write beside the previous figure, then atomically replace it. This avoids
    # transient Windows file-handle conflicts while VS Code previews a folder.
    temporary = destination.with_name(destination.stem + ".rendering.png")
    image.save(temporary)
    for attempt in range(20):
        try:
            temporary.replace(destination)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    return first_time, last_time


def output_path(subject: str, source: Path) -> Path:
    side = "l" if "_l_" in source.name else "r"
    return OUTPUT / subject / source.parent.name / f"trial_{trial_number(source):02d}_{side}.png"


def generate_subject(subject: str) -> tuple[int, list[tuple[str, str]]]:
    source_root = segmented_source(subject)
    if source_root is None:
        return 0, [("", "segmented data folder not found")]
    created = 0
    skipped: list[tuple[str, str]] = []
    for source in sorted(source_root.rglob("*_segmented.csv")):
        trial = read_trial(source)
        if trial is None or not np.any(trial[2] == "SSSW"):
            skipped.append((str(source.relative_to(source_root)), "no resolved SSSW interval"))
            continue
        destination = output_path(subject, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        render(destination, subject, source, *trial)
        created += 1
    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=[item.name for item in DATASET.iterdir() if item.is_dir()] + ["all"], default="all")
    args = parser.parse_args()
    subjects = [item.name for item in sorted(DATASET.iterdir()) if item.is_dir()] if args.subject == "all" else [args.subject]
    total = 0
    for subject in subjects:
        created, skipped = generate_subject(subject)
        total += created
        print(f"{subject}: created {created}; unresolved {len(skipped)}")
    print(f"Total SSSW reference-style graphs: {total}")


if __name__ == "__main__":
    main()
