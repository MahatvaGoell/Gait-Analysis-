"""Create GI-only per-trial graphs for every subject with segmented data."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_qs_plots import font
from subject08_review import COLORS, activity, runs


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
TEMPGRAPHS = ROOT / "tempgraphs"
OUTPUT = ROOT / "Individual Outputs" / "GI" / "Per Trial"
SIGNALS = tuple([f"FMG_channel_{number}" for number in range(1, 9)] + ["CoP", "vGRF"])


def trial_number(path: Path) -> int:
    match = re.search(r"trial_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot determine trial number from {path.name}")
    return int(match.group(1))


def segmented_source(subject: str) -> Path | None:
    """Locate the existing segmented data without changing it."""
    subject_folder = TEMPGRAPHS / subject.lower() / "segmented_data"
    root_folder = TEMPGRAPHS / "segmented_data" / subject
    if subject_folder.is_dir():
        return subject_folder
    if root_folder.is_dir():
        return root_folder
    return None


def initiation_peak_end(values: np.ndarray, start: int, current_end: int) -> int:
    """Extend GI to its first prominent, rising multi-sensor movement peak."""
    score = np.maximum(activity(values[:, :8]), activity(values[:, 8:]))
    score = np.nan_to_num(score, nan=0.0)
    # Merge momentary pauses within the same initiation movement.
    active = score > 1.0
    for gap_start, gap_end in runs(~active):
        if gap_start > 0 and gap_end < len(active) and gap_end - gap_start <= 35:
            active[gap_start:gap_end] = True
    # A short moving mean retains the rise while preventing one-sample noise
    # from becoming a false movement peak.
    smoothed = np.convolve(score, np.ones(11) / 11, mode="same")
    for index in range(max(start + 15, 20), len(smoothed) - 15):
        local = smoothed[index - 15 : index + 16]
        rise = smoothed[index] - np.min(smoothed[index - 20 : index + 1])
        if active[index] and smoothed[index] >= 2.0 and smoothed[index] >= np.max(local) and rise >= 0.3:
            return max(current_end, index + 1)
    return current_end


def read_gi_interval(path: Path) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Read a complete saved trial and its GI start through the movement peak."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[name]) if row[name] else np.nan for name in SIGNALS] for row in rows], dtype=float
    )
    gi_mask = np.asarray([row["provisional_phase"] == "GI" for row in rows], dtype=bool)
    intervals = [(start, end) for start, end in runs(gi_mask) if end - start >= 2]
    if not intervals:
        return None
    start, end = max(intervals, key=lambda interval: interval[1] - interval[0])
    return times, values, start, initiation_peak_end(values, start, end)


def full_trial_display(values: np.ndarray, times: np.ndarray, gi_start: float) -> np.ndarray:
    """Reuse the full-trial baseline and lane scale for the GI-only view."""
    anchors = (200, 170, 145, 120, 100, 80, 60, 40, 400, 380)
    baseline_window = times < max(0.25, min(gi_start, 0.75))
    display = np.full_like(values, np.nan)
    for index in range(values.shape[1]):
        signal = values[:, index]
        valid = np.isfinite(signal)
        if not valid.any():
            continue
        base_values = signal[baseline_window]
        baseline = float(np.nanmedian(base_values)) if np.isfinite(base_values).any() else float(np.nanmedian(signal))
        delta = signal - baseline
        anchor = anchors[index]
        lower, upper = (10, 340) if index < 8 else ((365, 435) if index == 8 else (345, 425))
        rise = max(0.0, float(np.nanmax(delta)))
        fall = max(0.0, float(-np.nanmin(delta)))
        gain_cap = 1.0 if index < 8 else (2.5 if index == 8 else 0.08)
        gain = min(gain_cap, (upper - anchor) / max(rise, 1e-9), (anchor - lower) / max(fall, 1e-9))
        display[:, index] = anchor + delta * gain
    return display


def output_name(source: Path) -> str:
    side = "l" if "_l_" in source.name else "r"
    return f"{source.parent.name}_trial_{trial_number(source):02d}_{side}_GI.png"


def render(destination: Path, subject: str, source: Path, times: np.ndarray, values: np.ndarray, start: int, end: int) -> None:
    """Render a graph from numeric data, limited exactly to GI rows."""
    gi_times = times[start:end]
    displayed = full_trial_display(values, times, float(gi_times[0]))[start:end]
    width, height = 2200, 880
    left, right, top, bottom = 120, 1940, 160, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    first_time, last_time = float(gi_times[0]), float(gi_times[-1])
    span = max(last_time - first_time, 0.01)

    def x_position(time_value: float) -> float:
        return left + (time_value - first_time) / span * (right - left)

    def y_position(value: float) -> float:
        return bottom - value / 500 * (bottom - top)

    side = "Left" if "_l_" in source.name else "Right"
    draw.text((left, 16), f"{subject} | {source.parent.name} | Trial {trial_number(source):02d} | {side} | Gait Initiation (GI)", font=font(25, True), fill="black")
    draw.text((left, 51), f"GI only: {first_time:.2f}-{last_time:.2f} s after trial start | Same display scale as its full-trial graph", font=font(18), fill="#744600")
    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for level in range(0, 501, 100):
        y = y_position(level)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((left - 48, y - 11), str(level), font=font(17), fill="black")
    for tick in np.linspace(first_time, last_time, 5):
        x = x_position(float(tick))
        draw.line((x, bottom, x, bottom + 9), fill="black", width=2)
        text = f"{tick:.2f}"
        box = draw.textbbox((0, 0), text, font=font(17))
        draw.text((x - (box[2] - box[0]) / 2, bottom + 15), text, font=font(17), fill="black")
    draw.text((715, 700), "Elapsed time from trial-start trigger (seconds)", font=font(21), fill="black")
    label = Image.new("RGBA", (480, 32), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((0, 0), "Scaled display amplitude (offset per signal)", font=font(19), fill="black")
    label = label.rotate(90, expand=True)
    image.paste(label, (25, 220), label)
    for index, name in enumerate(SIGNALS):
        for run_start, run_end in runs(np.isfinite(displayed[:, index])):
            if run_end - run_start >= 2:
                points = [(x_position(float(gi_times[row])), y_position(float(displayed[row, index]))) for row in range(run_start, run_end)]
                draw.line(points, fill=COLORS[index], width=2)
        legend_y = top + index * 30
        draw.line((1970, legend_y + 10, 2000, legend_y + 10), fill=COLORS[index], width=3)
        draw.text((2010, legend_y), name.replace("FMG_", ""), font=font(16), fill="black")
    draw.text((left, 754), "GI only. No QS, SSSW, SLT, SSLW, or GT rows are plotted.", font=font(18, True), fill="#744600")
    draw.text((left, 783), "GI starts at the saved onset and ends at the first prominent multi-sensor movement peak.", font=font(16), fill="#555555")
    image.save(destination)


def generate_subject(subject: str) -> tuple[int, list[str]]:
    source_root = segmented_source(subject)
    if source_root is None:
        return 0, ["segmented data folder not found"]
    created = 0
    skipped: list[str] = []
    for source in sorted(source_root.rglob("*_segmented.csv")):
        # Keep the already requested Sub08_H scope: trials 01 through 15 only.
        if subject == "Sub08_H" and trial_number(source) > 15:
            continue
        interval = read_gi_interval(source)
        if interval is None:
            skipped.append(str(source.relative_to(source_root)))
            continue
        times, values, start, end = interval
        destination = OUTPUT / subject / output_name(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        render(destination, subject, source, times, values, start, end)
        created += 1
    return created, skipped


if __name__ == "__main__":
    for subject_folder in sorted(item for item in DATASET.iterdir() if item.is_dir()):
        graphs, skipped = generate_subject(subject_folder.name)
        print(f"{subject_folder.name}: created {graphs} GI-only graphs; skipped {len(skipped)}")
        for item in skipped:
            print(f"  {item}")
