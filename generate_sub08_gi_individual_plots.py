"""Create proper GI-only plots for resolved Sub08_H trials 01-15."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_qs_plots import font
from subject08_review import COLORS, runs


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tempgraphs" / "segmented_data" / "Sub08_H"
DESTINATION = ROOT / "Individual Outputs" / "GI" / "Per Trial" / "Sub08_H"
SIGNALS = tuple([f"FMG_channel_{number}" for number in range(1, 9)] + ["CoP", "vGRF"])
MAX_TRIAL = 15


def trial_number(path: Path) -> int:
    match = re.search(r"trial_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot determine trial number from {path.name}")
    return int(match.group(1))


def load_trial(path: Path) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Return the complete saved signal and its exact existing GI interval."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=float)
    values = np.asarray(
        [[float(row[name]) if row[name] else np.nan for name in SIGNALS] for row in rows], dtype=float
    )
    gi = np.asarray([row["provisional_phase"] == "GI" for row in rows], dtype=bool)
    intervals = [(start, end) for start, end in runs(gi) if end - start >= 2]
    if not intervals:
        return None
    start, end = max(intervals, key=lambda interval: interval[1] - interval[0])
    return times, values, start, end


def full_trial_display(values: np.ndarray, times: np.ndarray, gi_start: float) -> np.ndarray:
    """Use the original full-trial baseline and display transform for GI data."""
    anchors = (200, 170, 145, 120, 100, 80, 60, 40, 400, 380)
    baseline_window = times < max(0.25, min(gi_start, 0.75))
    display = np.full_like(values, np.nan)
    for index in range(values.shape[1]):
        signal = values[:, index]
        valid = np.isfinite(signal)
        if not valid.any():
            continue
        baseline_values = signal[baseline_window]
        baseline = float(np.nanmedian(baseline_values)) if np.isfinite(baseline_values).any() else float(np.nanmedian(signal))
        delta = signal - baseline
        anchor = anchors[index]
        lower, upper = (10, 340) if index < 8 else ((365, 435) if index == 8 else (345, 425))
        rise = max(0.0, float(np.nanmax(delta)))
        fall = max(0.0, float(-np.nanmin(delta)))
        cap = 1.0 if index < 8 else (2.5 if index == 8 else 0.08)
        gain = min(cap, (upper - anchor) / max(rise, 1e-9), (anchor - lower) / max(fall, 1e-9))
        display[:, index] = anchor + delta * gain
    return display


def render(destination: Path, source: Path, times: np.ndarray, values: np.ndarray, start: int, end: int) -> None:
    """Render a clean graph containing exactly the existing GI-labelled rows."""
    gi_times = times[start:end]
    gi_values = values[start:end]
    gi_display = full_trial_display(values, times, float(gi_times[0]))[start:end]
    width, height = 2200, 880
    left, right, top, bottom = 120, 1940, 160, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x_start, x_end = float(gi_times[0]), float(gi_times[-1])
    duration = max(x_end - x_start, 0.01)

    def x_position(time_value: float) -> float:
        return left + (time_value - x_start) / duration * (right - left)

    def y_position(value: float) -> float:
        return bottom - value / 500 * (bottom - top)

    side = "Left" if "_l_" in source.name else "Right"
    record = source.parent.name
    trial = trial_number(source)
    draw.text((left, 16), f"Sub08_H | {record} | Trial {trial:02d} | {side} | Gait Initiation (GI)", font=font(25, True), fill="black")
    draw.text((left, 51), f"GI only: {x_start:.2f}-{x_end:.2f} s after trial start | Same display scale as the full-trial graph", font=font(18), fill="#744600")
    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for level in range(0, 501, 100):
        y = y_position(level)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((left - 48, y - 11), str(level), font=font(17), fill="black")
    for tick in np.linspace(x_start, x_end, 5):
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
        valid = np.isfinite(gi_display[:, index])
        for run_start, run_end in runs(valid):
            if run_end - run_start >= 2:
                points = [
                    (x_position(float(gi_times[row])), y_position(float(gi_display[row, index])))
                    for row in range(run_start, run_end)
                ]
                draw.line(points, fill=COLORS[index], width=2)
        legend_y = top + index * 30
        draw.line((1970, legend_y + 10, 2000, legend_y + 10), fill=COLORS[index], width=3)
        draw.text((2010, legend_y), name.replace("FMG_", ""), font=font(16), fill="black")

    draw.text((left, 754), "GI only. No QS, SSSW, SLT, SSLW, or GT rows are plotted.", font=font(18, True), fill="#744600")
    draw.text((left, 783), "The window uses the existing GI boundaries from the corresponding full Sub08_H graph in tempgraphs.", font=font(16), fill="#555555")
    draw.text((left, 811), "This is a new graph from the saved numeric data, not a screenshot crop.", font=font(16), fill="#555555")
    image.save(destination)


def output_name(source: Path) -> str:
    return f"{source.parent.name}_trial_{trial_number(source):02d}_{'l' if '_l_' in source.name else 'r'}_GI.png"


def generate() -> tuple[int, list[str]]:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped: list[str] = []
    for source in sorted(SOURCE.rglob("*_segmented.csv")):
        if trial_number(source) > MAX_TRIAL:
            continue
        trial = load_trial(source)
        if trial is None:
            skipped.append(str(source.relative_to(SOURCE)))
            continue
        times, values, start, end = trial
        render(DESTINATION / output_name(source), source, times, values, start, end)
        created += 1
    return created, skipped


if __name__ == "__main__":
    created, skipped = generate()
    print(f"GI-only graphs created: {created}")
    print(f"Skipped without an existing GI boundary: {len(skipped)}")
    for item in skipped:
        print(f"  {item}")
