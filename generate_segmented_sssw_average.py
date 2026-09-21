"""Create the combined Sub01_A SSSW display average from the corrected SSSW traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_qs_plots import font
from generate_sssw_reference_plots import SIGNALS, display_lanes, read_trial
from subject08_review import COLORS


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tempgraphs" / "sub01_a" / "segmented_data"
OUTPUT = ROOT / "Segmented Average SSSW"
DATA_OUTPUT = OUTPUT / "average_data" / "sssw_display_average_all_signals.csv"
GRAPH_OUTPUT = OUTPUT / "average_graphs" / "sssw_average_all_signals.png"
TRACE_COUNT = 300


def sssw_section(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    trial = read_trial(path)
    if trial is None:
        return None
    times, values, phases = trial
    rows = np.flatnonzero(phases == "SSSW")
    if len(rows) < 2:
        return None
    # Saved Sub01_A labels contain one continuous SSSW run per resolved trace.
    displayed = display_lanes(times, values, phases)
    return times[rows], displayed[rows]


def normalized_traces() -> tuple[np.ndarray, list[dict[str, object]]]:
    traces: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    target = np.linspace(0.0, 1.0, TRACE_COUNT)
    for path in sorted(SOURCE.rglob("*_segmented.csv")):
        section = sssw_section(path)
        if section is None:
            continue
        times, values = section
        local = np.linspace(0.0, 1.0, len(times))
        trace = np.column_stack([np.interp(target, local, values[:, column]) for column in range(values.shape[1])])
        traces.append(trace)
        manifest.append(
            {
                "source_csv": str(path.relative_to(ROOT)).replace("\\", "/"),
                "record": path.parent.name,
                "trial": path.stem.split("_")[1],
                "side": "L" if "_l_" in path.name else "R",
                "sssw_start_seconds": float(times[0]),
                "sssw_end_seconds": float(times[-1]),
                "source_samples": int(len(times)),
            }
        )
    if not traces:
        raise RuntimeError("No resolved Sub01_A SSSW traces found")
    return np.stack(traces), manifest


def write_data(normalized_percent: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray, counts: np.ndarray) -> None:
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    headings = ["normalized_sssw_percent"]
    headings += [f"display_{name.lower()}_mean" for name in SIGNALS]
    headings += [f"display_{name.lower()}_standard_deviation" for name in SIGNALS]
    headings += [f"{name.lower()}_contributing_trace_count" for name in SIGNALS]
    with DATA_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headings)
        for row in range(len(normalized_percent)):
            writer.writerow(
                [f"{normalized_percent[row]:.12g}"]
                + [f"{value:.12g}" for value in mean[row]]
                + [f"{value:.12g}" for value in standard_deviation[row]]
                + [int(value) for value in counts[row]]
            )


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, top: int, bottom: int) -> None:
    for y in range(top, bottom, 16):
        draw.line((x, y, x, min(y + 9, bottom)), fill="#222222", width=3)


def render(normalized_percent: np.ndarray, average: np.ndarray, trace_count: int) -> None:
    width, height = 1200, 1120
    left, right, top, bottom = 110, 1090, 300, 890
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def x_position(percent: float) -> float:
        return left + percent / 100.0 * (right - left)

    def y_position(value: float) -> float:
        return bottom - value / 500.0 * (bottom - top)

    phase_y = 235
    draw.line((left, phase_y, right, phase_y), fill="#222222", width=3)
    title = "SSSW Average*"
    box = draw.textbbox((0, 0), title, font=font(26, True))
    draw.rectangle((width / 2 - 140, phase_y - 44, width / 2 + 140, phase_y - 6), fill="white")
    draw.text(((width - (box[2] - box[0])) / 2, phase_y - 44), title, font=font(26, True), fill="black")
    draw.text((left - 3, phase_y + 12), "0%", font=font(16), fill="#333333")
    end_box = draw.textbbox((0, 0), "100%", font=font(16))
    draw.text((right - (end_box[2] - end_box[0]) + 3, phase_y + 12), "100%", font=font(16), fill="#333333")

    draw.line((left, top, left, bottom, right, bottom), fill="#333333", width=2)
    for level in range(0, 501, 100):
        y = y_position(level)
        draw.line((left, y, right, y), fill="#dedede", width=1)
        label = str(level)
        label_box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((left - 18 - (label_box[2] - label_box[0]), y - 11), label, font=font(17), fill="black")
    for tick in range(20, 100, 20):
        x = x_position(float(tick))
        draw.line((x, bottom, x, bottom + 9), fill="#333333", width=2)
        label = str(tick)
        label_box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((x - (label_box[2] - label_box[0]) / 2, bottom + 15), label, font=font(17), fill="black")
    draw_dashed_vertical(draw, left, phase_y, bottom)
    draw_dashed_vertical(draw, right, phase_y, bottom)

    for index in range(len(SIGNALS)):
        points = [(x_position(float(percent)), y_position(float(value))) for percent, value in zip(normalized_percent, average[:, index])]
        draw.line(points, fill=COLORS[index], width=2)

    x_label = "Normalized SSSW phase (%)"
    x_box = draw.textbbox((0, 0), x_label, font=font(20))
    draw.text(((left + right - (x_box[2] - x_box[0])) / 2, 957), x_label, font=font(20), fill="black")
    y_label = Image.new("RGBA", (500, 30), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((0, 0), "Mean scaled display amplitude (offset per signal)", font=font(18), fill="black")
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (20, 350), y_label)
    GRAPH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = GRAPH_OUTPUT.with_name(GRAPH_OUTPUT.stem + ".rendering.png")
    image.save(temporary)
    temporary.replace(GRAPH_OUTPUT)


def main() -> None:
    traces, manifest = normalized_traces()
    normalized_percent = np.linspace(0.0, 100.0, TRACE_COUNT)
    mean = np.nanmean(traces, axis=0)
    standard_deviation = np.nanstd(traces, axis=0, ddof=1)
    counts = np.sum(np.isfinite(traces), axis=0)
    write_data(normalized_percent, mean, standard_deviation, counts)
    render(normalized_percent, mean, len(traces))
    summary_path = OUTPUT / "average_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["combined_display_average"] = {
        "source_trace_count": len(traces),
        "signals": [name.replace("FMG_", "") for name in SIGNALS],
        "normalization": "Each trial's existing SSSW section was normalized to 300 phase-percent samples before calculating the point-by-point mean.",
        "display_transform": "Each source trace uses the full-trial baseline, gain and vertical lane offset used by the corrected SSSW per-trial figures.",
        "numerical_csv": "average_data/sssw_display_average_all_signals.csv",
        "average_graph": "average_graphs/sssw_average_all_signals.png",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Created combined SSSW display average from {len(traces)} traces")


if __name__ == "__main__":
    main()
