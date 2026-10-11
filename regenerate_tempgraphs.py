"""Create a non-destructive, event-refined replacement set of complete graphs.

The source is the existing time-aligned numerical exports under ``tempgraphs``.
No source waveform samples are changed.  New labels use signal-rise onset for
GI, one data-selected transition stride for SLT, sustained low activity for GT,
and mark five observed reference-foot cycles in each walking phase when those
cycles exist.  Insufficient observations remain explicitly flagged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from subject08_segmentation import change_window, cycle_features, load_peaks


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "tempgraphs"
OUTPUT_ROOT = ROOT / "tempgraphs_resegmented"
DT = 0.01
COLORS = ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#666666", "#d69200")
PHASE_COLORS = {"QS": "#d7e6f5", "GI": "#ffe7c2", "SSSW": "#cfe8cf", "SLT": "#f6c7d4", "SSLW": "#cfdcf5", "GT": "#dfdfdf", "UNRESOLVED": "#f2d2d2"}


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def finite_fill(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    x = np.arange(len(result))
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        if valid.any():
            result[:, column] = np.interp(x, x[valid], result[valid, column])
        else:
            result[:, column] = 0.0
    return result


def moving_mean(values: np.ndarray, width: int = 9) -> np.ndarray:
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0, dtype=float)
    cumsum = np.vstack([np.zeros((1, values.shape[1])), cumsum])
    return (cumsum[width:] - cumsum[:-width]) / width


def activity_series(combined: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Baseline-normalized movement score derived without altering data."""
    smooth = moving_mean(finite_fill(combined), 11)
    baseline_n = max(50, min(100, len(smooth) // 5))
    baseline = smooth[:baseline_n]
    center = np.median(baseline, axis=0)
    scale = np.maximum(1.0, 1.4826 * np.median(np.abs(baseline - center), axis=0))
    level = np.percentile(np.abs((smooth - center) / scale), 70, axis=1)
    derivative = np.percentile(np.abs(np.diff(smooth, axis=0, prepend=smooth[:1])) / scale, 70, axis=1)
    score = np.maximum(level, derivative * 8.0)
    score = moving_mean(score[:, None], 15)[:, 0]
    baseline_high = max(2.5, float(np.percentile(score[:baseline_n], 99)) * 1.8)
    rest_low = max(1.2, float(np.percentile(score[:baseline_n], 99)) * 1.25)
    return score, baseline_high, rest_low


def sustained_start(mask: np.ndarray, minimum: int, start: int = 0) -> int | None:
    run = 0
    for index in range(max(0, start), len(mask)):
        run = run + 1 if mask[index] else 0
        if run >= minimum:
            return index - minimum + 1
    return None


def parse_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    times = np.array([float(row["time_seconds"]) for row in rows], dtype=float)
    values = np.array([
        [float(row.get(f"FMG_channel_{channel}") or "nan") for channel in range(1, 9)] +
        [float(row.get("CoP") or "nan"), float(row.get("vGRF") or "nan")]
        for row in rows
    ], dtype=float)
    imputed = np.array([
        [int(float(row.get(f"FMG_channel_{channel}_imputed", "0") or 0)) for channel in range(1, 9)] +
        [int(float(row.get("CoP_imputed", "0") or 0)), int(float(row.get("vGRF_imputed", "0") or 0))]
        for row in rows
    ], dtype=bool)
    return times, values, imputed


def group_sources() -> dict[str, Path]:
    groups = {}
    for phase_file in SOURCE_ROOT.rglob("phase_boundaries_provisional.csv"):
        parent = phase_file.parent
        if not (parent / "segmented_data").is_dir():
            continue
        name = "sub08_h" if parent == SOURCE_ROOT else parent.name.lower()
        groups[name] = parent
    return dict(sorted(groups.items()))


def pair_files(group: Path):
    source = group / "segmented_data"
    pattern = re.compile(r"trial_(\d+)_l_segmented\.csv$")
    for left in sorted(source.rglob("trial_*_l_segmented.csv")):
        match = pattern.search(left.name)
        if not match:
            continue
        right = left.with_name(left.name.replace("_l_segmented.csv", "_r_segmented.csv"))
        if right.exists():
            yield left.parent.relative_to(source).as_posix(), int(match.group(1)), left, right


def choose_phases(times: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[list[dict], dict, dict]:
    combined = np.column_stack([left[:, :8], right[:, :8], left[:, 8:10], right[:, 8:10]])
    filled = finite_fill(combined)
    peak_l, _ = load_peaks(filled[:, 17])
    peak_r, _ = load_peaks(filled[:, 19])
    reference = "L" if len(peak_l) >= len(peak_r) else "R"
    peaks = peak_l if reference == "L" else peak_r
    activity, high, low = activity_series(filled)
    detail = {
        "reference_limb": reference,
        "load_peaks_left": len(peak_l),
        "load_peaks_right": len(peak_r),
        "onset_threshold": round(float(high), 4),
        "rest_threshold": round(float(low), 4),
        "flags": [],
    }
    if len(peaks) < 11:
        detail["flags"].append("fewer than 11 recurring load-peak steps; the requested 5 SSSW + 1 SLT + 5 SSLW display cannot be supported")
        return [], detail, {"SSSW": [], "SSLW": [], "valleys": []}
    median_stride = max(1, int(round(np.median(np.diff(peaks)))))
    onset = sustained_start(activity >= high, max(20, int(.25 / DT)), 20)
    if onset is None or onset >= peaks[0]:
        onset = max(0, peaks[0] - max(20, median_stride // 2))
        detail["flags"].append("signal-rise onset not independently sustained; conservative pre-peak fallback used")
    else:
        # A delayed activity threshold must not put the start of GI after an
        # already observed first gait peak. Keep GI anchored at the rise just
        # before that first peak, rather than allowing a non-physical phase
        # order.
        latest_valid_onset = max(0, peaks[0] - max(10, median_stride // 5))
        if onset > latest_valid_onset:
            onset = latest_valid_onset
            detail["flags"].append("late activity crossing constrained before the first recurring gait peak")
    usable = [peak for peak in peaks if peak >= peaks[0] and peak <= peaks[-1]]
    features = cycle_features(filled, usable)
    result = change_window(features)
    if result is None:
        detail["flags"].append("feature-only transition score unavailable; protocol-constrained one-step SLT retained")
    last_peak = peaks[-1]
    rest = sustained_start(activity <= low, max(60, int(.75 / DT)), min(len(activity) - 1, last_peak + 10))
    if rest is None:
        rest = last_peak
        detail["flags"].append("no sustained return-to-rest before recording end; GT begins at final accepted peak")
    # A "step" is one observed load peak plus its surrounding observed
    # valleys.  With 12 peaks, the first peak stays in GI, then five SSSW
    # peaks, one SLT peak and five SSLW peaks follow.  Eleven peaks omit the
    # initial GI peak but still support the requested 5 + 1 + 5 pattern.
    if len(peaks) == 11:
        sssw_peaks, slt_peak, sslw_peaks = peaks[:5], peaks[5], peaks[6:11]
        prior_peak = None
    else:
        prior_peak = peaks[0]
        sssw_peaks, slt_peak, sslw_peaks = peaks[1:6], peaks[6], peaks[7:12]
    reference_vgrf = filled[:, 17] if reference == "L" else filled[:, 19]
    def valley_between(start: int, end: int) -> int:
        if end <= start + 1:
            return start
        return start + int(np.argmin(reference_vgrf[start:end + 1]))
    if prior_peak is not None:
        first_sssw_valley = valley_between(prior_peak, sssw_peaks[0])
    else:
        # Do not reuse the quiet-standing minimum as the first walking valley.
        # The first SSSW boundary has to be after the rise that starts GI.
        valley_search_start = max(onset + 1, sssw_peaks[0] - max(10, median_stride // 2))
        first_sssw_valley = valley_between(valley_search_start, sssw_peaks[0])
    slt_start = valley_between(sssw_peaks[-1], slt_peak)
    slt_end = valley_between(slt_peak, sslw_peaks[0])
    rest = max(rest, slt_end + 1)
    if rest >= len(times) - 2:
        detail["flags"].append("terminal rest interval is too short")
        return [], detail, {"SSSW": [], "SSLW": [], "valleys": []}
    boundaries = [0, onset, first_sssw_valley, slt_start, slt_end, rest, len(times)]
    if any(b <= a for a, b in zip(boundaries[:-1], boundaries[1:])):
        detail["flags"].append("non-monotonic event order")
        return [], detail, {"SSSW": [], "SSLW": [], "valleys": []}
    phases = [
        {"phase": name, "start_index": int(a), "end_index": int(b), "start_seconds": round(float(times[a]), 2),
         "end_seconds": round(float(times[b - 1] if b == len(times) else times[b]), 2)}
        for name, a, b in zip(("QS", "GI", "SSSW", "SLT", "SSLW", "GT"), boundaries[:-1], boundaries[1:])
    ]
    def peak_step_spans(selected: list[int], before: int, after: int) -> list[tuple[int, int]]:
        sequence = [before, *selected, after]
        spans = []
        for index, peak in enumerate(selected, 1):
            spans.append((valley_between(sequence[index - 1], peak), valley_between(peak, sequence[index + 1])))
        return spans
    after_sslw = peaks[12] if len(peaks) > 12 else rest
    sssw = peak_step_spans(sssw_peaks, prior_peak if prior_peak is not None else onset, slt_peak)
    sslw = peak_step_spans(sslw_peaks, slt_peak, after_sslw)
    detail["flags"].append("Five-step display is protocol-constrained from observed load peaks; any later observed walking remains in SSLW until sustained rest")
    valleys = []
    for phase, spans in (("SSSW", sssw), ("SSLW", sslw)):
        for number, (start, end) in enumerate(spans, 1):
            valley = start + int(np.argmin(reference_vgrf[start:end + 1]))
            valleys.append({"phase": phase, "step": number, "start_index": start, "end_index": end, "valley_index": valley})
    detail.update({
        "onset_seconds": round(float(times[onset]), 2),
        "gi_end_seconds": round(float(times[first_sssw_valley]), 2),
        "slt_start_seconds": round(float(times[slt_start]), 2),
        "slt_end_seconds": round(float(times[slt_end]), 2),
        "rest_onset_seconds": round(float(times[rest]), 2),
        "transition_improvement": round(float(result["improvement"]), 4) if result is not None else "",
        "complete_five_step_evidence": True,
    })
    return phases, detail, {"SSSW": sssw, "SSLW": sslw, "valleys": valleys}


def runs(mask: np.ndarray):
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            yield start, index if not value else index + 1
            start = None


def draw_graph(path: Path, title: str, times: np.ndarray, values: np.ndarray, imputed: np.ndarray, phases: list[dict], steps: dict, side: str, detail: dict):
    width, height = 2400, 1050
    left, right, top, bottom = 135, 2100, 210, 770
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, normal_font, small_font = font(27, True), font(18), font(15)
    duration = max(float(times[-1]), 0.01)
    def xp(value): return left + value / duration * (right - left)
    draw.text((left, 18), title, fill="#172033", font=title_font)
    draw.text((left, 56), "Event-refined display: raw time-aligned waveforms; labels are operational and not ground truth.", fill="#4b5563", font=normal_font)
    # Phase bar and boundaries.
    for phase in phases:
        a, b, name = phase["start_seconds"], phase["end_seconds"], phase["phase"]
        x1, x2 = xp(a), xp(b)
        draw.rectangle((x1, 118, x2, 158), fill=PHASE_COLORS[name], outline="#6b7280")
        label = name
        box = draw.textbbox((0, 0), label, font=normal_font)
        if x2 - x1 > box[2] - box[0] + 12:
            draw.text(((x1 + x2 - (box[2] - box[0])) / 2, 127), label, fill="#18212f", font=normal_font)
        if a > 0:
            draw.line((x1, top, x1, bottom), fill="#4b5563", width=2)
    draw.line((left, top, left, bottom), fill="#374151", width=2)
    draw.line((left, bottom, right, bottom), fill="#374151", width=2)
    for fraction in range(6):
        x = left + (right - left) * fraction / 5
        time = duration * fraction / 5
        draw.line((x, bottom, x, bottom + 8), fill="#374151", width=2)
        text = f"{time:.1f}"
        box = draw.textbbox((0, 0), text, font=small_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 13), text, fill="#4b5563", font=small_font)
    draw.text(((left + right) / 2 - 145, bottom + 48), "Elapsed time from trigger start (seconds)", fill="#172033", font=normal_font)
    # Per-signal displayed lanes. A small display-only average reduces pixel aliasing; numerical output is unchanged.
    display = moving_mean(finite_fill(values), 5)
    anchors = [180, 150, 120, 90, 60, 35, 310, 280, 490, 455]
    base_end = max(20, min(len(values) // 5, int(round(detail.get("onset_seconds", 1.0) / DT))))
    line_positions = []
    for channel in range(10):
        baseline = float(np.median(display[:base_end, channel]))
        delta = display[:, channel] - baseline
        excursion = max(abs(float(np.percentile(delta, 2))), abs(float(np.percentile(delta, 98))), 1.0)
        scale = min(1.0 if channel < 8 else 1.7, 55.0 / excursion)
        lane = anchors[channel] + delta * scale
        points = [(xp(float(t)), bottom - float(v) / 540.0 * (bottom - top)) for t, v in zip(times, lane)]
        for start, end in runs(np.isfinite(values[:, channel]) & ~imputed[:, channel]):
            if end - start >= 2:
                draw.line(points[start:end], fill=COLORS[channel], width=2)
        legend_y = top + channel * 29
        name = f"FMG {channel+1}" if channel < 8 else ("CoP" if channel == 8 else "vGRF")
        draw.line((2140, legend_y + 10, 2170, legend_y + 10), fill=COLORS[channel], width=3)
        draw.text((2180, legend_y), name, fill="#18212f", font=small_font)
        line_positions.append(lane)
    # Five observed steps and their real valleys are explicitly marked where evidence exists.
    for phase, color, y in (("SSSW", "#1f7a1f", 177), ("SSLW", "#264eaa", 193)):
        spans = steps.get(phase, [])
        for number, (start, end) in enumerate(spans, 1):
            x1, x2 = xp(float(times[start])), xp(float(times[end]))
            draw.line((x1, y, x2, y), fill=color, width=3)
            draw.line((x1, y - 5, x1, y + 5), fill=color, width=2)
            draw.line((x2, y - 5, x2, y + 5), fill=color, width=2)
            draw.text(((x1 + x2) / 2 - 4, y - 20), str(number), fill=color, font=small_font)
    local_vgrf = line_positions[9]
    for valley in steps.get("valleys", []):
        index = valley["valley_index"]
        x = xp(float(times[index])); y = bottom - float(local_vgrf[index]) / 540.0 * (bottom - top)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#111827")
    flags = detail.get("flags", [])
    evidence = "Five observed steps marked in both walking phases." if detail.get("complete_five_step_evidence") else "Five-step evidence incomplete; see manifest flags."
    draw.text((left, 835), evidence, fill="#7c3e00" if flags else "#1f5f33", font=normal_font)
    draw.text((left, 868), "Valley dots mark actual reference-vGRF minima inside the displayed step intervals; they are not fabricated events.", fill="#4b5563", font=small_font)
    draw.text((left, 895), ("; ".join(flags) or "QS ends at sustained signal rise; GT begins at sustained low activity."), fill="#4b5563", font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_csv(path: Path, header: list[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def generate_group(name: str, source_group: Path) -> dict:
    destination = OUTPUT_ROOT / name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    records, intervals, step_rows, manifest = [], [], [], []
    total_graphs = total_pairs = full_five = unresolved = 0
    for record, trial, left_path, right_path in pair_files(source_group):
        times_l, left, imputed_l = parse_csv(left_path)
        times_r, right, imputed_r = parse_csv(right_path)
        if len(times_l) != len(times_r) or not np.allclose(times_l, times_r, atol=1e-9):
            raise ValueError(f"Unaligned limb exports: {left_path} / {right_path}")
        phases, detail, steps = choose_phases(times_l, left, right)
        total_pairs += 1
        status = "event_refined" if phases else "unresolved"
        if phases:
            intervals.extend({"record": record, "trial": trial, **phase} for phase in phases)
            full_five += int(detail.get("complete_five_step_evidence", False))
        else:
            unresolved += 1
        records.append({"record": record, "trial": trial, "status": status, **detail})
        labels = np.full(len(times_l), "UNRESOLVED", dtype="<U12")
        for phase in phases:
            labels[phase["start_index"]:phase["end_index"]] = phase["phase"]
        for item in steps.get("valleys", []):
            step_rows.append({"record": record, "trial": trial, **item,
                              "start_seconds": round(float(times_l[item["start_index"]]), 2),
                              "end_seconds": round(float(times_l[item["end_index"]]), 2),
                              "valley_seconds": round(float(times_l[item["valley_index"]]), 2)})
        for side, values, imputed in (("L", left, imputed_l), ("R", right, imputed_r)):
            csv_path = destination / "segmented_data" / record / f"trial_{trial:02d}_{side.lower()}_segmented.csv"
            rows = []
            for time, data, label, flag in zip(times_l, values, labels, imputed):
                rows.append([f"{time:.2f}", *[f"{value:.17g}" if math.isfinite(value) else "" for value in data], label,
                             "event_refined_not_ground_truth" if phases else "unresolved", int(not np.all(np.isfinite(data))), *flag.astype(int)])
            write_csv(csv_path, ["time_seconds", *[f"FMG_channel_{index}" for index in range(1, 9)], "CoP", "vGRF", "provisional_phase", "label_status", "signal_missing", *[f"FMG_channel_{index}_imputed" for index in range(1, 9)], "CoP_imputed", "vGRF_imputed"], rows)
            png_path = destination / record / f"trial_{trial:02d}_{side.lower()}_complete.png"
            draw_graph(png_path, f"{name.upper()} | {record} | trial {trial:02d} | {side}", times_l, values, imputed, phases, steps, side, detail)
            manifest.append({"record": record, "trial": trial, "side": side, "png": str(png_path.relative_to(destination)), "segmented_csv": str(csv_path.relative_to(destination)), "status": status, **detail})
            total_graphs += 1
    def write_dicts(filename, rows):
        rows = list(rows)
        keys = sorted({key for row in rows for key in row}) if rows else []
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    write_dicts("phase_boundaries_event_refined.csv", records)
    write_dicts("phase_intervals_event_refined.csv", intervals)
    write_dicts("five_step_valley_manifest.csv", step_rows)
    write_dicts("complete_trial_manifest.csv", manifest)
    summary = {
        "source_group": str(source_group.relative_to(SOURCE_ROOT)), "physical_windows": total_pairs, "limb_graphs": total_graphs,
        "event_refined_six_phase_windows": total_pairs - unresolved, "unresolved_windows": unresolved,
        "windows_with_five_observed_steps_marked_in_both_SSSW_and_SSLW": full_five,
        "method": "Existing time-aligned numerical waveform values retained. QS ends at sustained signal rise; GI runs from that rise to the first SSSW valley; SLT is one protocol-constrained observed peak step; GT starts at sustained low activity after the gait sequence. Five visible peak steps are marked in both walking phases only when at least 11 recurring load peaks exist.",
        "warning": "Operational signal labels only. Candidate peaks and valleys are not validated gait events or clinical ground truth.",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        "# Event-refined tempgraphs\n\n"
        "This non-destructive output is regenerated from the matching current `tempgraphs` numerical exports. Waveform samples are preserved; only phase labels, step annotations, and display rendering are new.\n\n"
        "Five SSSW and five SSLW peak steps are marked only when the recorded reference limb contains at least 11 recurring load peaks. No missing steps, valleys, or repetitions are fabricated. `five_step_valley_manifest.csv` records each displayed step and its observed vGRF minimum.\n",
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", help="Source group to regenerate; omit for all groups.")
    args = parser.parse_args()
    groups = group_sources()
    selected = args.group or list(groups)
    unknown = sorted(set(selected) - set(groups))
    if unknown:
        raise SystemExit(f"Unknown group(s): {', '.join(unknown)}. Available: {', '.join(groups)}")
    OUTPUT_ROOT.mkdir(exist_ok=True)
    refreshed = {item["source_group"]: item for item in (json.loads((OUTPUT_ROOT / "SUMMARY.json").read_text()) if (OUTPUT_ROOT / "SUMMARY.json").exists() else [])}
    for name in selected:
        item = generate_group(name, groups[name])
        refreshed[item["source_group"]] = item
    summaries = [refreshed[key] for key in sorted(refreshed)]
    (OUTPUT_ROOT / "SUMMARY.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([refreshed[("." if name == "sub08_h" else name)] for name in selected], indent=2))


if __name__ == "__main__":
    main()
