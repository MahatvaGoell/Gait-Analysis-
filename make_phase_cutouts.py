"""Normalised cutouts of one gait phase for every cut limb graph.

Reads the phase cuts written by `refine_sub08_h_valleys.py` (`phase_cuts.csv` in
each subject folder) and the 100 Hz exports, and writes one graph per limb graph
that shows only the requested phase.

Normalisation, identical for every graph so that they can be compared:
  time       0-100 % of the phase (its first sample to its last sample). In the
             walking phases (SSSW, SLT, SSLW) every step, from one CoP valley to
             the next, is drawn at the same width, so the step cuts sit at the
             same place on every graph; the real valley samples label the axis.
             GT shows the same 3 s after the twelfth valley on every graph, in
             real time, with the point where the person comes to rest marked;
  amplitude  each signal as its change from quiet standing (median of the last
             0.5 s before the rise), scaled so that its largest change inside
             the phase is 1. Every signal gets its own lane, so the shape and
             direction of its change are visible however small or large it is.
The size that the scaling removes is printed next to each lane: the largest
change as a percentage of that signal's own walking range (1st-99th percentile
between the rise and the start of GT). A signal that changes by less than 2 %
of its walking range, by less than four times its quiet-standing noise, or by
less than three sensor steps, is drawn flat.
QS is the exception: nothing is blown up to fill the lane there. Each signal is
its change from the pre-rise level as a fraction of its walking range (a lane is
+/-50 %), so quiet standing looks as flat as it is.

Outputs under NewSegmentedGraphsFinal/<PHASE>/:
  <Subject>/<record>_trial_##_<l|r>_<PHASE>.png   the cutouts
  <phase>_manifest.csv                            window, duration, change sizes and notes of every cutout
  <phase>_normalised_values.csv                   the normalised signals, resampled to 0..100 %
  README.md

Run:  python make_phase_cutouts.py GI
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from refine_sub08_h_valleys import (CHANNEL_COLORS, CHANNEL_NAMES, PHASES, SUBJECTS, VALUE_COLUMNS,
                                    finite_fill, font, pale, parse_csv, runs, write_dicts)


OUTPUT = Path(__file__).resolve().parent / "NewSegmentedGraphsFinal"
PHASE_TITLES = {"QS": "Quiet standing", "GI": "Gait initiation", "SSSW": "Steady-state short-step walking",
                "SLT": "Short-to-long transition", "SSLW": "Steady-state long-step walking", "GT": "Gait termination"}
PHASE_SPANS = {"GI": "rise out of quiet standing to the first CoP valley",
               "SSSW": "first CoP valley (v0) to the sixth (v5): five steps",
               "SLT": "sixth CoP valley (v5) to the seventh (v6): the transition step",
               "SSLW": "seventh CoP valley (v6) to the twelfth (v11): five steps"}
PHASE_SPANS.update({"QS": "recording start to the rise out of quiet standing",
                    "GT": "the 3 s after the twelfth CoP valley (v11): the last steps and coming to rest"})
STEP_PHASES = ("SSSW", "SLT", "SSLW")
GT_WINDOW_SECONDS = 3.0    # every GT cutout shows the same real-time window after v11
QS_LANE = 0.5              # in QS a lane spans +/- this fraction of the signal's walking range
STANCE_TINTS = {"SSSW": "#e6f4e7", "SLT": "#fbe4ec", "SSLW": "#e4edfa"}
BASELINE_SAMPLES = 50     # 0.5 s of quiet standing just before the rise
SHORT_PHASE_SECONDS = 0.30
MIN_CHANGE = 0.02         # of the walking range; below this a signal is drawn flat
LANES = (8, 9, 0, 1, 2, 3, 4, 5, 6, 7)   # top to bottom: CoP, vGRF, Channel 1-8


def normalise(values: np.ndarray, rise: int, gt_start: int, start: int, end: int, by_walking_range: bool = False):
    """Change from quiet standing inside [start, end], scaled per signal.

    Normally each signal is scaled to its largest change in the window. With `by_walking_range`
    (quiet standing) it is scaled to its walking range instead, so that nothing is blown up.
    Returns the normalised signals, the largest change of each signal as a signed fraction of its
    walking range, and which signals are drawn as a curve.
    """
    filled = finite_fill(values)
    quiet = filled[max(0, rise - BASELINE_SAMPLES):max(rise, 1)]
    baseline = np.median(quiet, axis=0)
    noise = 1.4826 * np.median(np.abs(quiet - baseline), axis=0)
    walking = filled[rise:max(gt_start, rise + 2)]
    span = np.percentile(walking, 99, axis=0) - np.percentile(walking, 1, axis=0)
    delta = filled - baseline
    window = delta[start:end + 1]
    at = np.argmax(np.abs(window), axis=0)
    peak = window[at, np.arange(window.shape[1])]            # signed largest change in the phase
    alive = span > 1e-9 * np.maximum(1.0, np.abs(baseline))  # a dead or constant channel has no range
    fraction = np.where(alive, peak / np.where(alive, span, 1.0), 0.0)
    if by_walking_range:
        return np.where(alive, delta / np.where(alive, span, 1.0) / QS_LANE, 0.0), fraction, alive.copy(), alive
    # Sensor resolution: the typical size of one step between neighbouring samples (1 count for FMG).
    steps = np.abs(np.diff(filled, axis=0))
    resolution = np.array([np.median(column[column > 1e-9]) if np.any(column > 1e-9) else 0.0 for column in steps.T])
    changes = alive & (np.abs(fraction) >= MIN_CHANGE) & (np.abs(peak) >= 4 * noise) & (np.abs(peak) >= 3 * resolution)
    normalised = np.where(changes, delta / np.where(changes, np.abs(peak), 1.0), 0.0)
    return normalised, fraction, changes, alive


def step_percent(steps: list[tuple[str, int, int, int]], sample: float) -> float:
    """Position of a sample on the 0-100 % axis when every step has the same width."""
    for k, (_, valley, _, following) in enumerate(steps):
        if sample <= following or k == len(steps) - 1:
            return (k + (sample - valley) / max(following - valley, 1)) / len(steps) * 100.0
    return 100.0


def draw(path: Path, title: str, phase: str, start: int, end: int, normalised: np.ndarray, fraction: np.ndarray,
         changes: np.ndarray, alive: np.ndarray, estimated: np.ndarray, notes: list[str],
         steps: list[tuple[str, int, int, int]] | None = None, context: str = "",
         seconds: float = 0.0, marks: list[tuple[int, str]] = (), by_walking_range: bool = False):
    """`steps` = (name, valley, peak, next valley) of every step of a walking phase; None for the other phases.

    `seconds` > 0 draws a real-time axis of that length starting at `start`; `marks` are (sample, label)
    events drawn as dashed lines; `by_walking_range` is the quiet-standing scaling.
    """
    w, h = 2200, 1270
    image = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 1890, 150, 1110
    lane = (bottom - top) / len(LANES)
    samples = np.arange(start, end + 1)
    if steps:
        percent = np.array([step_percent(steps, sample) for sample in samples])
    elif seconds:
        percent = (samples - start) / (seconds * 100.0) * 100.0
    else:
        percent = (samples - start) / max(end - start, 1) * 100.0
    def xp(p): return left + p / 100.0 * (right - left)
    if steps:   # stance (CoP valley to CoP peak) of every step
        for name, valley, peak, _ in steps:
            d.rectangle((xp(step_percent(steps, valley)), top, xp(step_percent(steps, peak)), bottom - 1), fill=STANCE_TINTS[phase])

    d.text((left, 20), title, font=font(23, True), fill="#111111")
    d.text((left, 57), f"{PHASE_TITLES[phase]} only: {PHASE_SPANS.get(phase, 'phase span from phase_cuts.csv')} | samples {start}-{end} "
                       f"({(end - start) / 100:.2f} s) | solid: source-grid values | pale: estimated values", font=font(15), fill="#8a6d1f")
    if context:
        d.text((left, 78), context, font=font(14), fill="#555555")
    text = phase + "*"
    tb = d.textbbox((0, 0), text, font=font(19, True))
    centre = (left + right) / 2
    d.text((centre - tb[2] / 2, 98), text, font=font(19, True), fill="#111111")
    d.line((left + 6, 108, centre - tb[2] / 2 - 14, 108), fill="#222222", width=2)
    d.line((centre + tb[2] / 2 + 14, 108, right - 6, 108), fill="#222222", width=2)

    d.line((left, top, left, bottom, right, bottom), fill="#555555", width=2)
    if steps:
        boundaries = [steps[0][1]] + [following for _, _, _, following in steps]
        for k, sample in enumerate(boundaries):      # step cuts, labelled with the real CoP valley sample
            x = xp(k / len(steps) * 100.0)
            if 0 < k < len(steps):
                for yy in range(top, bottom, 13):
                    d.line((x, yy, x, min(yy + 7, bottom)), fill="#444444", width=2)
            d.line((x, bottom, x, bottom + 7), fill="#555555", width=2)
            tb = d.textbbox((0, 0), str(sample), font=font(17))
            d.text((x - tb[2] / 2, bottom + 14), str(sample), font=font(17), fill="#444444")
        for k, (name, valley, peak, following) in enumerate(steps):
            middle = xp((k + 0.5) / len(steps) * 100.0)
            tb = d.textbbox((0, 0), name, font=font(17, True))
            d.text((middle - tb[2] - 6, top - 26), name, font=font(17, True), fill="#333333")
            d.text((middle + 2, top - 24), f"{(following - valley) / 100:.2f} s, stance {(peak - valley) / 100:.2f} s", font=font(14), fill="#666666")
    elif seconds:
        for tick in np.arange(0.0, seconds + 1e-9, 0.5):
            x = xp(tick / seconds * 100.0)
            d.line((x, bottom, x, bottom + 7), fill="#555555", width=2)
            tb = d.textbbox((0, 0), f"{tick:.1f}", font=font(17))
            d.text((x - tb[2] / 2, bottom + 14), f"{tick:.1f}", font=font(17), fill="#444444")
    else:
        for tick in range(0, 101, 10):
            d.line((xp(tick), bottom, xp(tick), bottom + 7), fill="#555555", width=2)
            tb = d.textbbox((0, 0), str(tick), font=font(17))
            d.text((xp(tick) - tb[2] / 2, bottom + 14), str(tick), font=font(17), fill="#444444")
    for sample, name in marks:
        x = xp((sample - start) / (seconds * 100.0) * 100.0) if seconds else xp((sample - start) / max(end - start, 1) * 100.0)
        for yy in range(top, bottom, 13):
            d.line((x, yy, x, min(yy + 7, bottom)), fill="#444444", width=2)
        d.text((x + 6, top - 26), f"{name} (sample {sample})", font=font(15, True), fill="#333333")
    d.text((right + 22, top - 34), "largest change", font=font(14, True), fill="#333333")
    d.text((right + 22, top - 16), "(% of walking range)", font=font(13), fill="#555555")
    for row, k in enumerate(LANES):
        middle = top + (row + 0.5) * lane
        reach = 0.40 * lane                    # a normalised value of 1 reaches this far from the lane's middle
        if row:
            d.line((left + 1, top + row * lane, right, top + row * lane), fill="#eeeeee", width=1)
        for x in range(left + 1, right, 14):   # quiet-standing level of this signal
            d.line((x, middle, x + 7, middle), fill="#bbbbbb", width=1)
        tb = d.textbbox((0, 0), CHANNEL_NAMES[k], font=font(16, True))
        d.text((left - 14 - tb[2], middle - 10), CHANNEL_NAMES[k], font=font(16, True), fill=CHANNEL_COLORS[k])
        label = "no range" if not alive[k] else (f"{fraction[k] * 100:+.0f} %" if changes[k] else f"{fraction[k] * 100:+.0f} % (flat)")
        if by_walking_range and alive[k] and abs(fraction[k]) > QS_LANE * 1.2:
            label += " (clipped)"
        d.text((right + 22, middle - 10), label, font=font(16), fill="#222222" if changes[k] else "#999999")
        values = np.clip(normalised[start:end + 1, k], -1.2, 1.2)
        points = [(xp(p), middle - float(v) * reach) for p, v in zip(percent, values)]
        flags = estimated[start:end + 1, k]
        for a, b in runs(flags):
            d.line(points[max(0, a - 1):min(len(points), b + 1)], fill=pale(CHANNEL_COLORS[k]), width=3)
        for a, b in runs(~flags):
            if b - a >= 2:
                d.line(points[a:b], fill=CHANNEL_COLORS[k], width=3)

    text = f"{phase} progress (%)   0 % = sample {start}, 100 % = sample {end}"
    if steps:
        text = "CoP valley samples: every step (one valley to the next) is drawn at the same width   |   tinted = stance (CoP valley to CoP peak)"
    elif seconds:
        text = f"Seconds after the twelfth CoP valley (v11 = sample {start})"
    tb = d.textbbox((0, 0), text, font=font(19, True))
    d.text(((left + right - tb[2]) / 2, bottom + 48), text, font=font(19, True), fill="#111111")
    label = Image.new("RGBA", (760, 30), (255, 255, 255, 0))
    axis_title = (f"Change from the pre-rise level, as a fraction of the walking range (lane = +/-{QS_LANE * 100:.0f} %)" if by_walking_range
                  else f"Change from quiet standing, each signal scaled to its largest change in {phase}")
    ImageDraw.Draw(label).text((0, 0), axis_title, font=font(19, True), fill="#111111")
    label = label.rotate(90, expand=True)
    image.paste(label, (14, int((top + bottom) / 2 - 380)), label)
    time_axis = ("every step has the same width, cut at its CoP valleys" if steps else
                 f"time is the {seconds:.0f} s after v11, in real time" if seconds else "time is 0-100 % of the phase")
    if by_walking_range:
        d.text((left, bottom + 92), f"* Same axes on every graph: {time_axis}; each signal is its change from its level just before the rise (dashed line), as a fraction of "
                                    f"its own walking range. A lane is +/-{QS_LANE * 100:.0f} %.", font=font(15, True), fill="#744600")
        d.text((left, bottom + 116), "Nothing is stretched to fill the lane here, so quiet standing looks as flat as it is. The numbers on the right are the largest "
                                     "deviation of each signal.", font=font(15), fill="#555555")
    else:
        d.text((left, bottom + 92), f"* Same axes on every graph: {time_axis}; each signal is its change from quiet standing (dashed line), scaled so its largest change "
                                    "fills its lane. Up = above quiet standing.", font=font(15, True), fill="#744600")
        d.text((left, bottom + 116), "The numbers on the right give the size that the scaling removes. A signal that changes by less than 2 % of its walking range, less than its "
                                     "quiet-standing noise, or less than three sensor steps is drawn flat.", font=font(15), fill="#555555")
    if notes:
        d.text((left, bottom + 140), ("NOTE: " + "; ".join(notes))[:235], font=font(15, True), fill="#9a4c00")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


QS_README = (
    "- **Time**: 0-100 % of quiet standing, from the start of the recording to the rise. The real sample range and duration are printed on each graph.\n"
    "- **Amplitude**: each signal is its change from its level just before the rise (median of the last 0.5 s; the dashed line), as a fraction of that "
    f"signal's own walking range (1st-99th percentile between the rise and the start of GT). A lane spans +/-{QS_LANE * 100:.0f} % of the walking range. "
    "Every signal has its own lane, in the order CoP, vGRF, Channel 1-8.\n"
    "- Unlike the other phases, nothing is stretched to fill its lane: stretching would turn sensor noise into curves. Quiet standing therefore looks as "
    "flat as it is, and a signal that does move (a weight shift, a drifting FMG channel) is visible at its true size.\n"
    "- **Size**: the number at the right of each lane is the largest deviation of that signal, in % of its walking range. A signal that leaves its lane is "
    "cut off at the lane edge and marked `(clipped)`. A dead channel is marked `no range`.\n"
    "- Nothing is smoothed. Estimated samples are drawn pale.\n\n")
GT_README = (
    f"- **Time**: the {GT_WINDOW_SECONDS:.0f} s after the twelfth CoP valley (v11), in real time, on every graph. The labelled GT phase runs on to the end "
    "of the recording (typically about 5 s after v11, up to 34 s, most of it standing), so the whole phase is not drawn. Where the recording ends earlier the graph stops there.\n"
    "- **Rest**: the dashed line marked `rest` is where the load under both feet stays steady (`rest_s` in `phase_cuts.csv`). It is found from both feet "
    "together, so it is the same sample on the left and the right graph of a trial. Where it falls outside the window the note on the graph says where it is.\n")


def process(phase: str):
    if phase not in PHASES:
        raise SystemExit(f"Unknown phase {phase}; choose from {', '.join(PHASES)}")
    following = PHASES[PHASES.index(phase) + 1] if phase != "GT" else None
    root = OUTPUT / phase
    if root.exists():
        shutil.rmtree(root)
    manifest, table = [], []
    made = 0
    for subject, (source_root, graphs_root) in sorted(SUBJECTS.items()):
        with (graphs_root / "phase_cuts.csv").open(encoding="utf-8", newline="") as f:
            cuts = list(csv.DictReader(f))
        stances, neighbours = {}, {}
        if phase in STEP_PHASES:   # the steps of this phase, exactly as cut by refine_sub08_h_valleys.py
            with (graphs_root / "stance_segments.csv").open(encoding="utf-8", newline="") as f:
                for entry in csv.DictReader(f):
                    if entry["phase"] == phase:
                        stances.setdefault((entry["record"], entry["trial"], entry["side"]), []).append(entry)
                    if entry["step"] in ("S5", "L1"):   # the steps on either side of the transition
                        neighbours[(entry["record"], entry["trial"], entry["side"], entry["step"])] = entry
        for cut in cuts:
            record, trial, side = cut["record"], int(cut["trial"]), cut["side"]
            row = {"subject": subject, "record": record, "trial": trial, "side": side, "status": "cutout" if cut["status"] == "cop_valley_cut" else "no_cutout"}
            if cut["status"] != "cop_valley_cut":
                row["reason"] = "graph is UNRESOLVED: " + cut.get("unresolved_reason", "")
                manifest.append(row)
                continue
            times, values, estimated = parse_csv(source_root / record / f"trial_{trial:02d}_{side.lower()}_segmented.csv")
            start = int(cut[f"{phase}_start_sample"])
            end = int(cut[f"{following}_start_sample"]) if following else len(times) - 1
            rise, gt_start = int(cut["GI_start_sample"]), int(cut["GT_start_sample"])
            seconds, marks, window_notes = 0.0, [], []
            if phase == "GT":   # the same real-time window on every graph, not the whole tail of the recording
                seconds = GT_WINDOW_SECONDS
                recorded_until = end
                end = min(recorded_until, start + int(seconds * 100))
                rest = int(np.argmin(np.abs(times - float(cut["rest_s"]))))   # where the load under both feet stays steady
                if rest <= start:
                    window_notes.append(f"foot load already steady (rest, sample {rest}) when this limb reaches v11")
                elif rest > end:
                    window_notes.append(f"rest reached after this window (sample {rest}, {(rest - start) / 100:.2f} s after v11)")
                else:
                    marks.append((rest, "rest"))
                if end - start < int(seconds * 100):
                    window_notes.append(f"recording ends {(end - start) / 100:.2f} s after v11")
            normalised, fraction, changes, alive = normalise(values, rise, gt_start, start, end, by_walking_range=phase == "QS")
            steps, step_rows = None, sorted(stances.get((record, str(trial), side), []), key=lambda entry: int(entry["cop_valley_sample"]))
            if phase in STEP_PHASES:
                steps = [(entry["step"], int(entry["cop_valley_sample"]), int(entry["cop_peak_sample"]), int(entry["next_valley_sample"])) for entry in step_rows]
                assert steps and steps[0][1] == start and steps[-1][3] == end, (subject, record, trial, side)
                assert all(a[3] == b[1] for a, b in zip(steps[:-1], steps[1:])), (subject, record, trial, side)
            notes = list(window_notes)
            for entry in step_rows:
                if entry["valley_kind"] != "cop_valley":
                    notes.append(f"{entry['step']} starts at: {entry['valley_kind'].replace('_', ' ')}")
                if entry["next_valley_kind"] != "cop_valley":
                    notes.append(f"{entry['step']} ends at: {entry['next_valley_kind'].replace('_', ' ')}")
            if (end - start) / 100 < SHORT_PHASE_SECONDS:
                notes.append(f"short {phase}: only {end - start + 1} samples")
            elif phase == "QS" and end - start < 100:
                notes.append(f"quiet standing lasts only {(end - start) / 100:.2f} s before the rise")
            onset = "GI start" if phase == "GI" else "QS end"
            if phase in ("GI", "QS") and cut.get("GI_start_kind") == "not_found_placed_before_first_step":
                notes.append(f"movement onset not found; {onset} is a placeholder")
            elif phase in ("GI", "QS") and cut.get("GI_start_kind") in ("local_departure_before_first_step", "individually_reviewed_onset"):
                notes.append(f"{onset}: " + cut["GI_start_kind"].replace("_", " "))
            if phase == "GT" and cut.get("GT_start_kind") != "cop_valley":
                notes.append("v11 is not a CoP valley: " + cut.get("GT_start_kind", "").replace("_", " "))
            if phase == "QS" and start < rise - BASELINE_SAMPLES and np.any(np.abs(fraction) > 0.25):
                notes.append("a signal drifts by more than 25 % of its walking range during quiet standing")
            if estimated[start:end + 1].any():
                notes.append("contains estimated samples")
            if not alive.all():
                notes.append("no walking range: " + ", ".join(CHANNEL_NAMES[k] for k in range(10) if not alive[k]))
            if cut.get("review_required") == "1":
                notes.append("source graph has a review note")
            context = ""
            if phase == "SLT":
                before, after = (neighbours.get((record, str(trial), side, step)) for step in ("S5", "L1"))
                if before and after:
                    context = (f"For comparison (not drawn): step before, S5 = {float(before['step_duration_s']):.2f} s with stance {float(before['stance_duration_s']):.2f} s"
                               f"   |   step after, L1 = {float(after['step_duration_s']):.2f} s with stance {float(after['stance_duration_s']):.2f} s")
            name = f"{record}_trial_{trial:02d}_{side.lower()}_{phase}.png"
            draw(root / subject / name, f"{subject} | {record} | Recording interval {trial:02d} | {side} | {phase} cutout", phase, start, end,
                 normalised, fraction, changes, alive, estimated, notes, steps, context, seconds, marks, phase == "QS")
            made += 1
            if phase == "GT":
                row.update({"rest_sample": rest, "rest_seconds_after_v11": round((rest - start) / 100, 2), "GT_start_kind": cut.get("GT_start_kind", "")})
            row.update({"start_sample": start, "end_sample": end, "duration_seconds": round((end - start) / 100, 2),
                        "baseline_samples": f"{max(0, rise - BASELINE_SAMPLES)}-{max(rise, 1) - 1}",
                        "GI_start_kind": cut.get("GI_start_kind", ""), "source_review_required": cut.get("review_required", ""),
                        **({f"{name}_{field}": value for name, valley, peak, following in steps
                            for field, value in (("valley_sample", valley), ("peak_sample", peak), ("next_valley_sample", following),
                                                 ("step_seconds", round((following - valley) / 100, 2)), ("stance_seconds", round((peak - valley) / 100, 2)))}
                           if steps else {}),
                        **({"S5_step_seconds": before["step_duration_s"], "S5_stance_seconds": before["stance_duration_s"],
                            "L1_step_seconds": after["step_duration_s"], "L1_stance_seconds": after["stance_duration_s"]} if context else {}),
                        "signals_drawn_flat": " ".join(VALUE_COLUMNS[k] for k in range(10) if not changes[k]),
                        **{f"{column}_largest_change_pct_of_walking_range": round(float(fraction[k]) * 100, 1) for k, column in enumerate(VALUE_COLUMNS)},
                        "notes": "; ".join(notes), "file": f"{subject}/{name}"})
            manifest.append(row)
            # Time-normalised copy of the normalised signals: 101 points (0..100 %) per phase, or per step in a walking phase.
            index = np.arange(len(times))
            for step_name, first, last in ([(name, valley, following) for name, valley, _, following in steps] if steps else [("", start, end)]):
                positions = np.arange(first, last + 1) if seconds else np.linspace(first, last, 101)   # GT keeps its real samples
                for percent, position in enumerate(positions):
                    entry = {"subject": subject, "record": record, "trial": trial, "side": side}
                    if steps:
                        entry.update({"step": step_name, "step_percent": percent})
                    elif seconds:
                        entry["seconds_after_v11"] = round(percent / 100.0, 2)
                    else:
                        entry[f"{phase.lower()}_percent"] = percent
                    for k, column in enumerate(VALUE_COLUMNS):
                        entry[column] = round(float(np.interp(position, index, normalised[:, k])), 5) if alive[k] else ""
                    table.append(entry)
    write_dicts(root / f"{phase.lower()}_manifest.csv", manifest)
    write_dicts(root / f"{phase.lower()}_normalised_values.csv", table)
    skipped = len(manifest) - made
    (root / "README.md").write_text(
        f"# {phase} cutouts\n\n"
        f"Generated by `make_phase_cutouts.py {phase}` from the CoP phase cuts in `tempgraphs/<subject>/phase_cuts.csv`.\n\n"
        f"{made} cutouts, one per cut limb graph, in one folder per subject. {skipped} graphs have no cutout because they are UNRESOLVED "
        f"(listed in `{phase.lower()}_manifest.csv`).\n\n"
        f"Each graph shows only {PHASE_TITLES[phase].lower()}: {PHASE_SPANS.get(phase, 'the phase span')}.\n\n"
        "## Normalisation\n\n"
        "Every graph uses the same axes, so they can be laid side by side.\n\n"
        + (QS_README if phase == "QS" else GT_README if phase == "GT" else "")
        + ("" if phase in ("QS", "GT") else "- **Time**: every step, from one CoP valley to the next, is drawn at the same width, so the step cuts are at the same place on every graph "
           "(SLT has a single step, which fills the graph). "
           "The axis is labelled with the real CoP valley samples, and the length and stance time of each step are printed above it. "
           "The tinted part of each step is its stance (CoP valley to CoP peak); the untinted part is swing.\n"
           if phase in STEP_PHASES else
           "- **Time**: 0-100 % of the phase. The real sample range and duration are printed on each graph.\n")
        + ("" if phase == "QS" else
        "- **Amplitude**: each signal is shown as its change from quiet standing (median of the last 0.5 s before the rise; the dashed line), "
        "scaled so that its largest change inside the phase fills its lane. Up means above the quiet-standing level. "
        "Every signal has its own lane, in the order CoP, vGRF, Channel 1-8.\n"
        "- **Size**: the scaling removes how big each change is. The number at the right of each lane gives it back: the largest change as a percentage "
        "of that signal's own walking range (1st-99th percentile between the rise and the start of GT). The same numbers are in the manifest.\n"
        "- A signal that changes by less than 2 % of its walking range, by less than four times its quiet-standing noise, or by less than three sensor steps "
        "is drawn flat and marked `(flat)`, so that noise is not blown up into a curve. A dead channel is marked `no range`.\n"
        "- Nothing is smoothed. Estimated samples are drawn pale.\n\n")
        + "## Things to know\n\n"
        + ("" if phase == "GT" else f"- {PHASE_TITLES[phase]} can be very short. Cutouts shorter than {SHORT_PHASE_SECONDS:.2f} s are marked on the graph and in the manifest; "
        "a window of a few samples cannot show much.\n")
        + "- The phase boundaries are signal-derived candidates. A cutout whose source graph carries a review note says so.\n"
        + (f"- The steps are the ones cut by `refine_sub08_h_valleys.py` (`stance_segments.csv`); nothing is re-detected here. "
           + {"SSSW": "S1 of the limb that steps first starts from quiet standing, not from a foot contact, and the note on the graph says so.\n",
              "SLT": "The transition step is the sixth step by count; a change in step length is not measured. For comparison each graph prints the length and "
                     "stance time of the step before it (S5) and after it (L1); only T is drawn.\n",
              "SSLW": "Where the foot of L5 lands straight into standing the CoP has no valley, so L5 ends at the foot-contact time; the note on the graph says so.\n"}[phase]
           + 
           f"- `{phase.lower()}_normalised_values.csv` holds the same normalised signals with every step resampled to 101 points (0..100 % of the step), for averaging.\n"
           if phase in STEP_PHASES else
           f"- `{phase.lower()}_normalised_values.csv` holds the same normalised signals at every sample of the window (`seconds_after_v11`), for averaging.\n"
           if phase == "GT" else
           f"- `{phase.lower()}_normalised_values.csv` holds the same normalised signals resampled to 101 points (0..100 %), for averaging.\n"),
        encoding="utf-8",
    )
    print(f"{phase}: {made} cutouts written to {root.relative_to(OUTPUT.parent)}; {skipped} unresolved graphs have none")


if __name__ == "__main__":
    for requested in (sys.argv[1:] or ["GI"]):
        process(requested.upper())
