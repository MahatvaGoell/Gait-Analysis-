"""CoP valley-to-peak (stance) segmentation for Sub08_H, Sub07_H and Sub06_H.

Everything is read from each limb's own CoP:
  valley = CoP minimum at foot contact (for the very first step, the low point
           the CoP climbs out of when the person leaves quiet standing);
  peak   = where the CoP climb ends at push-off.
A step runs from one valley to the next; its stance phase is valley -> peak and
is what gets saved. The swing between a peak and the next valley is not saved.

Phase cuts, all on CoP valleys counted from the first one (v0):
  QS   recording start -> rise out of quiet standing.
  GI   rise -> v0.
  SSSW v0 -> v5   (5 steps: S1-S5).
  SLT  v5 -> v6   (1 transition step: T).
  SSLW v6 -> v11  (5 steps: L1-L5).
  GT   v11 -> end (the person comes to rest inside GT).

When the foot of the eleventh step lands straight into standing, the CoP has no
valley there; that one cut then uses the foot-contact time and is flagged.

Sub06_H and Sub07_H contain model-estimated samples (flagged *_imputed in their
exports). Events on estimated CoP are flagged, never presented as measured, and
a limb whose walking CoP is mostly estimated is left unresolved.

Outputs go into each subject's own tempgraphs folder: the graphs, one CSV per
valley-to-peak stance segment in stance_segments/, and summary tables. In the
subject's segmented_data only the phase label columns are updated; waveform
values are written back exactly as read.

Run:  python refine_sub08_h_valleys.py [Sub08_H] [Sub07_H] [Sub06_H]   (default: Sub08_H)
"""
from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import find_peaks, medfilt

from regenerate_tempgraphs import SOURCE_ROOT, font, finite_fill, moving_mean, parse_csv, runs


# subject -> (folder with <record>/trial_##_<l|r>_segmented.csv, folder that holds the graphs)
SUBJECTS = {
    "Sub08_H": (SOURCE_ROOT / "segmented_data" / "Sub08_H", SOURCE_ROOT / "Sub08_H"),
    "Sub07_H": (SOURCE_ROOT / "sub07_h" / "segmented_data", SOURCE_ROOT / "sub07_h"),
    "Sub06_H": (SOURCE_ROOT / "sub06_h" / "segmented_data", SOURCE_ROOT / "sub06_h"),
}
SOURCE = SUBJECTS["Sub08_H"][0]
PHASES = ("QS", "GI", "SSSW", "SLT", "SSLW", "GT")
STEP_PLAN = [("SSSW", n) for n in range(1, 6)] + [("SLT", 1)] + [("SSLW", n) for n in range(1, 6)]
CHANNEL_NAMES = [f"Channel {n}" for n in range(1, 9)] + ["CoP", "vGRF"]
CHANNEL_COLORS = ("#4e79a7", "#c27d3b", "#d4b640", "#9f77b0", "#64a343", "#42a6b2", "#bd627b", "#28758a", "#925a37", "#d4a333")
# Display offset, lowest/highest drawn value and largest gain per signal (same as the reference figure).
ANCHORS = (200, 170, 145, 120, 100, 80, 60, 40, 400, 380)
BANDS = [(10, 340, 1.0)] * 8 + [(365, 435, 2.5), (345, 425, .08)]
PHASE_TITLES = {"QS": "QS", "GI": "GI", "SSSW": "Steady state short step (SSSW)", "SLT": "SLT",
                "SSLW": "Steady state long step (SSLW)", "GT": "GT"}


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    return moving_mean(np.asarray(x, float)[:, None], width)[:, 0]


# ---------------------------------------------------------------- rise and rest

def activity(left: np.ndarray, right: np.ndarray, load_only: bool = False) -> np.ndarray:
    """Movement score: slope of each vGRF (and FMG-sum) signal relative to its walking range."""
    signals = (left[:, 9], right[:, 9]) if load_only else (left[:, 9], right[:, 9], left[:, :8].sum(1), right[:, :8].sum(1))
    scores = []
    for signal in signals:
        s = smooth(signal, 9)
        span = max(float(np.percentile(s, 95) - np.percentile(s, 5)), 1e-6)
        scores.append(smooth(np.abs(np.gradient(s)) / span, 25))
    return np.max(scores, axis=0)


def walking_block(score: np.ndarray) -> tuple[int, int, float]:
    high = float(np.percentile(score, 75))
    # A noisy insole can idle above a quarter of the walking level while the person still stands;
    # stay clear of the level seen before the first strong movement.
    strong = np.flatnonzero(score > 0.5 * high)
    idle = float(np.median(score[:strong[0]])) if strong.size and strong[0] > 20 else 0.0
    moving = score > max(0.25 * high, 2.5 * idle)
    for a, b in list(runs(~moving)):
        if a > 0 and b < len(moving) and b - a < 80:
            moving[a:b] = True
    a, b = max(runs(moving), key=lambda r: r[1] - r[0])
    return a, b, high


def rise_index(score: np.ndarray, start: int, high: float) -> int:
    """Walk back from the walking block to the last quiet-standing sample."""
    quiet = max(2.0 * float(np.median(score[:max(50, start - 100)])), 0.10 * high)
    i = start
    while i > 10 and not np.all(score[i - 10:i] < quiet):
        i -= 1
    return i


def rest_index(score: np.ndarray, end: int, high: float) -> int:
    """Walk forward from the walking block to where the person stays at rest."""
    quiet = max(2.0 * float(np.median(score[min(len(score) - 50, end + 100):])), 0.15 * high)
    i = end
    while i < len(score) - 30 and not np.all(score[i:i + 30] < quiet):
        i += 1
    return i


def rise_and_rest(lf: np.ndarray, rf: np.ndarray) -> tuple[int, int]:
    score = activity(lf, rf)
    a, b, high = walking_block(score)
    # Rest is judged on foot load alone: FMG keeps drifting slowly after the person has stopped.
    # Small weight shifts after stopping are ignored: search from the last strong load change.
    load = activity(lf, rf, load_only=True)
    load_high = float(np.percentile(load, 75))
    strong = np.flatnonzero(load[:b] > 0.5 * load_high)
    last_strong = int(strong[-1]) if strong.size else b
    return rise_index(score, a, high), min(rest_index(load, last_strong, load_high), len(lf) - 1)


# ---------------------------------------------------------------- CoP stances

def climb_end(s: np.ndarray, valley: int, limit: int, step: float = 0.0) -> int:
    """CoP peak of the stance that starts at `valley`.

    Normally this is simply the highest CoP point before the next valley. On
    some steps the CoP stays on a plateau through swing and creeps to its
    maximum late in that plateau; the peak is then where the climb itself ends.
    """
    limit = max(limit, valley + 6)
    top = valley + int(np.argmax(s[valley:limit + 1]))
    amplitude = s[top] - s[valley]
    if amplitude <= 0:
        return top
    if step and top - valley <= 0.65 * step:   # maximum reached within the stance part of the step
        return top
    i = valley
    while i < top and s[i] < s[valley] + 0.60 * amplitude:
        i += 1
    while i < top and s[min(top, i + 12)] - s[i] > 0.04 * amplitude:
        i += 1
    return i + int(np.argmax(s[i:min(top, i + 12) + 1]))


def cop_stances(cop: np.ndarray, rise: int, rest: int) -> tuple[list[tuple[int, int, str]], float]:
    """(valley, peak, kind) for every stance of this limb from the rise onwards, and the CoP span."""
    s = smooth(medfilt(cop, 11), 5)   # finds the events; removes insole glitches
    fine = smooth(medfilt(cop, 5), 3)  # places them on the curve as drawn
    window = s[rise:rest]
    span = float(np.percentile(window, 97) - np.percentile(window, 3))
    if span <= 0:
        return [], 0.0
    lo, hi = rise, min(len(s), rest + 20)
    found, properties = find_peaks(-s[lo:hi], prominence=0.25 * span, distance=40)
    valleys = [lo + int(v) for v in found]
    depth = list(properties["prominences"])

    def flat_low_end(valley: int) -> int:
        """Last sample of a flat low that starts at `valley` (the CoP held low through swing)."""
        end = valley
        while end + 1 < hi and s[end + 1] <= s[valley] + 0.08 * span:
            end += 1
        return end if end - valley > 15 else valley

    # Two valleys much closer than a stride: the shallower one is a glitch after push-off.
    merged = True
    while merged and len(valleys) > 3:
        merged = False
        stride = float(np.median(np.diff(valleys)))
        for k in range(len(valleys) - 1):
            if valleys[k + 1] - valleys[k] < 0.75 * stride:
                # Not a glitch if the second "valley" is the start of a flat low: the stance
                # begins where that low ends, a normal stride after the previous valley.
                end = flat_low_end(valleys[k + 1])
                if end - valleys[k] >= 0.75 * stride:
                    valleys[k + 1] = end
                    merged = True
                    break
                drop = k if depth[k] < depth[k + 1] else k + 1
                del valleys[drop], depth[drop]
                merged = True
                break

    def place(valley: int, peak: int, kind: str):
        valley = max(rise, valley - 8) + int(np.argmin(fine[max(rise, valley - 8):valley + 9]))
        start = max(valley + 5, peak - 6)
        return valley, start + int(np.argmax(fine[start:peak + 7])), kind

    stances = []
    stride = float(np.median(np.diff(valleys))) if len(valleys) > 2 else 0.0
    for k, valley in enumerate(valleys):
        following = valleys[k + 1] if k + 1 < len(valleys) else min(len(s) - 1, rest + 60)
        step = following - valley if k + 1 < len(valleys) else stride
        peak = climb_end(s, valley, min(following - 3, valley + 110), step)
        if s[peak] - s[valley] >= 0.30 * span:
            stances.append(place(valley, peak, "cop_valley"))
    if stances and stances[0][0] - rise > 45:
        # The limb that steps first climbs out of standing before its first foot contact.
        # That climb is its first stance; its valley is the low point the climb starts from.
        end = stances[0][0] - 20
        top = rise + int(np.argmax(s[rise:end]))
        lowest = rise + int(np.argmin(s[rise:top + 1]))
        amplitude = s[top] - s[lowest]
        if amplitude >= 0.30 * span and top - lowest >= 6:
            standing = float(np.median(s[max(0, rise - 50):rise + 1]))
            if standing - s[lowest] >= 0.10 * span:
                valley, kind = lowest, "cop_valley_before_first_climb"
            else:   # flat standing: the last point still at the standing level
                foot = np.flatnonzero(s[lowest:top + 1] <= s[lowest] + 0.10 * amplitude)
                valley, kind = lowest + int(foot[-1]), "foot_of_first_climb_from_standing"
            peak = climb_end(s, valley, top)
            first = place(valley, peak, kind) if kind == "cop_valley_before_first_climb" else (valley, peak, kind)
            # A real first step is at least 0.7 of a stride long. A shorter one is the other
            # foot's weight shift showing up in this limb's CoP while the person still stands.
            stride = float(np.median(np.diff([v for v, _, _ in stances]))) if len(stances) > 2 else 0.0
            if stances[0][0] - first[0] >= 0.70 * stride:
                stances.insert(0, first)
    if len(stances) > 3:
        # The first valley can be a weight-shift dip long before the foot is actually used
        # (first step far longer than a stride). The step then starts where the climb gets going.
        stride = float(np.median(np.diff([v for v, _, _ in stances[1:]])))
        valley, peak, kind = stances[0]
        if stances[1][0] - valley > 1.40 * stride:
            low = np.flatnonzero(s[valley:peak + 1] <= s[valley] + 0.25 * (s[peak] - s[valley]))
            stances[0] = (valley + int(low[-1]), peak, "start_of_first_climb")
    return stances, span


def landing_contact(cop: np.ndarray, vgrf: np.ndarray, mine: list, other: list, span: float, estimated: np.ndarray | None = None):
    """End of the eleventh step when the foot lands into standing and the CoP shows no valley.

    The time is the foot contact: the steepest load rise close to where the other
    limb's rhythm puts it, moved onto a CoP notch if there is one within 0.08 s.
    """
    valleys = [v for v, _, _ in mine]
    stride = float(np.median(np.diff(valleys)))
    others = [v for v, _, _ in other]
    if len(others) >= 12:
        expected = others[11] + float(np.median([a - b for a, b in zip(valleys[1:11], others[1:11])]))
    else:
        expected = valleys[10] + stride
    expected = int(round(expected))
    slope = np.gradient(smooth(vgrf, 5))
    def steepest(centre, reach):
        a, b = max(0, centre - reach), min(len(slope), centre + reach + 1)
        return a + int(np.argmax(slope[a:b]))
    regular = [steepest(v, 25) for v in valleys[1:11]]
    lead = int(round(float(np.median([v - c for v, c in zip(valleys[1:11], regular)]))))   # CoP valley follows the load rise
    typical = float(np.median([slope[c] for c in regular]))
    contact = steepest(expected, 25)
    measured = estimated is None or not estimated[max(0, expected - 25):expected + 26].any()
    if not measured or slope[contact] < 0.15 * typical:
        return expected, "landing_time_from_other_limb_rhythm"
    contact += lead
    fine = smooth(medfilt(cop, 5), 3)
    a, b = max(0, contact - 8), min(len(fine), contact + 9)
    notch = a + int(np.argmin(fine[a:b]))
    if a < notch < b - 1 and min(fine[a], fine[b - 1]) - fine[notch] >= 0.03 * span:
        return notch, "landing_cop_notch"
    return contact, "landing_foot_contact"


MIN_GI = 10   # samples; when the first valley coincides with the movement onset


def layout(stances: list[tuple[int, int, str]], end_cut: tuple[int, str] | None, rise: int, n: int):
    """v0..v11 -> SSSW (v0-v5), SLT (v5-v6), SSLW (v6-v11); the 11 stances start at v0..v10."""
    if len(stances) < len(STEP_PLAN) or end_cut is None:
        return None
    valleys = [v for v, _, _ in stances[:len(STEP_PLAN)]] + [end_cut[0]]
    steps = [(phase, number, valley, peak, kind, valleys[k + 1])
             for k, ((phase, number), (valley, peak, kind)) in enumerate(zip(STEP_PLAN, stances))]
    rise = min(rise, valleys[0] - MIN_GI)
    bounds = [0, rise, valleys[0], valleys[5], valleys[6], valleys[11], n]
    if any(b <= a for a, b in zip(bounds[:-1], bounds[1:])) or any(step[5] <= step[3] for step in steps):
        return None
    return bounds, steps, end_cut[1]


def bridge_short_gaps(signal: np.ndarray, estimated: np.ndarray, longest: int = 40) -> np.ndarray:
    """Replace short estimated stretches by a straight line between the measured samples around them.

    A model estimate can meet the measured signal with a small jump; over a short gap a
    straight line is as informative and cannot create a false valley.
    """
    out = np.asarray(signal, float).copy()
    for a, b in runs(estimated):
        if b - a <= longest and a > 0 and b < len(out):
            out[a:b] = np.interp(np.arange(a, b), [a - 1, b], [out[a - 1], out[b]])
    return out


def near(mask: np.ndarray, index: int, reach: int = 3) -> bool:
    return bool(mask[max(0, index - reach):index + reach + 1].any())


def segment_trial(left: np.ndarray, right: np.ndarray, estimated_left: np.ndarray | None = None, estimated_right: np.ndarray | None = None):
    lf, rf = finite_fill(left), finite_fill(right)
    filled = {"L": lf, "R": rf}
    flags = {"L": estimated_left if estimated_left is not None else np.zeros(lf.shape, bool),
             "R": estimated_right if estimated_right is not None else np.zeros(rf.shape, bool)}
    rise, rest = rise_and_rest(lf, rf)
    cop = {side: bridge_short_gaps(filled[side][:, 8], flags[side][:, 8]) for side in filled}
    found = {side: cop_stances(cop[side], rise, rest) for side in filled}
    result = {"rise": rise, "rest": rest}
    for side, other in (("L", "R"), ("R", "L")):
        stances, span = found[side]
        end_cut = None
        if len(stances) > len(STEP_PLAN):
            end_cut = (stances[len(STEP_PLAN)][0], "cop_valley")
        elif len(stances) == len(STEP_PLAN):
            end_cut = landing_contact(cop[side], filled[side][:, 9], stances, found[other][0], span, flags[side][:, 9])
        lay = layout(stances, end_cut, rise, len(lf))
        reason = "" if lay else "fewer than 11 CoP valley-to-peak stances"
        estimated_cuts = []
        if lay:
            cop_flag = flags[side][:, 8]
            valleys = [step[2] for step in lay[1]] + [lay[0][5]]
            estimated_cuts = [f"v{k}" for k, v in enumerate(valleys) if near(cop_flag, v)]
            if len(estimated_cuts) >= 6:   # the CoP of this limb was not recorded while walking
                lay, reason = None, "CoP of this limb is estimated, not recorded, during walking"
        result[side] = {"stances": stances, "layout": lay, "reason": reason, "estimated_cuts": estimated_cuts}
    return result


def check(lay) -> list[str]:
    """Hard checks on the rules the graphs must satisfy."""
    bounds, steps, _ = lay
    problems = []
    if [(p, k) for p, k, *_ in steps] != STEP_PLAN:
        problems.append("step plan")
    if any(not valley < peak < following for _, _, valley, peak, _, following in steps):
        problems.append("valley/peak/next valley out of order")
    if any(b[2] != a[5] for a, b in zip(steps[:-1], steps[1:])):
        problems.append("steps are not valley-to-valley")
    if bounds[2:6] != [steps[0][2], steps[5][2], steps[6][2], steps[10][5]]:
        problems.append("phase cut not on a step valley")
    return problems


def step_name(phase: str, number: int) -> str:
    return "T" if phase == "SLT" else f"{'S' if phase == 'SSSW' else 'L'}{number}"


# ---------------------------------------------------------------- rendering

def pale(color: str) -> tuple[int, int, int]:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(int(255 - (255 - c) * 0.35) for c in (r, g, b))


def draw(path: Path, title: str, times: np.ndarray, values: np.ndarray, seg: dict, estimated: np.ndarray | None = None):
    """Same layout as the reference figure: sample axis, phase names on lines, arrowed dashed cuts, S1-S5 / T / L1-L5."""
    lay = seg["layout"]
    z = np.asarray(values, float)
    filled = finite_fill(values)
    n = len(times)
    w, h = 2200, 760
    image = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(image)
    left, right, top, bottom = 138, 2022, 178, 600
    xmax = max(2200, math.ceil(n / 200) * 200)
    def xp(sample): return left + sample / xmax * (right - left)
    def yp(v): return bottom - v / 500 * (bottom - top)
    d.text((left, 20), title, font=font(23, True), fill="#111111")
    d.text((left, 57), "Original time retained | solid: source-grid values | pale: estimated values | phase cuts provisional", font=font(15), fill="#8a6d1f")

    # Axes: left and bottom lines with tick marks, no grid.
    d.line((left, top, left, bottom, right, bottom), fill="#555555", width=2)
    for y in range(0, 501, 100):
        d.line((left - 7, yp(y), left, yp(y)), fill="#555555", width=2)
        tb = d.textbbox((0, 0), str(y), font=font(17))
        d.text((left - 16 - tb[2], yp(y) - 10), str(y), font=font(17), fill="#444444")
    for sample in range(0, xmax + 1, 200):
        d.line((xp(sample), bottom, xp(sample), bottom + 7), fill="#555555", width=2)
        tb = d.textbbox((0, 0), str(sample), font=font(17))
        d.text((xp(sample) - tb[2] / 2, bottom + 14), str(sample), font=font(17), fill="#444444")
    text = "Samples @ 100 Hz (original trial timeline)"
    tb = d.textbbox((0, 0), text, font=font(19, True))
    d.text(((left + right - tb[2]) / 2, bottom + 48), text, font=font(19, True), fill="#111111")
    label = Image.new("RGBA", (460, 30), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((0, 0), "Display amplitude (scaled and offset)", font=font(19, True), fill="#111111")
    label = label.rotate(90, expand=True)
    image.paste(label, (40, int((top + bottom) / 2 - 230)), label)

    # Display-only offset and gain per signal. Solid = exported values, pale = gaps bridged for display.
    base_mask = times < .75
    for k in range(10):
        x = z[:, k]
        anchor = ANCHORS[k]
        low, high, max_gain = BANDS[k]
        base = float(np.nanmedian(x[base_mask])) if np.any(np.isfinite(x[base_mask])) else float(np.nanmedian(x))
        delta = x - base
        up, down = max(0, float(np.nanmax(delta))), max(0, float(-np.nanmin(delta)))
        gain = min(max_gain, (high - anchor) / max(up, 1e-9), (anchor - low) / max(down, 1e-9))
        display = anchor + (filled[:, k] - base) * gain
        points = [(xp(j), yp(float(display[j]))) for j in range(n)]
        observed = np.isfinite(x) if estimated is None else np.isfinite(x) & ~estimated[:, k]
        for a, b in runs(~observed):
            d.line(points[max(0, a - 1):min(n, b + 1)], fill=pale(CHANNEL_COLORS[k]), width=2)
        for a, b in runs(observed):
            if b - a >= 2:
                d.line(points[a:b], fill=CHANNEL_COLORS[k], width=2)
        yy = 196 + k * 28
        d.line((2052, yy + 9, 2084, yy + 9), fill=CHANNEL_COLORS[k], width=2)
        d.text((2096, yy), CHANNEL_NAMES[k] + ("" if observed.any() else " [est.]"), font=font(15), fill="#222222")

    if lay:
        bounds, steps, _ = lay
        for name, a, b in zip(PHASES, bounds[:-1], bounds[1:]):
            x0, x1 = xp(a), xp(min(b, n - 1))
            text = PHASE_TITLES[name] + "*"
            tb = d.textbbox((0, 0), text, font=font(19, True))
            centre = (x0 + x1) / 2
            if name in ("GI", "SLT"):   # short phases: name above, line under it
                d.text((centre - tb[2] / 2, 80), text, font=font(19, True), fill="#111111")
                half = max((x1 - x0) / 2 - 4, tb[2] / 2 + 2)
                d.line((centre - half, 110, centre + half, 110), fill="#222222", width=2)
            else:                       # long phases: line - name - line
                d.text((centre - tb[2] / 2, 141), text, font=font(19, True), fill="#111111")
                gap = tb[2] / 2 + 14
                if centre - gap > x0 + 8:
                    d.line((x0 + 6, 151, centre - gap, 151), fill="#222222", width=2)
                    d.line((centre + gap, 151, x1 - 6, 151), fill="#222222", width=2)
            if a > 0:   # arrowed dashed cut
                x = xp(a)
                for yy in range(top + 6, bottom, 15):
                    d.line((x, yy, x, min(yy + 8, bottom)), fill="#333333", width=2)
                d.polygon([(x, top - 7), (x - 5, top + 4), (x + 5, top + 4)], fill="#333333")
        # Step names under the vGRF, over the swing that closes each step.
        for phase, number, _, peak, _, following in steps:
            text = step_name(phase, number)
            tb = d.textbbox((0, 0), text, font=font(12))
            d.text((xp((peak + following) / 2) - tb[2] / 2, yp(331)), text, font=font(12), fill="#8a8a8a")
        d.text((left, 696), "* Phase cuts are on this limb's CoP valleys. Steps S1-S5, T and L1-L5 each run from one CoP valley to the next; the saved stance of a step is CoP valley to CoP peak.", font=font(16, True), fill="#744600")
        note = "signal-derived, not validated gait events"
        if seg.get("estimated_cuts"):
            note = "CUTS ON ESTIMATED (pale) CoP: " + ", ".join(seg["estimated_cuts"]) + " - not measured"
        d.text((left, 724), f"QS: quiet standing | GI: gait initiation | SSSW: 5 short steps | SLT: transition step | SSLW: 5 long steps | GT: gait termination | {note}", font=font(15), fill="#555555")
    else:
        reason = seg.get("reason") or "fewer than 11 CoP valley-to-peak stances"
        d.text((left, 110), f"UNRESOLVED - {reason}, so 5 SSSW + SLT + 5 SSLW cannot be cut", font=font(21, True), fill="#9a4c00")
        d.text((left, 696), "* No phase cuts are drawn for this limb: its CoP does not show eleven measured stances.", font=font(16, True), fill="#744600")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# ---------------------------------------------------------------- output

VALUE_COLUMNS = [f"FMG_channel_{n}" for n in range(1, 9)] + ["CoP", "vGRF"]


def write_dicts(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def line_ending(path: Path) -> str:
    with path.open("rb") as f:
        return "\r\n" if f.readline().endswith(b"\r\n") else "\n"


def process(subject: str):
    source_root, graphs_root = SUBJECTS[subject]
    segments_root = graphs_root / "stance_segments"
    if segments_root.exists():
        shutil.rmtree(segments_root)
    for stale in ("step_valleys.csv", "phase_boundaries_valley_cut.csv", "phase_intervals_valley_cut.csv"):
        (graphs_root / stale).unlink(missing_ok=True)
    cuts, intervals, segment_rows = [], [], []
    graphs = resolved = 0
    for left_path in sorted(source_root.rglob("trial_*_l_segmented.csv")):
        right_path = left_path.with_name(left_path.name.replace("_l_segmented.csv", "_r_segmented.csv"))
        if not right_path.exists():
            continue
        record = left_path.parent.name
        trial = int(left_path.name.split("_")[1])
        times, left, flags_left = parse_csv(left_path)
        times_r, right, flags_right = parse_csv(right_path)
        assert np.allclose(times, times_r, atol=1e-9)
        result = segment_trial(left, right, flags_left, flags_right)
        rest = result["rest"]
        for side, values, flags, source in (("L", left, flags_left, left_path), ("R", right, flags_right, right_path)):
            seg = result[side]
            lay = seg["layout"]
            with source.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields, source_rows = reader.fieldnames, list(reader)
            flagged = "CoP_imputed" in fields   # this export marks model-estimated samples
            row = {"record": record, "trial": trial, "side": side, "status": "cop_valley_cut" if lay else "unresolved",
                   "cop_stances_found": len(seg["stances"])}
            if lay:
                problems = check(lay)
                assert not problems, (subject, record, trial, side, problems)
                bounds, steps, end_kind = lay
                resolved += 1
                labels = np.empty(len(times), dtype="<U4")
                for name, a, b in zip(PHASES, bounds[:-1], bounds[1:]):
                    labels[a:b] = name
                    end = times[b] if b < len(times) else times[-1]
                    intervals.append({"record": record, "trial": trial, "side": side, "phase": name,
                                      "start_seconds": round(float(times[a]), 2), "end_seconds": round(float(end), 2),
                                      "duration_seconds": round(float(end - times[a]), 2), "start_sample": a, "end_sample": min(b, len(times) - 1)})
                    row[f"{name}_start_s"] = round(float(times[a]), 2)
                    row[f"{name}_start_sample"] = a
                row["first_valley_kind"] = steps[0][4]
                row["GT_start_kind"] = end_kind
                row["rest_s"] = round(float(times[rest]), 2)
                if flagged:
                    row["cuts_on_estimated_cop"] = " ".join(seg["estimated_cuts"])
                cop = finite_fill(values)[:, 8]
                # One file per stance: the samples from the CoP valley to the CoP peak, values exactly as exported.
                for phase, number, valley, peak, kind, following in steps:
                    name = f"trial_{trial:02d}_{side.lower()}_{phase}_{step_name(phase, number)}.csv"
                    target = segments_root / record / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f, lineterminator="\n")
                        writer.writerow(["time_seconds", "sample", "time_from_valley_seconds", *VALUE_COLUMNS,
                                         *(["FMG_estimated", "CoP_estimated", "vGRF_estimated"] if flagged else [])])
                        for index in range(valley, peak + 1):
                            extra = [int(flags[index, :8].any()), int(flags[index, 8]), int(flags[index, 9])] if flagged else []
                            writer.writerow([source_rows[index]["time_seconds"], index, f"{times[index] - times[valley]:.2f}",
                                             *[source_rows[index][c] for c in VALUE_COLUMNS], *extra])
                    entry = {"record": record, "trial": trial, "side": side, "phase": phase, "step": step_name(phase, number),
                             "cop_valley_s": round(float(times[valley]), 2), "cop_peak_s": round(float(times[peak]), 2),
                             "stance_duration_s": round(float(times[peak] - times[valley]), 2),
                             "cop_valley_sample": valley, "cop_peak_sample": peak, "samples": peak - valley + 1,
                             "next_valley_s": round(float(times[following]), 2), "next_valley_sample": following,
                             "step_duration_s": round(float(times[following] - times[valley]), 2),
                             "cop_at_valley": round(float(cop[valley]), 3), "cop_at_peak": round(float(cop[peak]), 3),
                             "valley_kind": kind, "next_valley_kind": end_kind if (phase, number) == STEP_PLAN[-1] else "cop_valley"}
                    if flagged:
                        stance = flags[valley:peak + 1]
                        entry.update({"cop_estimated_samples": int(stance[:, 8].sum()), "fmg_estimated_samples": int(stance[:, :8].any(axis=1).sum()),
                                      "vgrf_estimated_samples": int(stance[:, 9].sum()),
                                      "fully_measured": int(not stance.any())})
                    entry["file"] = f"stance_segments/{record}/{name}"
                    segment_rows.append(entry)
                # Only the phase label columns change; every waveform value is written back as read.
                for source_row, label in zip(source_rows, labels):
                    source_row["provisional_phase"] = label
                    source_row["label_status"] = "cop_valley_cut_not_ground_truth"
                source_line_ending = line_ending(source)
                with source.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields, lineterminator=source_line_ending)
                    writer.writeheader()
                    writer.writerows(source_rows)
            else:
                row["unresolved_reason"] = seg["reason"]
            cuts.append(row)
            png = graphs_root / record / f"trial_{trial:02d}_{side.lower()}_complete.png"
            draw(png, f"{subject} | {record} | Recording interval {trial:02d} | {side}", times, values, seg, flags if flagged else None)
            graphs += 1

    write_dicts(graphs_root / "phase_cuts.csv", cuts)
    write_dicts(graphs_root / "phase_intervals.csv", intervals)
    write_dicts(graphs_root / "stance_segments.csv", segment_rows)
    (graphs_root / ("README.md" if subject == "Sub08_H" else "README_cop_segmentation.md")).write_text(
        f"# {subject} CoP valley-to-peak segmentation\n\n"
        "Generated by `refine_sub08_h_valleys.py`. Valleys and peaks are found in each limb's own CoP, not in vGRF.\n\n"
        "- **Valley**: CoP minimum at foot contact. For the limb that steps first, the first valley is the low point its CoP climbs out of when leaving quiet standing.\n"
        "- **Peak**: the highest CoP point of the stance, at push-off.\n"
        "- **Step**: one valley to the next. **Stance** = valley to peak (saved). The swing after the peak is not saved.\n\n"
        "Phase cuts are all on valleys, counted from the first one (v0):\n\n"
        "| Phase | Span | Steps |\n|---|---|---|\n"
        "| QS | start to the rise out of quiet standing | |\n"
        "| GI | rise to v0 | |\n"
        "| SSSW | v0 to v5 | S1-S5 |\n"
        "| SLT | v5 to v6 | T |\n"
        "| SSLW | v6 to v11 | L1-L5 |\n"
        "| GT | v11 to the end; the person comes to rest inside it | |\n\n"
        "When the foot of L5 lands straight into standing the CoP has no valley there. That one cut then uses the foot-contact time "
        "(`GT_start_kind` in `phase_cuts.csv`).\n\n"
        "Estimated samples (the `*_imputed` flags of the export, pale in the graphs) are never treated as measurements: "
        "`cuts_on_estimated_cop` in `phase_cuts.csv` names every cut that sits on estimated CoP, `stance_segments.csv` counts the estimated "
        "CoP / FMG / vGRF samples of each stance (`fully_measured` = 1 when there are none), and a limb whose walking CoP is mostly estimated "
        "is left UNRESOLVED.\n\n"
        "The graphs follow the reference figure: sample axis, phase names on lines, arrowed dashed cuts and S1-S5 / T / L1-L5 labels. "
        "Valley and peak positions are not drawn; they are in `stance_segments.csv`.\n\n"
        "Files:\n\n"
        "- `<record>/trial_##_<l|r>_complete.png`: the graphs.\n"
        "- `stance_segments/<record>/trial_##_<l|r>_<PHASE>_<S1..L5>.csv`: the samples of one stance, valley to peak.\n"
        "- `stance_segments.csv`: valley, peak and next-valley time and sample of every saved stance.\n"
        "- `phase_cuts.csv`, `phase_intervals.csv`: phase cuts per limb graph, in seconds and samples.\n\n"
        "Phase labels in the segmented_data exports (`provisional_phase`) cover the whole phase span, swing included. "
        "Graphs without eleven measured stances are marked UNRESOLVED rather than forced.\n"
    )
    print(f"{subject}: {graphs} limb graphs, {resolved} segmented, {graphs - resolved} unresolved, {len(segment_rows)} stance segments saved")


def main(argv: list[str]):
    names = argv or ["Sub08_H"]
    lookup = {name.lower(): name for name in SUBJECTS}
    for name in names:
        if name.lower() not in lookup:
            raise SystemExit(f"Unknown subject {name}; choose from {', '.join(SUBJECTS)}")
    for name in names:
        process(lookup[name.lower()])


if __name__ == "__main__":
    main(sys.argv[1:])
