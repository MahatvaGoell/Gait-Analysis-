"""CoP valley-to-peak (stance) segmentation for all subjects.

Everything is read from each limb's own CoP:
  valley = CoP minimum at foot contact (for the very first step, the low point
           the CoP climbs out of when the person leaves quiet standing);
  peak   = the highest CoP point of the stance, at push-off.
A step runs from one valley to the next; its stance phase is valley -> peak and
is what gets saved. The swing between a peak and the next valley is not saved.

Phase cuts, all on CoP valleys counted from the first one (v0):
  QS   recording start -> rise out of quiet standing.
  GI   rise -> v0.
  SSSW v0 -> v5   (5 steps: S1-S5).
  SLT  v5 -> v6   (1 transition step: T).
  SSLW v6 -> v11  (5 steps: L1-L5).
  GT   v11 -> end (the person comes to rest inside GT; further closing steps
                   belong to GT).
This layout reproduces the reviewer's marks on Sub08_H, sir_21, interval 04,
left (cuts at samples 1005 / 1119 / 1722).

A first CoP climb only counts as S1 when the limb then actually lifts off (its
load drops as in a normal swing). A weight shift made while still standing
stays in GI.

When the foot of L5 lands straight into standing the CoP has no valley there;
that one cut then uses the foot-contact time and is flagged.

Two detectors are used. The primary one is in this file. The second one
(`refine_subject03_04.py`, exported by `export_second_engine_trains.py`) finds
candidates independently; its candidates are re-placed onto the same valley and
peak convention, used as a cross-check everywhere, and used as the result where
the primary detector does not find eleven stances. For the amputee sets, whose
edges were reviewed one by one, the second detector is primary. A disagreement
between the two is reported on the graph and in `phase_cuts.csv`, never hidden.

Model-estimated samples (flagged *_imputed in the exports) are never presented
as measured: cuts on estimated CoP are flagged, and a limb whose walking CoP is
mostly estimated is left unresolved.

Outputs go into each subject's own tempgraphs folder: the graphs, one CSV per
valley-to-peak stance segment in stance_segments/, and summary tables. In the
subject's segmented_data only the phase label columns are updated; waveform
values are written back exactly as read.

Run:  python refine_sub08_h_valleys.py [subject ...]     (default: Sub08_H; "all" runs every subject)
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
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
    "Sub05_H": (SOURCE_ROOT / "sub05_h" / "segmented_data", SOURCE_ROOT / "sub05_h"),
    "Sub04_H": (SOURCE_ROOT / "sub04_h" / "segmented_data", SOURCE_ROOT / "sub04_h"),
    "Sub03_H": (SOURCE_ROOT / "sub03_h" / "segmented_data", SOURCE_ROOT / "sub03_h"),
    "Sub02_H": (SOURCE_ROOT / "sub02_h" / "segmented_data", SOURCE_ROOT / "sub02_h"),
    "Sub01_H": (SOURCE_ROOT / "sub01_h" / "segmented_data", SOURCE_ROOT / "sub01_h"),
    "Sub02_A": (SOURCE_ROOT / "sub02_a" / "segmented_data", SOURCE_ROOT / "sub02_a"),
    "Sub01_A": (SOURCE_ROOT / "sub01_a" / "segmented_data", SOURCE_ROOT / "sub01_a"),
}
# Amputee edges were reviewed one by one in the second detector: it is primary there.
SECOND_ENGINE_PRIMARY = {"Sub01_A", "Sub02_A"}
SECOND_ENGINE_TRAINS = Path(__file__).resolve().parent / "second_engine_trains.json"
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


LIFT_OFF_ABSENT = 0.35   # below this the foot did not leave the ground after the "stance"
LIFT_OFF_WEAK = 0.60     # below this the first step is kept but marked for review


def unload_ratio(load: np.ndarray, stances: list, onset: int) -> float:
    """How far the limb unloads after its first stance, relative to its next three swings (1 = a normal swing).

    Only neighbouring swings are compared: an insole can change its behaviour later in a recording.
    """
    if len(stances) < 5:
        return 1.0
    smoothed = smooth(load, 7)
    standing = float(np.median(load[max(0, onset - 100):max(onset, 20)]))
    drops = [standing - float(np.min(smoothed[p:max(p + 2, stances[k + 1][0])])) for k, (_, p, _) in enumerate(stances[:4])]
    typical = float(np.median(drops[1:4]))
    return drops[0] / typical if typical > 0 else 1.0


def drop_standing_shift(stances: list, load: np.ndarray, onset: int) -> tuple[list, int, float]:
    """Remove a first "stance" after which the foot never lifts: a weight shift while still standing.

    Returns the stances, how many were removed, and the lift-off ratio of the first stance that is kept.
    """
    dropped = 0
    while dropped < 2 and len(stances) > 4 and unload_ratio(load, stances, onset) < LIFT_OFF_ABSENT:
        stances = stances[1:]
        dropped += 1
    return stances, dropped, unload_ratio(load, stances, onset)


def place_valley(s: np.ndarray, fine: np.ndarray, load_slope: np.ndarray, candidate: int, previous_peak: int, climb: float) -> int:
    """Put a candidate from the second detector on this file's valley convention.

    The minimum of the detection-smoothed CoP inside the dip the candidate belongs to, refined to the
    lowest drawn sample within 0.08 s. Two cases where the stance starts at the end of the low instead:
    an exactly flat floor (CoP held constant while the foot is unloaded), and a long low that the CoP
    keeps through swing, recognised by the foot contact (steepest load rise) lying at its end.
    """
    lo = candidate
    floor = max(previous_peak + 1, candidate - 80)
    while lo > floor and fine[lo - 1] <= fine[candidate] + 0.35 * climb:
        lo -= 1
    centre = lo + int(np.argmin(s[lo:candidate + 4]))
    a = max(lo, centre - 8)
    valley = a + int(np.argmin(fine[a:centre + 9]))
    lowest = fine[valley]
    while valley + 1 <= candidate + 3 and fine[valley + 1] <= lowest + 1e-9:
        valley += 1
    if candidate - valley > 12:
        a, b = max(0, valley - 10), min(len(fine), candidate + 11)
        contact = a + int(np.argmax(load_slope[a:b]))
        if abs(candidate - contact) < abs(valley - contact):
            a = max(valley + 1, candidate - 8)
            valley = a + int(np.argmin(fine[a:candidate + 4]))
    return valley


def adopt_second_engine(train: list, cop: np.ndarray, load: np.ndarray) -> list[tuple[int, int, str]]:
    """Second-detector candidates, re-placed on this file's valley and peak convention."""
    s = smooth(medfilt(cop, 11), 5)
    fine = smooth(medfilt(cop, 5), 3)
    load_slope = np.gradient(smooth(load, 5))
    n = len(cop)
    placed = []
    for k, (valley, peak, kind) in enumerate(train):
        following = train[k + 1][0] if k + 1 < len(train) else min(n - 1, valley + 160)
        if kind != "filtered_cop_valley":   # first climb out of standing, or an individually reviewed edge
            placed.append((int(valley), int(peak), "second_engine_" + kind))
            continue
        previous_peak = placed[-1][1] if placed else max(0, valley - 80)
        valley = place_valley(s, fine, load_slope, valley, previous_peak, max(float(fine[peak] - fine[valley]), 1e-9))
        peak = climb_end(s, valley, min(following - 3, valley + 110), following - valley)
        start = max(valley + 5, peak - 6)
        placed.append((valley, start + int(np.argmax(fine[start:peak + 7])), "cop_valley"))
    placed = [x for k, x in enumerate(placed) if x[1] > x[0] and (k == 0 or x[0] > placed[k - 1][0])]
    # A "step" much shorter than a stride is a glitch dip inside a plateau: drop the weaker of the two valleys.
    changed = True
    while changed and len(placed) > 3:
        changed = False
        gaps = np.diff([v for v, _, _ in placed])
        stride = float(np.median(gaps))
        for k, gap in enumerate(gaps):
            if gap < 0.70 * stride:
                climbs = [fine[placed[j][1]] - fine[placed[j][0]] for j in (k, k + 1)]
                fixed = [placed[j][2].endswith("individually_reviewed_cop_edge") for j in (k, k + 1)]
                drop = k + 1 if fixed[0] and not fixed[1] else k if fixed[1] and not fixed[0] else (k if climbs[0] < climbs[1] else k + 1)
                del placed[drop]
                changed = True
                break
    return placed


def segment_trial(left: np.ndarray, right: np.ndarray, estimated_left: np.ndarray | None = None, estimated_right: np.ndarray | None = None,
                  second: dict | None = None, second_primary: bool = False):
    """Phase layout of both limbs. `second` holds the second detector's candidates for this recording."""
    lf, rf = finite_fill(left), finite_fill(right)
    filled = {"L": lf, "R": rf}
    flags = {"L": estimated_left if estimated_left is not None else np.zeros(lf.shape, bool),
             "R": estimated_right if estimated_right is not None else np.zeros(rf.shape, bool)}
    rise, rest = rise_and_rest(lf, rf)
    usable = second is not None and "error" not in second
    second_rise = second.get("rise") if usable else None
    onset = rise if rise > 0 else (second_rise or 0)   # the global detector can fail on a noisy quiet-standing stretch
    cop = {side: bridge_short_gaps(filled[side][:, 8], flags[side][:, 8]) for side in filled}
    result = {"rise": onset, "rest": rest}

    # Candidate stance trains of each detector, each with the lift-off test applied to its first stance.
    trains = {"own": {}, "second": {}}
    spans, dropped, lift_off = {}, {"own": {}, "second": {}}, {"own": {}, "second": {}}
    for side in ("L", "R"):
        load = filled[side][:, 9]
        stances, spans[side] = cop_stances(cop[side], rise, rest)
        trains["own"][side], dropped["own"][side], lift_off["own"][side] = drop_standing_shift(stances, load, onset)
        candidates = second[side]["stances"] if usable else []
        placed = adopt_second_engine(candidates, cop[side], load) if candidates else []
        trains["second"][side], dropped["second"][side], lift_off["second"][side] = drop_standing_shift(placed, load, onset)

    def movement_onset(name: str, side: str) -> tuple[int, str]:
        """Start of GI for this limb's layout."""
        valleys = [v for v, _, _ in trains[name][side]]
        stride = float(np.median(np.diff(valleys))) if len(valleys) > 2 else 120.0
        if name == "second" and usable and second[side].get("reviewed_onset") is not None:
            return second[side]["reviewed_onset"], "individually_reviewed_onset"
        if 0 < rise < valleys[0] and valleys[0] - rise <= 3 * stride:
            return rise, "rise_out_of_quiet_standing"
        if second_rise and 0 < second_rise < valleys[0] and valleys[0] - second_rise <= 3 * stride:
            return second_rise, "local_departure_before_first_step"
        return max(1, int(valleys[0] - 0.35 * stride)), "not_found_placed_before_first_step"

    def irregular(lay) -> int:
        """Steps whose length is far from this layout's median (a missed or an extra valley)."""
        lengths = np.array([following - valley for _, _, valley, _, _, following in lay[1]], float)
        typical = float(np.median(lengths))
        return int(np.sum(lengths[1:] > 1.5 * typical) + np.sum(lengths < 0.7 * typical))

    def end_cut(name: str, side: str, other: str):
        stances = trains[name][side]
        if len(stances) > len(STEP_PLAN):
            return stances[len(STEP_PLAN)][0], "cop_valley"
        if len(stances) < len(STEP_PLAN):
            return None
        info = second[side] if name == "second" and usable else None
        if info and info["resolved"] and info["end_cut"] is not None and len(info["stances"]) == len(stances) and not dropped[name][side]:
            return info["end_cut"], "second_engine_" + info["end_kind"]    # its own terminal contact
        partner = trains[name][other] if len(trains[name][other]) >= len(STEP_PLAN) else trains["own"][other]
        return landing_contact(cop[side], filled[side][:, 9], stances, partner, spans[side] or 1.0, flags[side][:, 9])

    order = ("second", "own") if second_primary else ("own", "second")
    layouts, onsets = {"L": {}, "R": {}}, {"L": {}, "R": {}}
    for side, other in (("L", "R"), ("R", "L")):
        for name in order:
            layouts[side][name], onsets[side][name] = None, ""
            if trains[name][side]:
                start, onsets[side][name] = movement_onset(name, side)
                layouts[side][name] = layout(trains[name][side], end_cut(name, side, other), start, len(lf))

    # Which detector's layout to use on each limb. Preference: the two feet alternate (their SLT cuts
    # are about half a stride apart), then regular step lengths, then the detector that is primary.
    options = {side: [name for name in order if layouts[side][name]] for side in ("L", "R")}
    picked = {side: (options[side][0] if options[side] else None) for side in ("L", "R")}
    if all(options.values()):
        def cost(left_name: str, right_name: str):
            pair = (layouts["L"][left_name], layouts["R"][right_name])
            stride = float(np.median([following - valley for lay in pair for _, _, valley, _, _, following in lay[1]]))
            apart = abs(pair[0][0][3] - pair[1][0][3])
            return (not 0.30 * stride <= apart <= 0.70 * stride, irregular(pair[0]) + irregular(pair[1]),
                    order.index(left_name) + order.index(right_name))
        picked["L"], picked["R"] = min(((a, b) for a in options["L"] for b in options["R"]), key=lambda names: cost(*names))
    else:
        for side in ("L", "R"):
            if len(options[side]) == 2 and irregular(layouts[side][options[side][0]]) > irregular(layouts[side][options[side][1]]):
                picked[side] = options[side][1]

    for side in ("L", "R"):
        chosen = picked[side]
        lay = layouts[side][chosen] if chosen else None
        layouts_here, onsets_here = layouts[side], onsets[side]
        found = max(len(trains[name][side]) for name in order)
        reason = "" if lay else ("fewer than 11 CoP valley-to-peak stances detected" if found < len(STEP_PLAN)
                                 else "candidate CoP stances do not form an ordered 5-1-5 layout")
        if second is None:
            cross_check = "not_run"
        elif all(layouts_here.values()):
            apart = abs(layouts_here["own"][0][3] - layouts_here["second"][0][3])
            cross_check = "detectors_agree" if apart <= 30 else "detectors_count_from_different_steps"
        elif lay:
            cross_check = "only_this_detector_found_11_stances"
        else:
            cross_check = "neither_detector_found_11_stances"
        estimated_cuts = []
        if lay:
            cop_flag = flags[side][:, 8]
            valleys = [step[2] for step in lay[1]] + [lay[0][5]]
            estimated_cuts = [f"v{k}" for k, v in enumerate(valleys) if near(cop_flag, v)]
            if len(estimated_cuts) >= 6:   # the CoP of this limb was not recorded while walking
                lay, reason = None, "CoP of this limb is estimated, not recorded, during walking"
        result[side] = {"stances": trains[chosen][side] if chosen else trains[order[0]][side], "layout": lay, "reason": reason,
                        "estimated_cuts": estimated_cuts, "cut_source": {"own": "primary_detector", "second": "second_detector_replaced"}.get(chosen, ""),
                        "cross_check": cross_check, "standing_shift_dropped": dropped[chosen][side] if chosen else 0,
                        "lift_off_ratio": round(float(lift_off[chosen][side]), 2) if chosen else None,
                        "onset_kind": onsets_here[chosen] if chosen else "",
                        "individual_review": second[side].get("individual_review", "") if chosen == "second" and second else ""}

    # The two feet alternate. If one limb's cuts are confirmed by both detectors and the other limb's
    # layout does not sit about half a stride away from them, that other layout counts the wrong steps
    # (typically a weak or noisy insole): leave it unresolved instead of showing misplaced cuts.
    if result["L"]["layout"] and result["R"]["layout"]:
        pair = (result["L"]["layout"], result["R"]["layout"])
        stride = float(np.median([following - valley for lay in pair for _, _, valley, _, _, following in lay[1]]))
        apart = abs(pair[0][0][3] - pair[1][0][3])
        confirmed = {side: result[side]["cross_check"] == "detectors_agree" for side in ("L", "R")}
        if not 0.30 * stride <= apart <= 0.70 * stride and confirmed["L"] != confirmed["R"]:
            weak = "R" if confirmed["L"] else "L"
            result[weak].update(layout=None, estimated_cuts=[], reason="its layout does not alternate with the other limb, whose cuts both "
                                "detectors confirm (weak or noisy insole signal)")
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


def review_flags(result: dict, left: np.ndarray, right: np.ndarray) -> None:
    """Review warnings; they never move a cut.

    A 5-1-5 count is not a validation. These checks surface layouts that deserve a
    look: the thresholds are review heuristics, not biomechanical acceptance limits.
    """
    for side, values in (("L", left), ("R", right)):
        seg = result[side]
        seg["review_notes"] = []
        if not seg["layout"]:
            continue
        if seg["cross_check"] == "detectors_count_from_different_steps":
            seg["review_notes"].append("the two detectors count from different first steps; S1 needs review")
        elif seg["cross_check"] == "only_this_detector_found_11_stances":
            seg["review_notes"].append("only one detector found eleven stances")
        if seg["estimated_cuts"]:
            seg["review_notes"].append("CoP valley cuts intersect estimated samples: " + ", ".join(seg["estimated_cuts"]))
        if seg["individual_review"]:
            seg["review_notes"].append("contains individually reviewed (hand-entered) edges")
        if seg["lift_off_ratio"] is not None and seg["lift_off_ratio"] < LIFT_OFF_WEAK:
            seg["review_notes"].append(f"limb barely unloads after S1 ({seg['lift_off_ratio']:.2f} of its next swings); S1 may be a weight shift")
        if seg["onset_kind"] == "not_found_placed_before_first_step":
            seg["review_notes"].append("movement onset not found; GI start is a placeholder")
        bounds, steps, _ = seg["layout"]
        cop = finite_fill(values)[:, 8]
        climbs = np.array([cop[p] - cop[v] for _, _, v, p, _, _ in steps])
        durations = np.array([nxt - v for _, _, v, _, _, nxt in steps])
        typical_climb = float(np.median(climbs))
        typical_duration = float(np.median(durations))
        if bounds[2] - bounds[1] > 1.5 * typical_duration:
            seg["review_notes"].append("GI spans more than 1.5 median CoP strides; movement onset needs review")
        for k, (phase, number, *_rest) in enumerate(steps):
            name = step_name(phase, number)
            if climbs[k] < 0.35 * typical_climb:
                seg["review_notes"].append(f"{name} CoP climb is weak relative to this trial")
            if durations[k] > 1.5 * typical_duration:
                seg["review_notes"].append(f"{name} interval is unusually long relative to this trial")
    if all(result[side]["layout"] for side in ("L", "R")):
        layouts = [result[side]["layout"] for side in ("L", "R")]
        stride = float(np.median([nxt - v for _, steps, _ in layouts for _, _, v, _, _, nxt in steps]))
        offset = abs(layouts[0][0][3] - layouts[1][0][3])
        if not 0.30 * stride <= offset <= 0.70 * stride:
            for side in ("L", "R"):
                result[side]["review_notes"].append(
                    f"L/R SLT offset {offset} samples is not near half this trial's CoP stride ({stride:.0f})"
                )


# Preserve the callable name introduced for Sub05.
sub05_review_flags = review_flags


def load_second_engine_trains() -> dict:
    if not SECOND_ENGINE_TRAINS.exists():
        return {}
    return json.loads(SECOND_ENGINE_TRAINS.read_text(encoding="utf-8"))


def segment_recording(subject: str, record: str, trial: int, left: np.ndarray, right: np.ndarray,
                      flags_left: np.ndarray, flags_right: np.ndarray, second_trains: dict | None = None) -> dict:
    """Segment one recording window, unless its trigger context does not support a protocol trial.

    Retained unpaired contexts are not verified protocol trials. Never select
    an arbitrary walking block inside them and call it a complete 5-1-5 trial.
    """
    window_method = ""
    windows = SUBJECTS[subject][1] / "recording_windows.csv"
    if windows.exists():
        with windows.open(encoding="utf-8", newline="") as f:
            window = next((r for r in csv.DictReader(f) if r["record"] == record and int(r["trial"]) == trial), None)
        window_method = window["boundary_method"] if window else ""
        if "unpaired" in window_method.lower():
            reason = "unpaired trigger context; protocol trial boundaries are unverified"
            return {"rise": 0, "rest": len(left) - 1, "source_window_method": window_method, "source_subject": subject,
                    **{side: {"stances": [], "layout": None, "reason": reason, "estimated_cuts": [], "cut_source": "", "cross_check": "not_run",
                              "standing_shift_dropped": 0, "individual_review": "", "lift_off_ratio": None, "onset_kind": ""} for side in ("L", "R")}}
    second = (second_trains or {}).get(f"{subject}|{record}|{trial}")
    result = segment_trial(left, right, flags_left, flags_right, second, subject in SECOND_ENGINE_PRIMARY)
    result["source_window_method"] = window_method
    result["source_subject"] = subject
    return result


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
    w, h = 2200, 760 + (28 if lay and seg.get("review_notes") else 0)
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
    tick_step = 200
    if xmax > 3000:  # keep full long contexts without overlapping tick labels
        target = xmax / 12
        magnitude = 10 ** math.floor(math.log10(target))
        tick_step = int(next(mult * magnitude for mult in (1, 2, 5, 10) if mult * magnitude >= target))
    for sample in range(0, xmax + 1, tick_step):
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
        if seg.get("review_notes"):
            d.text((left, 752), ("REVIEW: " + "; ".join(seg["review_notes"]))[:215], font=font(15, True), fill="#9a4c00")
    else:
        reason = seg.get("reason") or "fewer than 11 CoP valley-to-peak stances"
        d.text((left, 110), f"UNRESOLVED - {reason}, so 5 SSSW + SLT + 5 SSLW cannot be cut", font=font(21, True), fill="#9a4c00")
        d.text((left, 696), "* No phase cuts are drawn: the detector did not establish eleven ordered stance intervals for this limb.", font=font(16, True), fill="#744600")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save completely before replacing an existing graph. This also avoids a
    # truncate-open conflict with Windows preview/file-indexing readers.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as handle:
        pending = Path(handle.name)
        try:
            image.save(handle, format="PNG")
        except BaseException:
            handle.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


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
    second_trains = load_second_engine_trains()
    for left_path in sorted(source_root.rglob("trial_*_l_segmented.csv")):
        right_path = left_path.with_name(left_path.name.replace("_l_segmented.csv", "_r_segmented.csv"))
        if not right_path.exists():
            continue
        record = left_path.parent.name
        trial = int(left_path.name.split("_")[1])
        times, left, flags_left = parse_csv(left_path)
        times_r, right, flags_right = parse_csv(right_path)
        assert np.allclose(times, times_r, atol=1e-9)
        result = segment_recording(subject, record, trial, left, right, flags_left, flags_right, second_trains)
        review_flags(result, left, right)
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
            row["review_required"] = int(not lay or bool(seg["review_notes"]))
            row["review_notes"] = "; ".join(seg["review_notes"]) if lay else seg["reason"]
            row["cut_source"] = seg["cut_source"] if lay else ""
            row["detector_cross_check"] = seg["cross_check"]
            row["standing_shift_before_S1_left_in_GI"] = seg["standing_shift_dropped"]
            row["S1_lift_off_ratio"] = seg["lift_off_ratio"] if lay else ""
            row["GI_start_kind"] = seg["onset_kind"] if lay else ""
            if result.get("source_window_method"):
                row["source_window_method"] = result["source_window_method"]
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
            else:
                row["unresolved_reason"] = seg["reason"]
                labels = ["UNRESOLVED"] * len(times)   # no stale phase labels from an earlier segmentation
            # Only the phase label columns change; every waveform value is written back as read.
            status = "cop_valley_cut_not_ground_truth" if lay else "unresolved"
            if any(r["provisional_phase"] != label or r["label_status"] != status for r, label in zip(source_rows, labels)):
                for source_row, label in zip(source_rows, labels):
                    source_row["provisional_phase"] = label
                    source_row["label_status"] = status
                source_line_ending = line_ending(source)
                with source.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields, lineterminator=source_line_ending)
                    writer.writeheader()
                    writer.writerows(source_rows)
            cuts.append(row)
            png = graphs_root / record / f"trial_{trial:02d}_{side.lower()}_complete.png"
            draw(png, f"{subject} | {record} | Recording interval {trial:02d} | {side}", times, values, seg, flags if flagged else None)
            graphs += 1

    write_dicts(graphs_root / "phase_cuts.csv", cuts)
    write_dicts(graphs_root / "phase_intervals.csv", intervals)
    write_dicts(graphs_root / "stance_segments.csv", segment_rows)
    flagged = sum(bool(r["review_notes"]) for r in cuts if r["status"] == "cop_valley_cut")
    checks = {}
    for r in cuts:
        checks[r["detector_cross_check"]] = checks.get(r["detector_cross_check"], 0) + 1
    source_notes = {
        "Sub03_H": "`Amit_1` interval 01 is a 256.03-second unmatched-start context, not a verified protocol trial, and stays UNRESOLVED. "
                   "Ordinary paired intervals start at local interval 02. `Amit_21` has a missing left-insole end pulse; the original alignment caveats still apply.",
        "Sub04_H": "Noisy left triggers were not used to define windows; right-insole clock sections were aligned separately in the source export. "
                   "`ashu_1` interval 11 is a 77.54-second context without a measured end, so both limb graphs stay UNRESOLVED.",
        "Sub05_H": "The source synchronization report records a missing left-insole end trigger for `sub_21`, trial 16; its endpoint was inferred during the "
                   "original export and is not recovered here.",
        "Sub01_A": "Amputee recordings: no healthy-subject timing or bilateral symmetry is imposed. Edges that were reviewed one by one are listed in "
                   "`amputee_edge_reviews.json` (hand-entered sample indices) and are marked in `phase_cuts.csv`.",
        "Sub02_A": "Amputee recordings: no healthy-subject timing or bilateral symmetry is imposed. Edges that were reviewed one by one are listed in "
                   "`amputee_edge_reviews.json` (hand-entered sample indices) and are marked in `phase_cuts.csv`.",
    }
    (graphs_root / ("README.md" if subject == "Sub08_H" else "README_cop_segmentation.md")).write_text(
        f"# {subject} CoP valley-to-peak segmentation\n\n"
        "Generated by `refine_sub08_h_valleys.py`. Valleys and peaks are found in each limb's own CoP, not in vGRF.\n\n"
        f"{graphs} limb graphs: {resolved} cut, {graphs - resolved} UNRESOLVED, {flagged} of the cut graphs carry a review note.\n\n"
        "- **Valley**: CoP minimum at foot contact. For the limb that steps first, the first valley is the low point its CoP climbs out of when leaving quiet standing. "
        "Where the CoP is held exactly constant while the foot is unloaded, the valley is the last sample of that flat floor.\n"
        "- **Peak**: the highest CoP point of the stance, at push-off.\n"
        "- **Step**: one valley to the next. **Stance** = valley to peak (saved). The swing after the peak is not saved.\n\n"
        "Phase cuts are all on valleys, counted from the first one (v0):\n\n"
        "| Phase | Span | Steps |\n|---|---|---|\n"
        "| QS | start to the rise out of quiet standing | |\n"
        "| GI | rise to v0 | |\n"
        "| SSSW | v0 to v5 | S1-S5 |\n"
        "| SLT | v5 to v6 | T |\n"
        "| SSLW | v6 to v11 | L1-L5 |\n"
        "| GT | v11 to the end; the person comes to rest inside it, and any further closing step belongs to GT | |\n\n"
        "This is the layout that reproduces the reviewer's marks on Sub08_H, sir_21, interval 04, left (cuts at samples 1005 / 1119 / 1722). "
        "No other recording has reviewer marks, so every other cut is a signal-derived candidate.\n\n"
        "A first CoP climb counts as S1 only if the limb then lifts off. `S1_lift_off_ratio` is how far the limb's load drops after S1 "
        "relative to its next three swings: below 0.35 the climb is a weight shift made while still standing and stays in GI "
        "(`standing_shift_before_S1_left_in_GI`); between 0.35 and 0.60 S1 is kept and marked for review.\n\n"
        "When the foot of L5 lands straight into standing the CoP has no valley there. That one cut then uses the foot-contact time "
        "(`GT_start_kind` in `phase_cuts.csv`).\n\n"
        "## Cross-check between two detectors\n\n"
        "The primary detector is in `refine_sub08_h_valleys.py`. A second, independently written detector (`refine_subject03_04.py`, exported by "
        "`export_second_engine_trains.py`) proposes its own candidates; they are re-placed on the valley and peak convention above. "
        + ("For this subject the second detector is primary, because its edges were reviewed one by one. " if subject in SECOND_ENGINE_PRIMARY else "")
        + "`cut_source` says which detector supplied a graph's cuts and `detector_cross_check` how the two compare:\n\n"
        + "".join(f"- `{name}`: {count} graphs\n" for name, count in sorted(checks.items()))
        + "\n`not_run` means the second detector was not run for this subject. Where the two count from different first steps the graph is cut by the primary "
        "detector and marked for review; the disagreement is not resolved automatically.\n\n"
        "## Review notes\n\n"
        "`review_required` and `review_notes` in `phase_cuts.csv` (and a REVIEW line under the graph) mark layouts that deserve a look: detector disagreement, "
        "cuts on estimated CoP, hand-entered edges, a weak CoP climb (<35% of the trial median), an unusually long step (>1.5 times the trial median), "
        "GI longer than 1.5 median strides, or left/right SLT cuts that are not about half a stride apart. They are review heuristics and never move a cut. "
        "A matching 5-1-5 count or an unflagged graph is not a validation.\n\n"
        "Estimated samples (the `*_imputed` flags of the export, pale in the graphs) are never treated as measurements: "
        "`cuts_on_estimated_cop` names every cut that sits on estimated CoP, `stance_segments.csv` counts the estimated "
        "CoP / FMG / vGRF samples of each stance (`fully_measured` = 1 when there are none), and a limb whose walking CoP is mostly estimated "
        "is left UNRESOLVED.\n\n"
        "The graphs follow the reference figure: sample axis, phase names on lines, arrowed dashed cuts and S1-S5 / T / L1-L5 labels. "
        "They draw the exported values without smoothing; only a display offset and gain differ per signal. "
        "Valley and peak positions are not drawn; they are in `stance_segments.csv`.\n\n"
        + (f"## Source notes\n\n{source_notes[subject]}\n\n" if subject in source_notes else "")
        + "## Files\n\n"
        "- `<record>/trial_##_<l|r>_complete.png`: the graphs.\n"
        "- `stance_segments/<record>/trial_##_<l|r>_<PHASE>_<S1..L5>.csv`: the samples of one stance, valley to peak, values as exported.\n"
        "- `stance_segments.csv`: valley, peak and next-valley time and sample of every saved stance.\n"
        "- `phase_cuts.csv`, `phase_intervals.csv`: phase cuts per limb graph, in seconds and samples.\n\n"
        "Phase labels in the segmented_data exports (`provisional_phase`) cover the whole phase span, swing included, and match `phase_cuts.csv` for every cut graph. "
        "UNRESOLVED graphs get `provisional_phase` = UNRESOLVED. Graphs without a valid ordered stance layout are never forced.\n\n"
        "Older files in this folder (previous README, provisional tables, manifests, verification reports, review sheets) describe earlier segmentations, not these cuts. "
        "Legacy generation or resegmentation scripts can overwrite these outputs.\n\n"
        f"Reproduce: `python export_second_engine_trains.py` (once), then `python refine_sub08_h_valleys.py {subject}`.\n",
        encoding="utf-8",
    )
    summary = {"subject": subject, "graphs": graphs, "cut": resolved, "unresolved": graphs - resolved, "cut_graphs_with_review_note": flagged,
               "saved_stances": len(segment_rows), "detector_cross_check": checks,
               "validation": "Only Sub08_H sir_21 interval 04 left has reviewer marks; all other cuts are signal-derived candidates."}
    (graphs_root / "cop_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # The audit of the earlier smoothed-display run lists graph hashes that no longer exist.
    (graphs_root / "preprocessing_audit.json").unlink(missing_ok=True)
    print(f"{subject}: {graphs} limb graphs, {resolved} segmented, {graphs - resolved} unresolved, {len(segment_rows)} stance segments saved")


def main(argv: list[str]):
    names = argv or ["Sub08_H"]
    if [name.lower() for name in names] == ["all"]:
        names = list(SUBJECTS)
    lookup = {name.lower(): name for name in SUBJECTS}
    for name in names:
        if name.lower() not in lookup:
            raise SystemExit(f"Unknown subject {name}; choose from {', '.join(SUBJECTS)}")
    for name in names:
        process(lookup[name.lower()])


if __name__ == "__main__":
    main(sys.argv[1:])
