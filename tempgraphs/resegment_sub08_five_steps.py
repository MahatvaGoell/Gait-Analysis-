"""Redraw only Sub08_H complete graphs using each graph's own vGRF valleys.

Segmentation rule requested for the project:
  GI starts at the sustained vGRF rise before walking.
  SSSW contains the first five complete vGRF valleys.
  SLT contains the next (transition) valley.
  SSLW contains the following five complete vGRF valleys.
  GT starts only after recovery from the fifth SSLW valley.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subjects01_to04_renderer import render
from subject08_review import activity


SOURCE = ROOT / "tempgraphs" / "segmented_data" / "Sub08_H"
OUTPUT = ROOT / "tempgraphs" / "Sub08_H"
MANIFEST = OUTPUT / "five_step_segmentation_manifest.csv"
GLOBAL_MANIFEST = ROOT / "tempgraphs" / "five_step_vgrf_graph_manifest.csv"
SIGNALS = [*[f"FMG_channel_{index}" for index in range(1, 9)], "CoP", "vGRF"]


def filled(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    good = np.isfinite(values)
    if not good.any():
        return np.zeros_like(values)
    if good.sum() == 1:
        values[~good] = values[good][0]
        return values
    indices = np.arange(len(values))
    values[~good] = np.interp(indices[~good], indices[good], values[good])
    return values


def smooth(values: np.ndarray, dt: float, seconds: float = 0.17) -> np.ndarray:
    width = max(5, int(round(seconds / max(dt, 1e-6))))
    if width % 2 == 0:
        width += 1
    return np.convolve(np.pad(values, width // 2, mode="edge"), np.ones(width) / width, mode="valid")


def extrema(values: np.ndarray, radius: int, minimum: bool) -> list[int]:
    found: list[int] = []
    for index in range(radius, len(values) - radius):
        section = values[index - radius : index + radius + 1]
        target = np.min(section) if minimum else np.max(section)
        if values[index] == target and (not found or index - found[-1] > radius):
            found.append(index)
    return found


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def read_trial(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time_seconds"]) for row in rows])
    values = np.array(
        [[float(row[name]) if row[name] not in ("", "nan", "NaN") else np.nan for name in SIGNALS] for row in rows]
    )
    row_missing = np.array([row.get("signal_missing", "0") not in ("", "0", "False", "false") for row in rows])
    imputed = np.repeat(row_missing[:, None], len(SIGNALS), axis=1) | ~np.isfinite(values)
    for j, name in enumerate(SIGNALS):
        imputed[:, j] |= np.array([row.get(f"{name}_imputed", "0") not in ("", "0", "False", "false") for row in rows])
    return times, values, imputed


def movement_onset_seconds(times: np.ndarray, values: np.ndarray) -> float | None:
    """Start of this recording's dominant sustained walking-activity bout."""
    fmg_score = activity(values[:, :8])
    pressure_score = activity(values[:, 8:])
    combined = np.maximum(fmg_score, pressure_score)
    moving = np.isfinite(combined) & (combined > 1.0)
    for start, end in runs(~moving):
        if start > 0 and end < len(moving) and end - start <= 35 and np.all(np.isfinite(combined[start:end])):
            moving[start:end] = True
    bouts = [(start, end) for start, end in runs(moving) if end - start >= 100]
    if not bouts:
        return None
    start, _ = max(bouts, key=lambda bounds: bounds[1] - bounds[0])
    return float(times[max(0, start - 15)])


def earliest_landing_seconds(times: np.ndarray, vgrf: np.ndarray, movement_onset: float | None) -> float | None:
    """First strong landed-step valley after this trial's movement onset."""
    if movement_onset is None:
        return None
    candidates, meta = candidate_valleys(times, vgrf, min_relative_prominence=0.06)
    signal = np.asarray(meta["smoothed"])
    dt = float(meta["dt"])
    lookback = max(2, int(round(1.60 / max(dt, 1e-6))))
    spread = float(meta["spread"])
    for index in candidates:
        if times[index] < movement_onset:
            continue
        preceding = signal[max(0, index - lookback) : index]
        if len(preceding) and float(np.max(preceding) - signal[index]) >= 0.43 * spread:
            return float(times[index] - dt)
    return movement_onset


def candidate_valleys(
    times: np.ndarray,
    values: np.ndarray,
    min_relative_prominence: float = 0.06,
) -> tuple[list[int], dict[str, object]]:
    dt = float(np.nanmedian(np.diff(times)))
    signal = smooth(filled(values), dt)
    initial = signal[(times >= 0.10) & (times <= min(1.0, float(times[-1])))]
    baseline = float(np.median(initial)) if len(initial) else float(np.median(signal))
    spread = max(float(np.percentile(signal, 95) - np.percentile(signal, 5)), 1e-9)
    radius = max(15, int(round(0.34 / max(dt, 1e-6))))
    prominence_radius = max(radius, int(round(0.45 / max(dt, 1e-6))))
    valleys: list[int] = []
    for index in extrema(signal, radius, minimum=True):
        before = signal[max(0, index - prominence_radius) : index]
        after = signal[index + 1 : min(len(signal), index + prominence_radius + 1)]
        if not len(before) or not len(after):
            continue
        depth = baseline - signal[index]
        prominence = min(float(np.max(before) - signal[index]), float(np.max(after) - signal[index]))
        if depth >= min_relative_prominence * spread and prominence >= min_relative_prominence * spread:
            valleys.append(index)
    return valleys, {
        "smoothed": signal,
        "baseline": baseline,
        "spread": spread,
        "dt": dt,
        "radius": radius,
    }


def select_eleven(
    times: np.ndarray,
    candidates: list[int],
    signal: np.ndarray,
    baseline: float,
    spread: float,
    prefer_earliest_start: bool = False,
) -> list[int]:
    """Choose the most coherent eleven-valley train without fixed phase times."""
    best: tuple[float, list[int]] | None = None
    # Dynamic paths skip duplicate/ripple candidates while preserving chronological events.
    paths: list[list[int]] = [[index] for index in candidates]
    for _ in range(10):
        extended: list[list[int]] = []
        for path in paths:
            previous = path[-1]
            for index in candidates:
                gap = float(times[index] - times[previous])
                if index > previous and 0.55 <= gap <= 2.80:
                    extended.append([*path, index])
        paths = extended
        if not paths:
            break
    if prefer_earliest_start and paths:
        earliest = min(path[0] for path in paths)
        paths = [path for path in paths if path[0] == earliest]
    for path in paths:
        gaps = np.diff(times[path])
        regularity = float(np.std(gaps) / max(np.mean(gaps), 1e-9))
        depths = np.mean(np.maximum(0.0, (baseline - signal[path]) / spread))
        # Prefer the complete walking train with regular cycles and strong valleys.
        score = regularity - 0.20 * float(depths) + 0.002 * float(times[path[0]])
        if best is None or score < best[0]:
            best = (score, path)
    if best is None:
        raise ValueError(f"no coherent eleven-valley train (detected {len(candidates)} significant valleys)")
    return best[1]


def recovery_peak(signal: np.ndarray, valley: int, stop: int) -> int:
    if stop <= valley + 2:
        return valley + 1
    return valley + 1 + int(np.argmax(signal[valley + 1 : stop]))


def valley_descent_onset(
    signal: np.ndarray,
    start: int,
    valley: int,
    dt: float,
    spread: float,
    prefer_last_crest: bool = False,
) -> int:
    """Return the immediate vGRF crest before the first landing valley.

    The search window begins at the preparatory unloading trough when one is
    available, so its dominant crest is the true start of the downward slope
    into the first landed-step valley.
    """
    section = signal[start : valley + 1]
    if len(section) < 2:
        return max(start + 1, valley - 1)
    radius = max(3, int(round(0.08 / max(dt, 1e-6))))
    maxima = [start + index for index in extrema(section, radius, minimum=False)]
    if maxima:
        meaningful = [index for index in maxima if signal[index] - signal[valley] >= 0.10 * spread]
        selected = meaningful[-1] if prefer_last_crest and meaningful else max(
            maxima, key=lambda index: float(signal[index] - signal[valley])
        )
        return max(start + 1, min(valley - 1, selected))
    peak = float(np.max(section))
    relative = int(np.flatnonzero(np.isclose(section, peak, rtol=0.0, atol=1e-9))[-1])
    return max(start + 1, min(valley - 1, start + relative))


def foot_rise_onset(
    signal: np.ndarray,
    search_start: int,
    landing_start: int,
    dt: float,
    spread: float,
) -> int:
    """Find the immediate local vGRF trough where the foot-rise upslope begins."""
    section = signal[search_start : landing_start + 1]
    if len(section) < 2:
        return max(search_start, landing_start - 1)
    radius = max(3, int(round(0.08 / max(dt, 1e-6))))
    minima = [search_start + index for index in extrema(section, radius, minimum=True)]
    meaningful = [index for index in minima if signal[landing_start] - signal[index] >= 0.08 * spread]
    if meaningful:
        return min(landing_start - 1, min(meaningful, key=lambda index: float(signal[index])))
    return min(landing_start - 1, search_start + int(np.argmin(section)))


def phases_from_vgrf(
    times: np.ndarray,
    vgrf: np.ndarray,
    min_relative_prominence: float = 0.06,
    earliest_valley_seconds: float | None = None,
    activity_onset_seconds: float | None = None,
    allow_unfiltered_fallback: bool = True,
    gt_duration_seconds: float | None = None,
    prefer_earliest_sequence: bool = False,
    activity_search_lead_seconds: float | None = None,
    prefer_last_landing_crest: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    all_candidates, meta = candidate_valleys(times, vgrf, min_relative_prominence)
    candidates = all_candidates
    if earliest_valley_seconds is not None:
        candidates = [index for index in candidates if times[index] >= earliest_valley_seconds]
    signal = np.asarray(meta["smoothed"])
    try:
        sequence = select_eleven(
            times, candidates, signal, float(meta["baseline"]), float(meta["spread"]), prefer_earliest_sequence
        )
    except ValueError:
        if candidates == all_candidates or not allow_unfiltered_fallback:
            raise
        candidates = all_candidates
        sequence = select_eleven(
            times, candidates, signal, float(meta["baseline"]), float(meta["spread"]), prefer_earliest_sequence
        )
    dt = float(meta["dt"])

    first = sequence[0]
    median_gap = int(round(float(np.median(np.diff(sequence)))))
    # Search only the immediately preceding step cycle.  A wider window can
    # select an arbitrary high sample from the quiet-standing plateau.
    pre_landing_start = max(1, first - max(3, int(round(1.10 * median_gap))))
    preparatory = []
    if activity_onset_seconds is not None:
        preparatory = [
            index
            for index in all_candidates
            if activity_onset_seconds <= times[index] < times[first]
            and times[first] - times[index] <= max(2.0, 1.80 * float(np.median(np.diff(times[sequence]))))
        ]
    if preparatory:
        pre_landing_start = preparatory[-1]
    elif activity_onset_seconds is not None and activity_search_lead_seconds is not None:
        activity_floor = int(np.searchsorted(times, activity_onset_seconds - activity_search_lead_seconds))
        pre_landing_start = max(pre_landing_start, max(1, activity_floor))
    spread = float(meta["spread"])
    sssw_start = valley_descent_onset(
        signal, pre_landing_start, first, dt, spread, prefer_last_landing_crest
    )
    gi_start = preparatory[-1] if preparatory else foot_rise_onset(signal, pre_landing_start, sssw_start, dt, spread)
    gi_start = max(1, min(gi_start, sssw_start - 1))

    sssw_end = recovery_peak(signal, sequence[4], sequence[5])
    slt_end = recovery_peak(signal, sequence[5], sequence[6])
    final_stop = min(len(times), sequence[10] + max(3, int(round(1.15 * median_gap))))
    gt_start = recovery_peak(signal, sequence[10], final_stop)
    gt_end = len(times)
    if gt_duration_seconds is not None:
        gt_end = min(len(times), gt_start + max(2, int(round(gt_duration_seconds / max(dt, 1e-6)))))
    boundaries = [0, gi_start, sssw_start, sssw_end, slt_end, gt_start, gt_end]
    if any(right <= left for left, right in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError(f"non-ordered boundaries: {boundaries}")
    names = ("QS", "GI", "SSSW", "SLT", "SSLW", "GT")
    phases = [
        {
            "phase": name,
            "start_index": int(start),
            "end_index": int(end),
            "start_seconds": float(times[start]),
            "end_seconds": float(times[end - 1]) if end == len(times) else float(times[end]),
        }
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:])
    ]
    return phases, {
        "valleys": [float(times[index]) for index in sequence],
        "first_valley_descent_seconds": float(times[sssw_start]),
        "candidate_count": len(candidates),
        "foot_rise_seconds": float(times[gi_start]),
        "gt_end_index": int(gt_end),
        "gt_available_seconds": float(times[-1] - times[gt_start]),
    }


def destination(source: Path) -> Path:
    return OUTPUT / source.parent.name / source.name.replace("_segmented.csv", "_complete.png")


def graph_title(source: Path) -> str:
    side = "L" if "_l_" in source.name else "R"
    trial = int(source.name.split("_")[1])
    return f"Sub08_H | {source.parent.name} | Recording interval {trial:02d} | {side}"


def process(apply: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for source in sorted(SOURCE.rglob("*_segmented.csv")):
        result: dict[str, object] = {
            "source_csv": str(source.relative_to(ROOT)).replace("\\", "/"),
            "png": str(destination(source).relative_to(ROOT)).replace("\\", "/"),
        }
        try:
            times, values, imputed = read_trial(source)
            movement_onset = movement_onset_seconds(times, values)
            earliest_landing = earliest_landing_seconds(times, values[:, 9], movement_onset)
            segmentation_method = "activity-supported own-vGRF sequence"
            try:
                phases, details = phases_from_vgrf(
                    times,
                    values[:, 9],
                    min_relative_prominence=0.06,
                    earliest_valley_seconds=earliest_landing,
                    activity_onset_seconds=movement_onset,
                    allow_unfiltered_fallback=False,
                    gt_duration_seconds=3.0,
                    prefer_earliest_sequence=True,
                    activity_search_lead_seconds=0.60,
                    prefer_last_landing_crest=True,
                )
            except ValueError:
                # Missing-data gaps can fragment the activity score while the
                # vGRF still contains a complete, coherent eleven-valley train.
                phases, details = phases_from_vgrf(
                    times,
                    values[:, 9],
                    min_relative_prominence=0.06,
                    gt_duration_seconds=3.0,
                )
                segmentation_method = "own-vGRF sequence; activity fragmented by missing data"
            valley_times = [float(value) for value in details["valleys"]]
            candidates, _ = candidate_valleys(times, values[:, 9], 0.06)
            extra = [float(times[index]) for index in candidates
                     if phases[2]['start_seconds'] < times[index] < phases[5]['end_seconds']
                     and all(abs(float(times[index]) - chosen) > 0.30 for chosen in valley_times)]
            result['review_issue'] = (
                'Additional vGRF valley candidates at ' + ', '.join(f'{v:.2f}s' for v in extra)
                + '; 5-1-5 selection is provisional.' if extra else ''
            )
            if not all(phases[2]["start_seconds"] < value < phases[2]["end_seconds"] for value in valley_times[:5]):
                raise ValueError("SSSW does not contain its five selected valleys")
            if not (phases[3]["start_seconds"] < valley_times[5] < phases[3]["end_seconds"]):
                raise ValueError("SLT transition valley is outside SLT")
            if not all(phases[4]["start_seconds"] < value < phases[4]["end_seconds"] for value in valley_times[6:]):
                raise ValueError("SSLW does not contain its five selected valleys")
            result.update(
                status="redrawn" if apply else "ready",
                segmentation_method=segmentation_method,
                movement_onset_seconds="" if movement_onset is None else f"{movement_onset:.2f}",
                gi_start_seconds=f"{phases[1]['start_seconds']:.2f}",
                sssw_start_seconds=f"{phases[2]['start_seconds']:.2f}",
                sssw_end_seconds=f"{phases[2]['end_seconds']:.2f}",
                slt_end_seconds=f"{phases[3]['end_seconds']:.2f}",
                gt_start_seconds=f"{phases[5]['start_seconds']:.2f}",
                gt_end_seconds=f"{phases[5]['end_seconds']:.2f}",
                gt_requested_duration_seconds="3.00",
                gt_available_seconds=f"{float(details['gt_available_seconds']):.2f}",
                vgrf_valleys_seconds=";".join(f"{value:.2f}" for value in details["valleys"]),
                candidate_count=details["candidate_count"],
            )
            if apply:
                info = {"onset_seconds": phases[1]["start_seconds"], "view_mode": "model_completed"}
                quality = f"{segmentation_method}: five SSSW valleys, one SLT valley, five SSLW valleys."
                note = "GI: rise to first descent | SSSW: valleys 1-5 | SLT: valley 6 | SSLW: valleys 7-11 | GT: after valley 11 recovery."
                if result['review_issue']:
                    note = 'REVIEW REQUIRED: ' + str(result['review_issue'])
                plot_end = int(details["gt_end_index"])
                render(
                    destination(source), graph_title(source), times[: plot_end + 1], values[: plot_end + 1], [],
                    info, quality, phases, note, imputed[: plot_end + 1],
                )
        except Exception as error:
            result["status"] = f"not redrawn: {error}"
        results.append(result)
    return results


def write_manifest(rows: list[dict[str, object]]) -> None:
    fields = [
        "source_csv", "png", "status", "segmentation_method", "movement_onset_seconds",
        "gi_start_seconds", "sssw_start_seconds", "sssw_end_seconds", "slt_end_seconds",
        "gt_start_seconds", "gt_end_seconds", "gt_requested_duration_seconds", "gt_available_seconds",
        "vgrf_valleys_seconds", "candidate_count", "review_issue",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_global_manifest(rows: list[dict[str, object]]) -> None:
    """Keep the earlier all-subject audit consistent with corrected Sub08 results."""
    if not GLOBAL_MANIFEST.exists():
        return
    by_source = {str(row["source_csv"]): row for row in rows}
    with GLOBAL_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        existing = list(csv.DictReader(stream))
    fields = list(existing[0]) if existing else []
    for row in existing:
        replacement = by_source.get(row.get("source_csv", ""))
        if replacement is None:
            continue
        for field in fields:
            if field in replacement:
                row[field] = str(replacement[field])
    with GLOBAL_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def write_review_sheets(rows: list[dict[str, object]]) -> None:
    review = OUTPUT / "review_sheets"
    review.mkdir(exist_ok=True)
    images = [ROOT / str(row["png"]) for row in rows if (ROOT / str(row["png"])).exists()]
    for page, offset in enumerate(range(0, len(images), 4), start=1):
        sheet = Image.new("RGB", (1600, 640), "white")
        for position, path in enumerate(images[offset : offset + 4]):
            with Image.open(path) as source:
                source.thumbnail((800, 320))
                sheet.paste(source, ((position % 2) * 800, (position // 2) * 320))
        target = review / f"page_{page:02d}.png"
        pending = target.with_suffix('.pending.png')
        sheet.save(pending)
        for attempt in range(5):
            try:
                pending.replace(target)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = process(args.apply)
    if args.apply:
        write_manifest(rows)
        update_global_manifest(rows)
        write_review_sheets(rows)
    good = sum(row["status"] in ("ready", "redrawn") for row in rows)
    print(json.dumps({"mode": "apply" if args.apply else "audit", "ready": good, "not_redrawn": len(rows) - good}, indent=2))
    for row in rows:
        if row["status"] not in ("ready", "redrawn"):
            print(row["source_csv"], row["status"])


if __name__ == "__main__":
    main()
