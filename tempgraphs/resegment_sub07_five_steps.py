"""Redraw only Sub07_H complete graphs with the validated per-graph gait rules."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tempgraphs"))
sys.path.insert(0, str(ROOT))

from resegment_sub08_five_steps import candidate_valleys, extrema, phases_from_vgrf, read_trial
from subject08_review import activity, runs
from subjects01_to04_renderer import render


SOURCE = ROOT / "tempgraphs" / "sub07_h" / "segmented_data"
OUTPUT = ROOT / "tempgraphs" / "sub07_h"
MANIFEST = OUTPUT / "five_step_segmentation_manifest.csv"
GLOBAL_MANIFEST = ROOT / "tempgraphs" / "five_step_vgrf_graph_manifest.csv"


def destination(source: Path) -> Path:
    return OUTPUT / source.parent.name / source.name.replace("_segmented.csv", "_complete.png")


def graph_title(source: Path) -> str:
    side = "L" if "_l_" in source.name else "R"
    trial = int(source.name.split("_")[1])
    return f"Sub07_H | {source.parent.name} | Recording interval {trial:02d} | {side}"


def movement_onset_seconds(times, values) -> float | None:
    """Start of the dominant sustained walking-activity bout."""
    score = activity(values[:, :8])
    pressure_score = activity(values[:, 8:])
    combined = np.maximum(score, pressure_score)
    moving = np.isfinite(combined) & (combined > 1.0)
    # Join only short pauses inside an active onset sequence.
    for start, end in runs(~moving):
        if start > 0 and end < len(moving) and end - start <= 35 and np.all(np.isfinite(combined[start:end])):
            moving[start:end] = True
    bouts = [(start, end) for start, end in runs(moving) if end - start >= 100]
    if not bouts:
        return None
    start, _ = max(bouts, key=lambda bounds: bounds[1] - bounds[0])
    return float(times[max(0, start - 15)])


def earliest_landing_seconds(times, vgrf, movement_onset: float | None) -> float | None:
    """Return the first strong landing valley after movement begins.

    Small early troughs are preparatory unloading/weight shifts.  A landed
    step must have a preceding vGRF crest-to-valley drop of at least 43% of
    the trial's robust vGRF range.  This retains shallow first landings while
    rejecting the smaller preparatory unloading troughs.
    """
    if movement_onset is None:
        return None
    candidates, meta = candidate_valleys(times, vgrf, min_relative_prominence=0.015)
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


def process(apply: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for source in sorted(SOURCE.rglob("*_segmented.csv")):
        result: dict[str, object] = {
            "source_csv": str(source.relative_to(ROOT)).replace("\\", "/"),
            "png": str(destination(source).relative_to(ROOT)).replace("\\", "/"),
        }
        try:
            times, values, imputed = read_trial(source)
            # Sub07's final heel-strike valleys are often shallower during
            # deceleration, so retain low-prominence valleys while the
            # eleven-event regularity test rejects standing noise.
            movement_onset = movement_onset_seconds(times, values)
            earliest_landing = earliest_landing_seconds(times, values[:, 9], movement_onset)
            phases, details = phases_from_vgrf(
                times,
                values[:, 9],
                min_relative_prominence=0.015,
                earliest_valley_seconds=earliest_landing,
                activity_onset_seconds=movement_onset,
                allow_unfiltered_fallback=False,
                gt_duration_seconds=3.0,
                prefer_earliest_sequence=True,
            )
            valley_times = [float(value) for value in details["valleys"]]
            if not (phases[2]["start_seconds"] < valley_times[0] < phases[2]["end_seconds"]):
                raise ValueError("first SSSW landing valley is outside SSSW")
            if not all(phases[2]["start_seconds"] < value < phases[2]["end_seconds"] for value in valley_times[:5]):
                raise ValueError("SSSW does not contain exactly its five selected valleys")
            if not (phases[3]["start_seconds"] < valley_times[5] < phases[3]["end_seconds"]):
                raise ValueError("SLT transition valley is outside SLT")
            if not all(phases[4]["start_seconds"] < value < phases[4]["end_seconds"] for value in valley_times[6:]):
                raise ValueError("SSLW does not contain exactly its five selected valleys")
            gt_available = float(details["gt_available_seconds"])
            result.update(
                status="redrawn" if apply else "ready",
                movement_onset_seconds="" if movement_onset is None else f"{movement_onset:.2f}",
                gi_start_seconds=f"{phases[1]['start_seconds']:.2f}",
                sssw_start_seconds=f"{phases[2]['start_seconds']:.2f}",
                sssw_end_seconds=f"{phases[2]['end_seconds']:.2f}",
                slt_end_seconds=f"{phases[3]['end_seconds']:.2f}",
                gt_start_seconds=f"{phases[5]['start_seconds']:.2f}",
                gt_end_seconds=f"{phases[5]['end_seconds']:.2f}",
                gt_requested_duration_seconds="3.00",
                gt_available_seconds=f"{gt_available:.2f}",
                vgrf_valleys_seconds=";".join(f"{value:.2f}" for value in details["valleys"]),
                candidate_count=details["candidate_count"],
            )
            if apply:
                info = {"onset_seconds": phases[1]["start_seconds"], "view_mode": "model_completed"}
                quality = "Own-vGRF five-step fit: five SSSW valleys, one SLT valley, five SSLW valleys."
                note = "GI: foot-rise upslope | SSSW: first landing + 5 valleys | GT: 3.0 s after fifth SSLW recovery."
                plot_end = int(details["gt_end_index"])
                render(
                    destination(source), graph_title(source), times[: plot_end + 1], values[: plot_end + 1], [],
                    info, quality, phases, note, imputed[: plot_end + 1],
                )
        except Exception as error:
            result["status"] = f"not redrawn: {error}"
        results.append(result)
    return results


FIELDS = [
    "source_csv", "png", "status", "movement_onset_seconds", "gi_start_seconds", "sssw_start_seconds",
    "sssw_end_seconds", "slt_end_seconds", "gt_start_seconds", "gt_end_seconds",
    "gt_requested_duration_seconds", "gt_available_seconds", "vgrf_valleys_seconds", "candidate_count",
]


def write_manifest(rows: list[dict[str, object]]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_global_manifest(rows: list[dict[str, object]]) -> None:
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
    """Rebuild the existing Sub07 visual-audit sheets from the current PNGs."""
    review = OUTPUT / "review_sheets"
    review.mkdir(exist_ok=True)
    images = [ROOT / str(row["png"]) for row in rows if (ROOT / str(row["png"])).exists()]
    for page, offset in enumerate(range(0, len(images), 4), start=1):
        sheet = Image.new("RGB", (1600, 640), "white")
        for position, path in enumerate(images[offset : offset + 4]):
            with Image.open(path) as source:
                source.thumbnail((800, 320))
                sheet.paste(source, ((position % 2) * 800, (position // 2) * 320))
        sheet.save(review / f"page_{page:02d}.png")


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
