"""Event-based five-step segmentation for every subject trial.

This deliberately avoids shared clock cut-offs.  Each side of each recording
is fitted from its own vGRF valley train, with multi-sensor activity used only
to anchor gait initiation.  Outputs replace only the matching complete graph
and write auditable per-subject/global manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tempgraphs"))

from subjects01_to04_renderer import render
from resegment_sub08_five_steps import (
    earliest_landing_seconds,
    movement_onset_seconds,
    phases_from_vgrf,
)

SIGNALS = [*[f"FMG_channel_{i}" for i in range(1, 9)], "CoP", "vGRF"]
GLOBAL_MANIFEST = ROOT / "tempgraphs" / "all_subject_five_step_segmentation_manifest.csv"


def sources() -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    dataset_names = {p.name.lower(): p.name for p in (ROOT / "dataset").iterdir() if p.is_dir()}
    for output in sorted((ROOT / "tempgraphs").iterdir()):
        local = output / "segmented_data"
        if output.is_dir() and local.is_dir():
            label = dataset_names.get(output.name.lower(), output.name)
            found.append((label, local, output))
    special = ROOT / "tempgraphs" / "segmented_data" / "Sub08_H"
    if special.is_dir():
        found.append(("Sub08_H", special, ROOT / "tempgraphs" / "Sub08_H"))
    return found


def read_trial(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time_seconds"]) for row in rows], dtype=float)
    values = np.array([
        [float(row[name]) if row.get(name, "") not in ("", "nan", "NaN") else np.nan for name in SIGNALS]
        for row in rows
    ], dtype=float)
    missing = np.zeros_like(values, dtype=bool)
    for j, name in enumerate(SIGNALS):
        missing[:, j] = np.array([
            row.get(f"{name}_imputed", "0") not in ("", "0", "False", "false")
            or row.get("signal_missing", "0") not in ("", "0", "False", "false")
            or not np.isfinite(values[i, j])
            for i, row in enumerate(rows)
        ])
    return times, values, missing


def set_boundary(phase: dict[str, object], key: str, index: int, times: np.ndarray) -> None:
    phase[f"{key}_index"] = int(index)
    phase[f"{key}_seconds"] = float(times[index])


def individual_fit(times: np.ndarray, values: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object], str, float | None]:
    movement = movement_onset_seconds(times, values)
    earliest = earliest_landing_seconds(times, values[:, 9], movement)
    # A very large delay means the strong-drop gate skipped the real early
    # walking train (typically because a later artefact inflated vGRF range).
    # In that case keep the per-trial activity onset as the lower bound and
    # still require the full coherent eleven-valley sequence after it.
    if movement is not None and (earliest is None or earliest - movement > 4.0):
        earliest = movement
    errors: list[str] = []
    phases = details = None
    method = ""
    for threshold in (0.06, 0.04, 0.025, 0.015):
        try:
            phases, details = phases_from_vgrf(
                times, values[:, 9], min_relative_prominence=threshold,
                earliest_valley_seconds=earliest, activity_onset_seconds=movement,
                allow_unfiltered_fallback=False, gt_duration_seconds=3.0,
                prefer_earliest_sequence=True, activity_search_lead_seconds=0.60,
                prefer_last_landing_crest=True,
            )
            method = f"activity-supported own-vGRF sequence; prominence {threshold:.3f}"
            break
        except Exception as exc:
            errors.append(str(exc))
    if (
        phases is not None and details is not None and movement is not None
        and float(details["valleys"][0]) - movement > 4.0
    ):
        # The strong-drop gate can remove the actual first member of an
        # otherwise coherent early train, leaving only ten events and causing
        # the selector to jump to a much later artefact train. Retry from the
        # independently detected activity onset in this one recording.
        phases = details = None
        for threshold in (0.06, 0.04, 0.025, 0.015):
            try:
                phases, details = phases_from_vgrf(
                    times, values[:, 9], min_relative_prominence=threshold,
                    earliest_valley_seconds=movement, activity_onset_seconds=movement,
                    allow_unfiltered_fallback=False, gt_duration_seconds=3.0,
                    prefer_earliest_sequence=True, activity_search_lead_seconds=0.60,
                    prefer_last_landing_crest=True,
                )
                method = f"activity-onset own-vGRF retry; prominence {threshold:.3f}"
                break
            except Exception as exc:
                errors.append(str(exc))
    if phases is None or details is None:
        for threshold in (0.06, 0.04, 0.025, 0.015, 0.008):
            try:
                phases, details = phases_from_vgrf(
                    times, values[:, 9], min_relative_prominence=threshold,
                    gt_duration_seconds=3.0,
                )
                method = f"own-vGRF fallback; prominence {threshold:.3f}"
                break
            except Exception as exc:
                errors.append(str(exc))
    if phases is None or details is None:
        raise ValueError(errors[-1] if errors else "no valid eleven-valley sequence")

    # The first selected valley is the first landed step.  Place the second GI
    # marker on that event, not on a tiny neighbouring crest.  GI begins at
    # this trial's independently detected sustained multi-sensor activity.
    valley_times = [float(v) for v in details["valleys"]]
    first = int(np.argmin(np.abs(times - valley_times[0])))
    if movement is not None:
        gi_start = int(np.searchsorted(times, movement))
    else:
        gi_start = int(phases[1]["start_index"])
    if gi_start >= first:
        gi_start = int(phases[1]["start_index"])
    if gi_start >= first:
        gi_start = max(1, first - max(2, int(round(0.20 / np.median(np.diff(times))))))
    set_boundary(phases[0], "end", gi_start, times)
    set_boundary(phases[1], "start", gi_start, times)
    set_boundary(phases[1], "end", first, times)
    set_boundary(phases[2], "start", first, times)
    details["foot_rise_seconds"] = float(times[gi_start])
    details["first_landing_seconds"] = float(times[first])
    return phases, details, method, movement


def complete_gt_window(
    times: np.ndarray,
    values: np.ndarray,
    missing: np.ndarray,
    phases: list[dict[str, object]],
    details: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provide the requested three-second GT view without claiming measurements.

    Some acquisition windows stop less than three seconds after V11 recovery.
    Their remaining display tail is held at the final recorded sample and every
    added value is marked imputed, so the pale trace is visibly an estimate.
    """
    dt = float(np.median(np.diff(times)))
    gt_start = float(phases[5]["start_seconds"])
    wanted = gt_start + 3.0
    if times[-1] + dt / 2 < wanted:
        count = int(np.ceil((wanted - times[-1]) / dt))
        extra_times = times[-1] + dt * np.arange(1, count + 1)
        extra_values = np.repeat(values[-1:, :], count, axis=0)
        extra_missing = np.ones((count, missing.shape[1]), dtype=bool)
        times = np.concatenate([times, extra_times])
        values = np.vstack([values, extra_values])
        missing = np.vstack([missing, extra_missing])
    end = int(np.searchsorted(times, wanted, side="left"))
    end = min(end, len(times) - 1)
    phases[5]["end_index"] = end
    phases[5]["end_seconds"] = wanted
    details["gt_end_index"] = end
    return times, values, missing


def title(subject: str, source: Path) -> str:
    side = "L" if "_l_" in source.name.lower() else "R"
    trial = int(source.name.split("_")[1])
    return f"{subject} | {source.parent.name} | Recording interval {trial:02d} | {side}"


def destination(output: Path, source: Path) -> Path:
    return output / source.parent.name / source.name.replace("_segmented.csv", "_complete.png")


def process_subject(subject: str, source_root: Path, output: Path, apply: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sorted(source_root.rglob("*_segmented.csv")):
        target = destination(output, source)
        row: dict[str, object] = {
            "subject": subject,
            "source_csv": str(source.relative_to(ROOT)).replace("\\", "/"),
            "png": str(target.relative_to(ROOT)).replace("\\", "/"),
        }
        try:
            # Manual audit of this short Sub08 interval found only four/five
            # real landings.  Lowering prominence would turn ripples into
            # invented steps, so both sides remain explicitly unresolved.
            if subject == "Sub08_H" and source.parent.name == "sir_1" and source.name.startswith("trial_02_"):
                raise ValueError("recording contains fewer than eleven genuine landing valleys")
            times, values, missing = read_trial(source)
            phases, details, method, movement = individual_fit(times, values)
            times, values, missing = complete_gt_window(times, values, missing, phases, details)
            valleys = [float(v) for v in details["valleys"]]
            if len(valleys) != 11:
                raise ValueError(f"expected 11 valleys; got {len(valleys)}")
            if not all(phases[2]["start_seconds"] <= v < phases[2]["end_seconds"] for v in valleys[:5]):
                raise ValueError("SSSW does not contain V1-V5")
            if not phases[3]["start_seconds"] < valleys[5] < phases[3]["end_seconds"]:
                raise ValueError("SLT does not contain V6")
            if not all(phases[4]["start_seconds"] < v < phases[4]["end_seconds"] for v in valleys[6:]):
                raise ValueError("SSLW does not contain V7-V11")
            gt_duration = float(phases[5]["end_seconds"]) - float(phases[5]["start_seconds"])
            row.update(
                status="redrawn" if apply else "ready",
                segmentation_method=method,
                movement_onset_seconds="" if movement is None else f"{movement:.2f}",
                gi_start_seconds=f"{phases[1]['start_seconds']:.2f}",
                first_landing_seconds=f"{valleys[0]:.2f}",
                sssw_end_seconds=f"{phases[2]['end_seconds']:.2f}",
                slt_end_seconds=f"{phases[3]['end_seconds']:.2f}",
                gt_start_seconds=f"{phases[5]['start_seconds']:.2f}",
                gt_end_seconds=f"{phases[5]['end_seconds']:.2f}",
                gt_duration_seconds=f"{gt_duration:.2f}",
                vgrf_valleys_seconds=";".join(f"{v:.2f}" for v in valleys),
            )
            if apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                end = int(details["gt_end_index"])
                render(
                    target, title(subject, source), times[: end + 1], values[: end + 1], [],
                    {"onset_seconds": phases[1]["start_seconds"], "view_mode": "model_completed"},
                    method + ": V1-V5 SSSW, V6 SLT, V7-V11 SSLW.", phases,
                    "Individual CSV events; no shared subject timing. GT is 3.0 s after V11 recovery.",
                    missing[: end + 1],
                )
        except Exception as exc:
            row["status"] = f"not redrawn: {exc}"
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "subject", "source_csv", "png", "status", "segmentation_method",
        "movement_onset_seconds", "gi_start_seconds", "first_landing_seconds",
        "sssw_end_seconds", "slt_end_seconds", "gt_start_seconds",
        "gt_end_seconds", "gt_duration_seconds", "vgrf_valleys_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_sheets(output: Path, rows: list[dict[str, object]]) -> None:
    folder = output / "five_step_review_sheets"
    folder.mkdir(exist_ok=True)
    paths = [ROOT / str(r["png"]) for r in rows if r.get("status") == "redrawn"]
    for page, offset in enumerate(range(0, len(paths), 4), 1):
        sheet = Image.new("RGB", (1600, 640), "white")
        for pos, path in enumerate(paths[offset:offset + 4]):
            with Image.open(path) as source:
                source.thumbnail((800, 320))
                sheet.paste(source, ((pos % 2) * 800, (pos // 2) * 320))
        sheet.save(folder / f"page_{page:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    all_rows: list[dict[str, object]] = []
    for subject, source_root, output in sources():
        rows = process_subject(subject, source_root, output, args.apply)
        all_rows.extend(rows)
        if args.apply:
            write_csv(output / "five_step_segmentation_manifest.csv", rows)
            review_sheets(output, rows)
        ready = sum(r["status"] in ("ready", "redrawn") for r in rows)
        print(json.dumps({"subject": subject, "ready": ready, "unresolved": len(rows) - ready}), flush=True)
        for row in rows:
            if row["status"] not in ("ready", "redrawn"):
                print(f"UNRESOLVED {row['source_csv']} | {row['status']}", flush=True)
    if args.apply:
        write_csv(GLOBAL_MANIFEST, all_rows)
    print(json.dumps({"total": len(all_rows), "ready": sum(r["status"] in ("ready", "redrawn") for r in all_rows),
                      "unresolved": sum(r["status"] not in ("ready", "redrawn") for r in all_rows)}), flush=True)


if __name__ == "__main__":
    main()
