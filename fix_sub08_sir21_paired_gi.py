"""Synchronize GI within each Sub08_H/sir_21 left/right trial pair only."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from resegment_all_subjects_five_steps import (
    GLOBAL_MANIFEST,
    complete_gt_window,
    destination,
    individual_fit,
    read_trial,
    review_sheets,
    set_boundary,
    title,
    write_csv,
)
from subjects01_to04_renderer import render

SUBJECT = "Sub08_H"
SOURCE = ROOT / "tempgraphs" / "segmented_data" / SUBJECT / "sir_21"
OUTPUT = ROOT / "tempgraphs" / SUBJECT
SUBJECT_MANIFEST = OUTPUT / "five_step_segmentation_manifest.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def trial_number(path: Path) -> int:
    return int(path.name.split("_")[1])


def main() -> None:
    updated: dict[str, dict[str, object]] = {}
    sources = sorted(SOURCE.glob("*_segmented.csv"))
    by_trial: dict[int, list[Path]] = {}
    for source in sources:
        by_trial.setdefault(trial_number(source), []).append(source)

    for trial, pair in sorted(by_trial.items()):
        if len(pair) != 2:
            raise RuntimeError(f"sir_21 trial {trial:02d} does not have one L/R pair")
        prepared = []
        for source in pair:
            times, values, missing = read_trial(source)
            phases, details, method, movement = individual_fit(times, values)
            prepared.append([source, times, values, missing, phases, details, method, movement])

        # individual_fit already replaces a fragmented/late activity onset
        # with that side's validated vGRF fallback.  Use those fitted onsets
        # when constructing the shared trial-level GI marker.
        common_gi = min(float(item[4][1]["start_seconds"]) for item in prepared)
        common_landing = min(float(item[5]["valleys"][0]) for item in prepared)
        if common_landing <= common_gi:
            raise RuntimeError(f"sir_21 trial {trial:02d} has non-ordered paired GI events")

        for source, times, values, missing, phases, details, method, movement in prepared:
            gi_index = int(abs(times - common_gi).argmin())
            landing_index = int(abs(times - common_landing).argmin())
            if not 0 < gi_index < landing_index < int(phases[2]["end_index"]):
                raise RuntimeError(f"sir_21 trial {trial:02d} paired GI falls outside {source.name}")
            set_boundary(phases[0], "end", gi_index, times)
            set_boundary(phases[1], "start", gi_index, times)
            set_boundary(phases[1], "end", landing_index, times)
            set_boundary(phases[2], "start", landing_index, times)
            times, values, missing = complete_gt_window(times, values, missing, phases, details)

            target = destination(OUTPUT, source)
            end = int(details["gt_end_index"])
            paired_method = method + "; paired-trial GI anchor"
            render(
                target, title(SUBJECT, source), times[: end + 1], values[: end + 1], [],
                {"onset_seconds": common_gi, "view_mode": "model_completed"},
                paired_method + ": common L/R GI; own-side V1-V5, V6, V7-V11.",
                phases,
                "GI is shared by the simultaneous L/R pair; later phase valleys remain side-specific.",
                missing[: end + 1],
            )
            rel_source = str(source.relative_to(ROOT)).replace("\\", "/")
            valleys = [float(value) for value in details["valleys"]]
            updated[rel_source] = {
                "subject": SUBJECT,
                "source_csv": rel_source,
                "png": str(target.relative_to(ROOT)).replace("\\", "/"),
                "status": "redrawn",
                "segmentation_method": paired_method,
                "movement_onset_seconds": f"{float(movement):.2f}" if movement is not None else "",
                "gi_start_seconds": f"{common_gi:.2f}",
                "first_landing_seconds": f"{common_landing:.2f}",
                "sssw_end_seconds": f"{float(phases[2]['end_seconds']):.2f}",
                "slt_end_seconds": f"{float(phases[3]['end_seconds']):.2f}",
                "gt_start_seconds": f"{float(phases[5]['start_seconds']):.2f}",
                "gt_end_seconds": f"{float(phases[5]['end_seconds']):.2f}",
                "gt_duration_seconds": f"{float(phases[5]['end_seconds']) - float(phases[5]['start_seconds']):.2f}",
                "vgrf_valleys_seconds": ";".join(f"{value:.2f}" for value in valleys),
            }

    subject_rows = load_rows(SUBJECT_MANIFEST)
    for index, row in enumerate(subject_rows):
        if row["source_csv"] in updated:
            subject_rows[index] = updated[row["source_csv"]]
    write_csv(SUBJECT_MANIFEST, subject_rows)
    review_sheets(OUTPUT, subject_rows)

    global_rows = load_rows(GLOBAL_MANIFEST)
    for index, row in enumerate(global_rows):
        if row["source_csv"] in updated:
            global_rows[index] = updated[row["source_csv"]]
    write_csv(GLOBAL_MANIFEST, global_rows)
    print(f"updated {len(updated)} sir_21 graphs across {len(by_trial)} paired trials")


if __name__ == "__main__":
    main()
