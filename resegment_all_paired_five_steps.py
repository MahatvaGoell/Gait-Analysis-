"""Final paired-trial, event-based segmentation for all tempgraphs outputs.

Within a simultaneous L/R pair, QS/GI and the first landing are shared trial
events. Each side uses five vGRF landings for SSSW, a valley-free transition
gap for SLT, and the next five landings for SSLW. Existing unresolved rows are
preserved rather than manufacturing missing steps.
"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tempgraphs"))

from resegment_all_subjects_five_steps import (
    GLOBAL_MANIFEST,
    complete_gt_window,
    destination,
    individual_fit,
    read_trial,
    review_sheets,
    set_boundary,
    sources,
    title,
    write_csv,
)
from resegment_sub08_five_steps import (
    candidate_valleys,
    phases_from_vgrf,
    recovery_peak,
    runs,
    valley_descent_onset,
)
from subject08_review import activity
from subjects01_to04_renderer import render


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def source_key(path: Path) -> tuple[str, int]:
    return path.parent.name, int(path.name.split("_")[1])


def pair_refit(times, values, common_onset):
    # Do not let the movement-onset deflection itself be counted as heel
    # landing.  The first landing must be a later, distinct vGRF valley.
    sample_period = float(abs(times[1] - times[0])) if len(times) > 1 else 0.01
    landing_search_start = common_onset + max(0.15, 4.0 * sample_period)
    errors = []
    successful = []
    for threshold in (0.06, 0.04, 0.025, 0.015, 0.008):
        try:
            phases, details = phases_from_vgrf(
                times, values[:, 9], min_relative_prominence=threshold,
                earliest_valley_seconds=landing_search_start,
                activity_onset_seconds=common_onset,
                allow_unfiltered_fallback=False,
                gt_duration_seconds=3.0,
                prefer_earliest_sequence=True,
                activity_search_lead_seconds=0.60,
                prefer_last_landing_crest=True,
            )
            valleys = [float(value) for value in details["valleys"]]
            gaps = [right - left for left, right in zip(valleys[:-1], valleys[1:])]
            mean_gap = sum(gaps) / len(gaps)
            gap_cv = (sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)) ** 0.5 / mean_gap
            successful.append((gap_cv, threshold, phases, details))
            # A coherent high-prominence train is preferred.  If it is
            # irregular, continue to lower prominence so a real but shallow
            # intervening landing is not skipped.
            if gap_cv <= 0.35:
                break
        except Exception as exc:
            errors.append(str(exc))
    if successful:
        gap_cv, threshold, phases, details = min(successful, key=lambda item: item[0])
        details["chosen_prominence"] = float(threshold)
        return (
            phases,
            details,
            f"paired-onset earliest own-vGRF sequence; prominence {threshold:.3f}; gap CV {gap_cv:.3f}",
        )
    raise ValueError(errors[-1] if errors else "no paired-onset eleven-valley train")


def walking_activity_end(times, values, first_valley_seconds):
    """End of the activity episode containing the selected walking train.

    Brief quiet gaps are joined because several recordings fragment one walk
    into separate activity bouts.  Distant later bouts are not joined.
    """
    combined = np.maximum(activity(values[:, :8]), activity(values[:, 8:]))
    moving = np.isfinite(combined) & (combined > 1.0)
    for start, end in runs(~moving):
        if start > 0 and end < len(moving) and end - start <= 35 and np.all(np.isfinite(combined[start:end])):
            moving[start:end] = True
    bouts = [(start, end) for start, end in runs(moving) if end - start >= 100]
    if not bouts:
        return float(times[-1])
    first_index = int(np.searchsorted(times, first_valley_seconds))
    selected = min(
        range(len(bouts)),
        key=lambda index: 0 if bouts[index][0] <= first_index <= bouts[index][1]
        else min(abs(first_index - bouts[index][0]), abs(first_index - bouts[index][1])),
    )
    end_index = bouts[selected][1]
    for next_start, next_end in bouts[selected + 1:]:
        gap = float(times[next_start] - times[min(end_index, len(times) - 1)])
        if gap > 3.5:
            break
        end_index = next_end
    return float(times[min(end_index, len(times) - 1)])


def rise_onset_before_first_landing(times, values, first_valley_seconds, fallback):
    """Beginning of the sustained sensor rise immediately before landing."""
    combined = np.maximum(activity(values[:, :8]), activity(values[:, 8:]))
    moving = np.isfinite(combined) & (combined > 1.0)
    for start, end in runs(~moving):
        if start > 0 and end < len(moving) and end - start <= 20 and np.all(np.isfinite(combined[start:end])):
            moving[start:end] = True
    first_index = int(np.searchsorted(times, first_valley_seconds))
    lower = int(np.searchsorted(times, max(float(times[0]), first_valley_seconds - 3.0)))
    candidates = []
    for start, end in runs(moving):
        if start < first_index and end >= first_index - 50 and end >= lower:
            candidates.append((start, end))
    if candidates:
        start, _ = min(candidates, key=lambda bounds: bounds[0])
        return float(times[max(lower, start - 15)])
    return max(float(times[lower]), min(float(fallback), first_valley_seconds - 0.15))


def first_sustained_activity_onset(times, values):
    """Earliest sustained movement after the initial standing samples."""
    combined = np.maximum(activity(values[:, :8]), activity(values[:, 8:]))
    moving = np.isfinite(combined) & (combined > 1.0)
    for start, end in runs(~moving):
        if start > 0 and end < len(moving) and end - start <= 35 and np.all(np.isfinite(combined[start:end])):
            moving[start:end] = True
    bouts = [(start, end) for start, end in runs(moving) if end - start >= 100]
    if not bouts:
        return None
    start, _ = bouts[0]
    return float(times[max(0, start - 15)])


def place_gt_after_final_walking_valley(times, values, phases, details):
    """Make two five-step phases separated by a valley-free transition.

    SSSW contains selected landings 1-5. SLT is only the recovery/descent gap
    between landing 5 and landing 6. SSLW contains selected landings 6-10.
    GT starts after recovery from the fifth SSLW landing, when deceleration
    and settling begin.
    """
    threshold = float(details.get("chosen_prominence", 0.06))
    candidates, meta = candidate_valleys(times, values[:, 9], threshold)
    signal = np.asarray(meta["smoothed"])
    original = [float(value) for value in details["valleys"]]
    first_index = int(np.argmin(np.abs(times - original[0])))
    original_indices = [int(np.argmin(np.abs(times - value))) for value in original]
    first_five = original_indices[:5]
    later_candidates = [
        index for index in candidates
        if float(times[index] - times[first_five[-1]]) >= 0.55
    ]
    collapsed_later = []
    for index in later_candidates:
        if collapsed_later and float(times[index] - times[collapsed_later[-1]]) < 0.55:
            if signal[index] < signal[collapsed_later[-1]]:
                collapsed_later[-1] = index
        else:
            collapsed_later.append(index)
    final_five = collapsed_later[:5]
    if len(final_five) < 5:
        final_five = original_indices[5:10]
    coherent = [*first_five, *final_five]
    # SLT is the step-free crest between the fifth short-step landing and the
    # first long-step landing.  Do not move its left edge to a small ripple:
    # doing that can collapse the visible transition to a single sample.
    sssw_end = recovery_peak(signal, first_five[-1], final_five[0])
    dt = float(meta["dt"])
    spread = float(meta["spread"])
    slt_end = valley_descent_onset(
        signal, sssw_end, final_five[0], dt, spread, prefer_last_crest=True
    )
    if slt_end <= sssw_end:
        slt_end = min(final_five[0] - 1, sssw_end + max(2, int(round(0.10 / max(dt, 1e-6)))))
    # Keep a clearly visible, valley-free transition even when recovery and
    # descent meet at a flat/sampled crest.
    min_slt_samples = max(2, int(round(0.20 / max(dt, 1e-6))))
    if slt_end - sssw_end < min_slt_samples:
        crest = int(np.argmax(signal[first_five[-1]:final_five[0] + 1])) + first_five[-1]
        half = max(1, min_slt_samples // 2)
        sssw_end = max(first_five[-1] + 1, crest - half)
        slt_end = min(final_five[0] - 1, sssw_end + min_slt_samples)
        if slt_end - sssw_end < min_slt_samples:
            sssw_end = max(first_five[-1] + 1, final_five[0] - 1 - min_slt_samples)
            slt_end = final_five[0] - 1

    # Start from the recovery after the fifth SSLW landing.  If one final,
    # immediate terminal heel strike follows that recovery, keep it on the
    # walking side of the GT boundary.  The bounded look-ahead is deliberate:
    # it must not attach a distant noisy/later activity bout to SSLW.
    later_after_ten = [index for index in collapsed_later if index > final_five[-1]]
    next_after_ten = later_after_ten[0] if later_after_ten else len(times)
    recovery_cap = int(np.searchsorted(times, float(times[final_five[-1]]) + 2.0, side="right"))
    base_gt_start = recovery_peak(signal, final_five[-1], min(next_after_ten, recovery_cap))
    terminal_candidates = [
        index for index in later_after_ten
        if 0.0 <= float(times[index] - times[base_gt_start]) <= 1.50
        and float(times[index] - times[final_five[-1]]) <= 2.50
    ]
    last_walking_valley = terminal_candidates[-1] if terminal_candidates else final_five[-1]
    later_after_last = [index for index in collapsed_later if index > last_walking_valley]
    next_after_last = later_after_last[0] if later_after_last else len(times)
    final_recovery_cap = int(np.searchsorted(times, float(times[last_walking_valley]) + 2.0, side="right"))
    gt_start = recovery_peak(signal, last_walking_valley, min(next_after_last, final_recovery_cap))
    set_boundary(phases[2], "end", sssw_end, times)
    set_boundary(phases[3], "start", sssw_end, times)
    set_boundary(phases[3], "end", slt_end, times)
    set_boundary(phases[4], "start", slt_end, times)
    set_boundary(phases[4], "end", gt_start, times)
    set_boundary(phases[5], "start", gt_start, times)
    details["valleys"] = [float(times[index]) for index in [*first_five, *final_five]]
    details["walking_valleys"] = [float(times[index]) for index in coherent]
    details["gt_end_index"] = min(
        len(times), gt_start + max(2, int(round(3.0 / max(float(meta["dt"]), 1e-6))))
    )
    return 0


def validate(phases, details):
    valleys = [float(v) for v in details["valleys"]]
    if len(valleys) != 10:
        raise ValueError(f"expected ten walking valleys; got {len(valleys)}")
    if not all(float(phases[2]["start_seconds"]) <= v < float(phases[2]["end_seconds"]) for v in valleys[:5]):
        raise ValueError("SSSW does not contain exactly selected V1-V5")
    if any(float(phases[3]["start_seconds"]) <= v < float(phases[3]["end_seconds"]) for v in valleys):
        raise ValueError("SLT contains a selected landing valley")
    if not all(float(phases[4]["start_seconds"]) < v < float(phases[4]["end_seconds"]) for v in valleys[5:]):
        raise ValueError("SSLW does not contain exactly selected V6-V10")
    return valleys


def process_subject(subject: str, source_root: Path, output: Path, apply: bool):
    manifest_path = output / "five_step_segmentation_manifest.csv"
    old_rows = load_rows(manifest_path)
    by_source = {row["source_csv"]: row for row in old_rows}
    groups = defaultdict(list)
    for path in sorted(source_root.rglob("*_segmented.csv")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if by_source.get(rel, {}).get("status") == "redrawn":
            groups[source_key(path)].append(path)

    updated = {}
    errors = []
    for (record, trial), paths in sorted(groups.items()):
        prepared = []
        for path in paths:
            try:
                times, values, missing = read_trial(path)
                base_phases, base_details, base_method, movement = individual_fit(times, values)
                early_movement = first_sustained_activity_onset(times, values)
                if early_movement is not None:
                    try:
                        early_phases, early_details, early_method = pair_refit(times, values, early_movement)
                        base_phases, base_details = early_phases, early_details
                        base_method = early_method + "; earliest sustained activity bout"
                        movement = early_movement
                    except Exception:
                        pass
                prepared.append([path, times, values, missing, base_phases, base_details, base_method, movement])
            except Exception as exc:
                errors.append(f"{subject}/{record}/{trial:02d}/{path.name}: {exc}")
        if not prepared:
            continue

        common_onset = min(
            rise_onset_before_first_landing(
                item[1], item[2], float(item[5]["valleys"][0]), float(item[4][1]["start_seconds"])
            )
            for item in prepared
        )
        fitted = []
        for item in prepared:
            path, times, values, missing, base_phases, base_details, base_method, movement = item
            try:
                phases, details, method = pair_refit(times, values, common_onset)
            except Exception:
                phases, details, method = base_phases, base_details, base_method + "; paired retry unavailable"
            place_gt_after_final_walking_valley(times, values, phases, details)
            method += "; SSSW first 5; SLT valley-free transition; SSLW next 5"
            fitted.append([path, times, values, missing, phases, details, method, movement])

        # When one simultaneously recorded side jumps to a much later bout
        # because its pressure sensor is missing/fragmented, use the reliable
        # earlier side's trial-level phase timestamps for both graphs.
        first_landings = [float(item[5]["valleys"][0]) for item in fitted]
        if len(fitted) > 1 and max(first_landings) - min(first_landings) > 3.0:
            leader_index = int(np.argmin(first_landings))
            leader_phases = fitted[leader_index][4]
            leader_details = fitted[leader_index][5]
            for index, item in enumerate(fitted):
                if index == leader_index:
                    continue
                item[4] = copy.deepcopy(leader_phases)
                item[5] = copy.deepcopy(leader_details)
                item[6] += "; paired earlier-side rescue for fragmented pressure data"

        # The earliest selected landing across the simultaneously recorded
        # feet ends global GI.  A later foot's own V1 still remains inside SSSW.
        common_landing = min(float(item[5]["valleys"][0]) for item in fitted)
        if common_landing <= common_onset:
            errors.append(f"{subject}/{record}/{trial:02d}: non-ordered paired GI")
            continue

        trial_rows = []
        trial_renders = []
        try:
            for path, times, values, missing, phases, details, method, movement in fitted:
                gi_index = int(abs(times - common_onset).argmin())
                landing_index = int(abs(times - common_landing).argmin())
                if not 0 < gi_index < landing_index < int(phases[2]["end_index"]):
                    raise ValueError(f"paired GI outside {path.name}")
                set_boundary(phases[0], "end", gi_index, times)
                set_boundary(phases[1], "start", gi_index, times)
                set_boundary(phases[1], "end", landing_index, times)
                set_boundary(phases[2], "start", landing_index, times)
                times, values, missing = complete_gt_window(times, values, missing, phases, details)
                valleys = validate(phases, details)
                target = destination(output, path)
                paired_method = method + "; shared L/R GI and first landing"
                rel = str(path.relative_to(ROOT)).replace("\\", "/")
                row = {
                    "subject": subject, "source_csv": rel,
                    "png": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "status": "redrawn", "segmentation_method": paired_method,
                    "movement_onset_seconds": "" if movement is None else f"{float(movement):.2f}",
                    "gi_start_seconds": f"{common_onset:.2f}",
                    "first_landing_seconds": f"{common_landing:.2f}",
                    "sssw_end_seconds": f"{float(phases[2]['end_seconds']):.2f}",
                    "slt_end_seconds": f"{float(phases[3]['end_seconds']):.2f}",
                    "gt_start_seconds": f"{float(phases[5]['start_seconds']):.2f}",
                    "gt_end_seconds": f"{float(phases[5]['end_seconds']):.2f}",
                    "gt_duration_seconds": f"{float(phases[5]['end_seconds']) - float(phases[5]['start_seconds']):.2f}",
                    "vgrf_valleys_seconds": ";".join(f"{v:.2f}" for v in valleys),
                }
                trial_rows.append((rel, row))
                trial_renders.append((path, target, times, values, missing, phases, details, paired_method))
        except Exception as exc:
            errors.append(f"{subject}/{record}/{trial:02d}: {exc}")
            continue

        if apply:
            for path, target, times, values, missing, phases, details, paired_method in trial_renders:
                end = int(details["gt_end_index"])
                render(
                    target, title(subject, path), times[: end + 1], values[: end + 1], [],
                    {"onset_seconds": common_onset, "view_mode": "model_completed"},
                    paired_method + ": SSSW five steps; SLT has no landing; SSLW next five steps.", phases,
                    "Trial-specific paired events; no fixed clock cuts. GT starts at post-SSLW deceleration/recovery.",
                    missing[: end + 1],
                )
        for rel, row in trial_rows:
            updated[rel] = row

    final_rows = [updated.get(row["source_csv"], row) for row in old_rows]
    if apply:
        write_csv(manifest_path, final_rows)
        review_sheets(output, final_rows)
    return final_rows, updated, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    all_final = []
    all_updated = {}
    all_errors = []
    for subject, source_root, output in sources():
        rows, updated, errors = process_subject(subject, source_root, output, args.apply)
        all_final.extend(rows)
        all_updated.update(updated)
        all_errors.extend(errors)
        print(f"{subject}: validated {len(updated)} existing graph rows; retained unresolved {sum(r['status'] != 'redrawn' for r in rows)}")
    if args.apply:
        global_rows = load_rows(GLOBAL_MANIFEST)
        global_rows = [all_updated.get(row["source_csv"], row) for row in global_rows]
        write_csv(GLOBAL_MANIFEST, global_rows)
    print(f"total validated {len(all_updated)}; pair errors {len(all_errors)}")
    for error in all_errors:
        print("PAIR_REVIEW", error)


if __name__ == "__main__":
    main()
