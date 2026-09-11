# Subject 08: provisional six-phase segmentation

## What was produced

The current PNGs under `Sub08_H/sir_1` and `Sub08_H/sir_21` contain the requested sequence QS -> GI -> SSSW -> SLT -> SSLW -> GT, fitted separately to each recorded interval where a complete fit is possible. Left and right plots use the same **trial-specific whole-body boundaries**, estimated using both insoles and all 16 FMG channels. Their signal traces remain separate. No timing is shared across different trials.

There are 41 trigger-pair intervals and 82 limb plots. Of these, 39 intervals (78 plots) receive provisional six-phase fits; 2 intervals (4 plots) are explicitly unresolved. Of the 39 transition fits, 13 have moderate pattern evidence under the implemented stability checks and 26 have weak/ambiguous evidence. These categories are computational diagnostics, not validated accuracy or probability estimates.

**All phase labels are provisional. The short-to-long order is the user's requested assumption, not a measured result. Do not treat these files as validated ground-truth labels for classifier training or performance evaluation.** The planning document contains both walking directions; this fit does not establish which direction actually occurred in each interval.

## Files

- `Sub08_H/.../*_complete.png`: full-duration graphs, phase bars, actual boundary times, and uncertainty warnings.
- `phase_boundaries_provisional.csv`: one row per recorded interval, with fitted boundaries, supporting diagnostics, and flags.
- `phase_intervals_provisional.csv`: start/end times and indices for every fitted phase. Index intervals are half-open `[start_index, end_index)`; the final time is the last recorded grid time.
- `segmented_data/Sub08_H/.../*_segmented.csv`: one 100 Hz, timestamp-aligned numeric file per limb/interval. Contains the unscaled interpolated signals, provisional phase, label status, and a missing-signal flag. Blank values are missing, not zero. Display offsets are not included in these numeric values.
- `load_landmarks_provisional.csv`: accepted recurring pressure-peak candidates. **These are not heel-strike annotations.**
- `display_transforms.csv`: exact positive gains, offsets and baselines used for plotting; display = offset + gain * (value - baseline).
- `complete_trial_manifest.csv`: one row per plot linking its numeric data and quality checks.
- `segmentation_review_sheets`: current 2x2 visual contact sheets.
- `_before_sequence_segmentation`: recoverable copies of the preceding 82 plots and associated manifests. The old `review_sheets` folder also shows the preceding, unsegmented version.
- `segmentation_provenance.json`: source hashes and generation-script hash.

## How the provisional boundaries are estimated

1. Each source stream retains elapsed time relative to its matching trial-start trigger. Missing intervals over 0.10 s remain gaps; repeated timestamps retain the first packet. Nothing is stretched to the reference's duration. Full recordings are plotted.
2. Repeated prominent pressure-load peaks are detected separately for both feet using a short detection-only average. The original plotted signals are not smoothed. The algorithm checks for a sustained repeated sequence and does not label pressure valleys or peaks as heel strikes.
3. QS ends at a sustained increase in local signal variability near the first accepted walking sequence. The first prominent loading peak in that sequence is used as an **operational GI-end candidate**, not a clinically validated end-of-initiation event.
4. Per-stride FMG level/range, insole level/range and stride duration are compared. A two-regime change-point fit selects one intervening candidate stride as the SLT window, with at least two observed strides on each side and no preferred clock time. Early and late regimes are named SSSW and SSLW under the requested short-to-long assumption. Signal pattern change by itself does not measure step length.
5. Candidate timing is tested by leaving out feature groups. Alternative nearly equal fits and group-ablation fits define the exported alternative timing range. This is **not a confidence interval**. Weak fits remain visibly marked instead of being presented as certain.
6. GT begins at the earlier peak in the final accepted bilateral peak pair. The rest of the record remains in this terminal region, including subsequent settling or postural movement, to match the requested six-region display. This is an **operational stopping-region estimate**, not proof that motion ceased exactly at the marker or that the full remaining signal is quiet standing.

Detection thresholds (including the peak-spacing limits) and operational definitions are heuristics recorded in `subject08_segmentation.py`. The tests establish computational consistency, delayed-onset sensitivity, abstention on flat signals, and timestamp-gap preservation; they do not validate physiological event timing.

## Unresolved intervals

- `sir_1`, trigger-pair trial 02, left and right: too few recurring load cycles for a supported six-phase fit. All sample labels are `UNRESOLVED`.
- `sir_21`, trigger-pair trial 18, left and right: missing data prevent a complete stride-feature transition estimate. All sample labels are `UNRESOLVED`.

`sir_1` contains 21 trigger-pair intervals although the protocol describes 20 trials for this file. Original trigger-pair numbering is preserved; no interval was silently discarded or renumbered.

## Reproduction and scope

Run `python generate_tempgraphs.py --overwrite` to regenerate these outputs. For read-only estimates, use `python subject08_segmentation.py --diagnose`. Optional filters: `--record sir_1 --trial 1 --side R`. Filtered runs do not rewrite full-dataset aggregate reports. Tests: `python -m unittest test_subject08_segmentation -v`.

The six raw Subject 08 source-file hashes were checked before and after generation. Raw data, other subjects, and the QS/GI output directories were not changed. The earlier `SUB08_REVIEW.md` describes the preceding unsegmented review and is superseded by this document for current phase labels.

Final consistency checks: all six automated tests passed, all 82 numeric exports were compared against recomputed timestamp-aligned source values, all 82 PNGs were opened successfully, and all 82 plots were visually inspected on the 21 current contact sheets. All 820 display transforms are unchanged from the preceding plots. These checks verify data handling and presentation, not the physiological truth of the provisional phase labels.
