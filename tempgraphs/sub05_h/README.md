# Subject 05 - full-trial provisional segmentation

All Subject 05 outputs are confined to this folder. Existing Subject 08 graphs, numeric exports, reports, source code and raw files are preserved. File hashes, sizes and modification times are checked before and after generation.

## Contents

- `sub_1` and `sub_21`: 20 trials per recording, left and right PNGs; 80 full-duration graphs total.
- `segmented_data`: corresponding numeric exports on the actual elapsed-time 100 Hz grid, with phase and missing-data flags. Values are not display-scaled.
- `phase_boundaries_provisional.csv` and `phase_intervals_provisional.csv`: trial-specific estimates. Intervals use half-open sample indices; the last displayed end time is the final grid sample.
- `load_landmarks_provisional.csv`: recurring load-peak candidates, not heel-strike labels.
- `complete_trial_manifest.csv`, `synchronization_report.csv`, `display_transforms.csv`: traceability, timestamp checks and plot scaling.
- `review_sheets`: 2x2 contact sheets.
- `provenance.json`: protected-file fingerprints and summary; `verification.json`: independent export checks.

## Same method as Subject 08

The existing Subject 08 segmentation and rendering functions are imported without modification. Only Subject 05 files are passed to them; no Subject 08 data or phase timings are used. Whole-body phase boundaries are estimated separately for each trial using both insoles and all 16 FMG channels, and shared between that trial's left and right plots.

The requested QS -> GI -> SSSW -> SLT -> SSLW -> GT order is **assumed, not confirmed by measured step length**. A two-regime comparison of cycle-wise FMG levels/ranges, insole levels/ranges and cycle durations selects a candidate SLT window. Movement onset and beginning/end loading landmarks provide operational GI and GT boundaries. The GT region includes the remaining recording, including settling or subsequent movement; it is not proof that the person is motionless throughout.

All labels remain provisional, not validated classification ground truth. The plots do not use the reference image's fixed times, and peaks are not artificially created. Y values use documented positive gains and offsets, as for Subject 08; they are not shared raw physical units. Timestamp gaps larger than 0.10 seconds remain blank.

## Results and exceptions

- 39 of 40 trials receive provisional six-phase fits (78 limb graphs).
- 3 transition fits have moderate pattern evidence and 36 have weak/ambiguous pattern evidence. These are algorithmic diagnostics, not accuracy probabilities or clinical validation.
- `sub_21`, trial 13: initial standing/onset is not reliably resolved by the same detector. Both limb plots and numeric phase labels are explicitly `UNRESOLVED`.
- `sub_21L` contains 39 observed rising trigger edges, while FMG and the right insole contain 40. Timestamp matching identifies the missing left-insole **end trigger of trial 16**. This end boundary is inferred from the robust clock offset of the other matched triggers, not from a fabricated trigger event. The inference is flagged on that graph and in the reports. Later trials are matched by timestamp rather than being shifted by one event. All sampled signal values still come from the recording.
- Malformed input rows are excluded and counted in the source audit; timestamps are not compressed to hide missing samples. Start-trigger alignment and small residuals are consistency checks, not proof of perfect sensor synchronization.

## Reproduction

From the project directory: `python -B tempgraphs/sub05_h/generate_subject05.py`. Add `--diagnose` for a read-only estimate. Verify outputs with `python -B tempgraphs/sub05_h/verify_subject05.py`. These scripts write only inside this folder; `-B` prevents imported bytecode caches being updated outside it.
