# SUB01_H: complete graphs with provisional segmentation

Sub01_H: abh_1 and abh_21. The latter has 44 insole trigger edges but only 42 FMG edges; the fuller insole sequence preserves the additional interval and exposes its missing FMG coverage.

## Outputs

Record-named folders contain one left and one right PNG per retained interval, each showing 8 FMG channels, CoP and vGRF. segmented_data contains the corresponding 100 Hz numeric series, phase labels and ten per-channel _imputed flags. review_sheets contains contact sheets. summary.json reports counts, remaining blanks and partial phase annotations.

No reference-image phase timing is imposed. Times are elapsed seconds on the aligned recording clock, not time-normalized strides. Positive display gains and offsets distinguish signals on the figure; the y-axis does not represent one common physical unit.

## Synchronization and windows

Trigger-offset consensus matches timestamps despite missing pulses or extra trigger activity. Each right-insole clock section in Subject 4 is treated independently; an early section without trigger anchors remains unmapped. Subject 4 uses FMG as its stable clock reference, never the noisy left trigger sequence. recording_windows.csv logs the chosen windows and inferred boundaries. synchronization_report.csv contains offsets, matched edges and residual errors. These residuals quantify trigger alignment, not physiological synchronization validation.

Paired pulses define the ordinary windows. If a start is followed only much later by the next plausible start, the intervening context is retained and explicitly labelled unpaired, rather than consuming the next start as its end and shifting all subsequent trials. Retained interval counts are not independently verified protocol-trial counts.

## Missing values and segmentation

The method matches the model-completed Subject 06/07 views. Every finite value from the aligned source grid remains unchanged. Missing samples are estimated using cross-sensor ridge regression or temporal interpolation with endpoint holding. For within-recording fits, four observed one-second blocks compare the two methods. An entirely absent group can use other intervals from this same dataset (never another subject or A/H group). All estimates are flagged. Pale lines denote estimated values; dark lines denote available source-grid values.

Full numeric coverage is not a guarantee of reconstruction accuracy. Long gaps, edge holds and wholly absent streams have no target measurements to verify. The method, donor intervals and available holdout errors are in missing_value_report.csv. Setting flagged values to blank recovers the incomplete source grid. Raw text files remain untouched.

QS -> GI -> SSSW -> SLT -> SSLW -> GT is the user-requested short-to-long sequence assumption. Boundaries are fitted independently per recording from signal activity and cycle-feature changes, including reconstructed data where present. Pressure extrema are not verified heel strikes, and step length is not measured by these labels. A partial WALK/UNKNOWN annotation is retained where all six phases cannot be supported. GT includes the ending window, not proof of a motionless final posture.

**These figures are model-completed visualizations, not recovered measurements or classification ground truth.** For a classification experiment, validate event labels independently and refit imputation inside training folds to avoid leakage.

## Reproduction and verification

Ending-marker audit: when the two pressure-cycle sequences end at different times, GT is extended only if the later cycle has a cadence-consistent interval (0.8-1.25 times the median interval), strong FMG activity around that cycle (90th percentile of the normalized activity score at least 2), and at least 0.8 seconds of ending data. Otherwise the bilateral discrepancy remains flagged for review. This is an operational signal rule, not a validated stopping event.

Generate only this dataset:
python -B generate_subjects01_to04.py --subject Sub01_H

Verify:
python -B generate_subjects01_to04.py --subject Sub01_H --verify

Use --audit for a read-only clock/window audit. Omit --subject to process all six datasets. verification.json checks every numeric export, estimate flag, phase partition and PNG, plus preservation of raw and previous-subject files. It does not validate physiological correctness. Subjects 05-08 and their generation scripts remain untouched.
