# Subject 06: continuous model-completed graphs

80 PNGs cover 40 recording intervals. All 40 have model-assisted provisional six-stage fits. Numeric series have no blank values, but estimates are not recovered measurements.

These are Rishabh_1 through Rishabh_40. Most have only a start trigger: the retained window ends at the observed FMG file end, which is not a verified end-of-trial event. Missing start triggers use documented clock calibration.

## Reading the completed graphs

Dark lines are finite values from the original 100 Hz source grid. Pale lines are estimates. A fully absent channel is labelled [est.] in the legend. The y-axis uses positive display gains and offsets; it does not place FMG, CoP and vGRF in one physical unit. No source-grid value is changed, no reference-image timing is imposed, and no recording is stretched or extended.

These are **model-completed visualizations, not recovered measurements**. Full coverage is guaranteed, not physiological accuracy. This replaces the earlier conservative pass that left uncertain gaps blank.

## Processing and estimation

The existing timestamp/trigger alignment, separate clock epochs, malformed-row exclusion and 100 Hz resampling are retained. The timestamp-outlier test is vectorized without changing its conditions.

For each recording, missing FMG, left-insole and right-insole groups are processed separately. A ridge-regression model predicts a target group using the other available sensor groups at the current time and +/-0.10 seconds, with linear, squared and tanh features. It never uses the target group's channels as predictors. This is offline reconstruction, not causal real-time prediction.

When the target group has data, four observed one-second blocks are hidden. The regression is compared against time interpolation/endpoint holding using group-normalized RMSE. The lower-error method is selected when this comparison is available; otherwise temporal interpolation/endpoint holding is used. Models use at most 5,000 training rows, ridge penalty 10, and training-range bounds on their predictions. Model splices are adjusted only inside the missing interval to join available endpoints.

If an entire target group is absent, a model is learned from other intervals of the **same subject**, preferring intervals from the same source recording. Donor-interval holdout error is reported where possible. These predictions have no target-trial measurements to verify them. A constant subject prior is an explicitly named last resort if no usable predictive model exists; the report identifies the method actually used.

There is no acceptance threshold that leaves a gap blank in this completion mode. Large holdout errors, long gaps, edge extrapolations and wholly absent streams may be inaccurate. One-second holdout checks do not validate multi-second missing intervals. Estimated curves must not be described as recorded data.

## Segmentation

QS -> GI -> SSSW -> SLT -> SSLW -> GT is the requested, unverified short-to-long order. Phase boundaries are fitted separately to each completed recording, so estimates can affect them. Recurring pressure extrema are loading landmarks, not verified heel strikes. Short/long labels reflect an assumed sequence and a change in cycle features, not measured step length. GT includes the ending window and does not prove a motionless final posture.

Where the six-stage estimator fails, a single-limb cycle sequence can support a provisional fit. Otherwise the plot receives partial annotations: QS/GI/WALK, with WALK combining short/long states that cannot be distinguished. Phases absent from a truncated recording are not manufactured.

## Files and reproducibility

The existing record folders contain updated PNGs; segmented_data contains the complete numeric series and ten per-channel _imputed flags. All rows have signal_missing=0 in this model-completed view. Set flagged estimates to blank to recover the original incomplete grid. Labels carry model_assisted_not_ground_truth status.

missing_value_report.csv records every filled channel interval, its method, training source, donors and holdout errors. summary.json gives coverage/segmentation counts. verification.json checks reproducibility, preserved source-grid values, no remaining NaNs, estimate flags, PNG integrity and protected raw/other-subject files. These checks do not validate missing measurements or physiological labels.

Run: python -B generate_additional_subjects.py --complete-model
Verify: python -B verify_complete_subjects.py

Add --subject Sub06_H or --subject Sub07_H to regenerate just one subject. Omitting --complete-model invokes the older conservative mode and can reintroduce blank gaps. Raw datasets and Subjects 05/08 remain unchanged, and no new directories are created.
