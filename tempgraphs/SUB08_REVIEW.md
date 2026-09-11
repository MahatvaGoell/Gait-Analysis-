# Subject 08 plot correction

> Historical review of the preceding unsegmented plots. Current provisional sequence labels and limitations are documented in SEGMENTATION_README.md; previous graphs are preserved in _before_sequence_segmentation.

82 full-duration plots: 21 trigger-pair intervals in sir_1 and 20 in sir_21, with separate left and right plots. These are trigger-pair indices, not independently verified experimental trial labels. All 82 plots were visually inspected using the 11 contact sheets in review_sheets.

## Corrected time and signal handling

- Each stream uses its own recorded timestamps relative to its corresponding trial-start trigger. Recordings are not stretched to 22 seconds or shifted to a common movement onset. FMG durations span 13.08–29.30 seconds.
- Values are interpolated to a 100 Hz time grid; timestamp gaps greater than 0.10 seconds remain blank. Repeated timestamps retain the first packet, and out-of-order packets are sorted. Counts and end-trigger differences are audited per trial.
- The plot contains eight FMG channels and the paired insole CoP and vGRF. Display-only offsets and positive scale factors separate traces; the y-axis explicitly says scaled amplitude. Exact transforms are in display_transforms.csv: display = offset + gain * (value - baseline). These are not shared raw physical units and should not be used to compare amplitudes across trials.
- Raw dataset files are unchanged. QS and GI output folders were not regenerated in this correction.

## Segmentation status — provisional, not ground truth

Movement and final-settling candidates are estimated separately for every limb/trial from local signal variability relative to its baseline. This is one reusable analysis method with trial-specific measurements, not a universal timing schedule. The present method identifies 82 onset candidates and only 9 final-settling candidates meeting its conservative tail criterion. An absent marker does not prove that the person never stopped.

Movement onset is not a validated heel-rise event, and final settling is not a validated gait-termination start. The labels QS/GI/SSSW/SLT/SSLW/GT cannot all be established by amplitude or peak size alone. No universal phase boundaries are drawn. Walking-stage transitions remain explicitly unresolved until corresponding event annotations or video are available. The earlier claim that complete phase sequences had been verified was not justified.

## Notable individual observations

- sir_1, trigger-pair trial 01: full duration about 26.31 seconds; onset candidates 6.21 seconds left and 6.47 seconds right. The old 22-second view omitted part of the ending. Separate limb estimates are provisional, not evidence of clock offset.
- sir_1, trial 02, both limbs: approximately 13-second atypical interval, without the same sustained walking sequence as adjacent trials. Check the acquisition/trial labels before treating this as a complete protocol trial.
- sir_1, trials 11 and 15: late missing intervals are visible; do not interpolate those into a steady ending.
- sir_1, trial 16 right: additional movement follows the main repetitive walking pattern; do not label the first reduction in peaks as final standing.
- sir_21, trial 03, both limbs: substantial post-walking posture/sensor changes remain visible around 20–26 seconds. Ending at the last regular stride would omit these changes.
- sir_21, trial 18: late timestamp gaps require care when interpreting settling.

The manifest provides every plot's measured duration, candidate markers, and timestamp-quality flags. Start-trigger alignment plus a small end-trigger difference is a consistency check, not proof of perfect sensor synchronization throughout a trial.
