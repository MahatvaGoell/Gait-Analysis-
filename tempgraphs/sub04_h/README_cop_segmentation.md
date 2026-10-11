# Sub04_H: noise-aware CoP graph refinement

80 complete limb graphs; 61 provisional layouts; 19 unresolved. 0 graphs contain supported additional closing intervals.

Each trial is evaluated independently. The original CoP valley-to-peak convention is retained, but narrow minima use a 3-sample detection smoother rather than an 11-sample median. The dominant CoP cadence and recurring loading changes restrict candidates to the main walking train. A short initial weight-shift interval is excluded when it is under 0.7 of this trial's typical stride. GI uses a local departure from quiet standing before that train, not a generic time.

SSSW contains the first five candidate intervals, SLT the sixth, and SSLW the next five. A candidate step runs valley-to-valley; its saved stance is valley-to-peak. Supported extra intervals are shown after SSLW as Extra step(s), E1, E2 ... . GT starts after these complete. Small stopping adjustments are not automatically counted as extra steps. The final contact fallback requires an actual loading change near the per-trial cadence prediction. If evidence is insufficient or trial triggers are unpaired, the full graph remains visible with unresolved cuts.

Display preprocessing: isolated 1-2 sample spikes are repaired only when surrounding samples agree; all traces receive a centred 3-sample moving average. True unloaded lows are preserved. Amplitude gains and offsets keep FMG below the CoP/vGRF display bands. No reference-shaped waveform, artificial valley floor, time stretching or invented step is used. Pale samples denote existing estimates or spike repairs.

All raw and segmented_data CSV bytes, including their previous label columns, are unchanged. Those old provisional_phase values may no longer match the graphs: current phase_cuts.csv and phase_intervals.csv are authoritative for this refinement. Saved stance CSVs contain original, not smoothed, measurements and existing estimate flags. preprocessing_audit.json records input/output hashes and repair counts for every graph. Prior derived stances and summaries are backed up outside the repository.

These are operational signal heuristics, not verified heel strikes, push-off or measured step lengths. Matching 5-1-5 counts do not establish physiological accuracy. Review warnings and missing measurements must be resolved against independent event evidence before model training. Other subjects are unchanged.

## Recheck on 2026-10-11

Native CoP candidate sheets were reviewed while refining detection, followed by a visual check of every final full graph. Of 80 graphs, 61 have candidate phase cuts; 28 of those retain additional review warnings. The remaining 19 show the complete recording without unsupported cuts. Unpaired triggers, insufficient CoP cycles or unclear final landing evidence are recorded in phase_cuts.csv; they are not repaired by inventing steps. Sensor saturation and long estimated sections remain visible rather than being disguised as measured waveforms.

Automated checks passed for all graph files, original source/output hashes, contiguous phase intervals, 5-1-5 candidate counts, extra-step ordering and the original measurement values in all 671 stance slices. Across both subjects, all 10,295 protected pre-existing files (including source CSVs and other subjects) remained byte-identical. These checks establish file consistency, not physiological ground truth.
