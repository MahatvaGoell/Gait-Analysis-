# Sub01_H: noise-aware CoP graph refinement

80 complete limb graphs; 76 provisional layouts; 4 unresolved. 0 graphs contain supported additional closing intervals.

Each trial is evaluated independently. The original CoP valley-to-peak convention is retained, but narrow minima use a 3-sample detection smoother rather than an 11-sample median. The dominant CoP cadence and recurring loading changes restrict candidates to the main walking train. A short initial weight-shift interval is excluded when it is under 0.7 of this trial's typical stride. GI uses a local departure from quiet standing before that train, not a generic time.

SSSW contains the first five candidate intervals, SLT the sixth, and SSLW the next five. A candidate step runs valley-to-valley; its saved stance is valley-to-peak. Supported extra intervals are shown after SSLW as Extra step(s), E1, E2 ... . GT starts after these complete. Small stopping adjustments are not automatically counted as extra steps. The final contact fallback requires an actual loading change near the per-trial cadence prediction. If evidence is insufficient or trial triggers are unpaired, the full graph remains visible with unresolved cuts.

Display preprocessing: isolated 1-2 sample spikes are repaired only when surrounding samples agree; all traces receive a centred 3-sample moving average. True unloaded lows are preserved. Amplitude gains and offsets keep FMG below the CoP/vGRF display bands. No reference-shaped waveform, artificial valley floor, time stretching or invented step is used. Pale samples denote existing estimates or spike repairs.

All raw and segmented_data CSV bytes, including their previous label columns, are unchanged. Those old provisional_phase values may no longer match the graphs: current phase_cuts.csv and phase_intervals.csv are authoritative for this refinement. Saved stance CSVs contain original, not smoothed, measurements and existing estimate flags. preprocessing_audit.json records input/output hashes and repair counts for every graph. Prior derived stances and summaries are backed up outside the repository.

These are operational signal heuristics, not verified heel strikes, push-off or measured step lengths. Matching 5-1-5 counts do not establish physiological accuracy. Review warnings and missing measurements must be resolved against independent event evidence before model training. Other subjects are unchanged.

## Review on 2026-10-11

All 80 graphs were inspected in two visual passes: native CoP valley/peak candidates, then complete graphs with all ten signals. The full previews were hash-compared with production; both sides of abh_1 trial 13 were additionally inspected after the left initial preparatory weight shift was excluded and the paired onset recalculated. For subjects 01/02 only, a first CoP climb below 60% of the following median climb is excluded when its loading excursion is also below 30% of this recording's load range. This is a relative signal check, not a fixed timestamp or an invented cycle.

76 graphs have provisional 5-1-5 layouts; 3 of these retain visible review warnings. Four graphs are unresolved: abh_1 trials 02 L, 05 L, 07 L and 14 R. Three lack a sufficiently supported terminal contact; 07 L has fewer than eleven supported CoP stance candidates. Their complete traces are retained without fabricated phase cuts. Unresolved graphs are not ready for supervised segmentation training.

Automated checks passed for PNG integrity, original source hashes, ordered continuous intervals, 5-1-5 counts and original-valued valley-to-peak exports. All 12,032 protected pre-existing files, including source CSVs and other subjects, were byte-identical. Visual review and count checks do not certify physiological zero-error accuracy.
