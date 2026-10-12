# Sub01_A: noise-aware CoP graph refinement

40 complete limb graphs; 39 provisional layouts; 1 unresolved. 2 graphs contain supported additional closing intervals.

Each trial is evaluated independently. The original CoP valley-to-peak convention is retained, but narrow minima use a 3-sample detection smoother rather than an 11-sample median. The dominant CoP cadence and recurring loading changes restrict candidates to the main walking train. A short initial weight-shift interval is excluded when it is under 0.7 of this trial's typical stride. GI uses a local departure from quiet standing before that train, not a generic time.

SSSW contains the first five candidate intervals, SLT the sixth, and SSLW the next five. A candidate step runs valley-to-valley; its saved stance is valley-to-peak. Supported extra intervals are shown after SSLW as Extra step(s), E1, E2 ... . GT starts after these complete. Small stopping adjustments are not automatically counted as extra steps. The final contact fallback requires an actual loading change near the per-trial cadence prediction. If evidence is insufficient or trial triggers are unpaired, the full graph remains visible with unresolved cuts.

Display preprocessing: isolated 1-2 sample spikes are repaired only when surrounding samples agree; all traces receive a centred 3-sample moving average. True unloaded lows are preserved. Amplitude gains and offsets keep FMG below the CoP/vGRF display bands. No reference-shaped waveform, artificial valley floor, time stretching or invented step is used. Pale samples denote existing estimates or spike repairs.

All raw and segmented_data CSV bytes, including their previous label columns, are unchanged. Those old provisional_phase values may no longer match the graphs: current phase_cuts.csv and phase_intervals.csv are authoritative for this refinement. Saved stance CSVs contain original, not smoothed, measurements and existing estimate flags. preprocessing_audit.json records input/output hashes and repair counts for every graph. Prior derived stances and summaries are backed up outside the repository.

These are operational signal heuristics, not verified heel strikes, push-off or measured step lengths. Matching 5-1-5 counts do not establish physiological accuracy. Review warnings and missing measurements must be resolved against independent event evidence before model training. Other subjects are unchanged.

## Individual edge review correction

9 trial/limb edge reviews are stored in amputee_edge_reviews.json, locked to source CSV SHA256. 1 graphs retain supported early phase cuts but have an unconfirmed final contact. Their SSLW_CANDIDATE range includes the unassigned tail, NOT a verified long-walking interval. Their last saved valley-to-peak climb has interval_status=unclosed and a blank next_valley_sample. No GT boundary or fifth complete long interval is manufactured. These cases are not training labels.
