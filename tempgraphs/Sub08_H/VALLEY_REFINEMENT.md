# Sub08_H valley marker refinement

The PNG curves retain their input values and original 100 Hz timeline.
Only valley annotation sample positions were refined. No flattening, smoothing,
rescaling in time, or artificial valleys were added by this update.

Each previously reviewed S1–S5, transition T, and L1–L5 event is searched
locally within 15 samples on either side, confined to its reviewed phase.
The lowest unsmoothed input value is used. Where consecutive samples share
exactly the same minimum, the middle observed minimum sample is marked.
This is an explicit timing convention, not a claim of confirmed heel contact.

Two S4 events were inspected individually: sir_21/trial_09_r uses a 17-sample
window, and sir_21/trial_14_r uses an 18-sample window, to include their nearby
input troughs. Their original narrow search windows ended at the minimum.

The valley sample positions are retained in the audit and label locations;
solid valley guides and dots were removed at the user's request. Full-height
dashed lines remain the individually reviewed phase boundaries. A valley
minimum and the start/end of a walking phase are distinct annotations.

`valley_sample_audit.csv` lists the old and new times, sample shifts, values,
flat-bottom ties, search windows, and estimated-input flags for all 880 events.
The input CSVs contain previously model-completed values; the audit flags
estimated marker samples wherever the input provides this information.
13 plots retain ambiguous additional minima; two recordings do not support
the requested complete 5–1–5 cycle sequence. Those limitations are retained.

The supplied sample is a visual reference, not the mentor's sample-index labels
for these recordings. Exact agreement with unseen manual marks cannot be tested.
