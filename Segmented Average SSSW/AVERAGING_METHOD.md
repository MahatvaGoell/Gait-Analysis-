# Sub01_A SSSW averages

This output averages the 40 existing limb-specific SSSW traces for Sub01_A. Only numerical rows already labelled `SSSW` by the saved segmentation were used.

Each trace was linearly resampled to 300 points spanning 0–100% of its own SSSW section. A point-by-point arithmetic mean was calculated separately for each of the eight FMG channels. No five-repetition or 200-cycle grouping was created.

`average_graphs/sssw_average_all_signals.png` is the combined presentation average. It includes all eight FMG channels, CoP, and vGRF in the same SSSW phase-band layout as the corrected per-trial graphs. Before averaging for this figure, every trace receives the baseline, gain, and lane offset from its own complete trial graph; the accompanying `average_data/sssw_display_average_all_signals.csv` records those displayed means, standard deviations, and trace counts.

- `average_data/` contains one 300-row numerical CSV per channel.
- `average_graphs/` contains the existing per-channel mean PNGs and the combined SSSW presentation-average graph.
- `average_trace_manifest.csv` records all source traces used.
- `average_summary.json` records the processing result.

All 40 traces were valid and used. No traces were rejected.
