# Sub01_A SSLW averages

This output averages the 40 existing limb-specific SSLW traces for Sub01_A. Only numerical rows already labelled `SSLW` by the saved segmentation were used.

Each trace was linearly resampled to 300 points spanning 0–100% of its own SSLW section. A point-by-point arithmetic mean was calculated separately for each of the eight FMG channels. No five-repetition or 200-cycle grouping was created.

- `average_data/` contains one 300-row numerical CSV per channel.
- `average_graphs/` contains one mean waveform PNG per channel.
- `average_trace_manifest.csv` records all source traces used.
- `average_summary.json` records the processing result.

All 40 traces were valid and used. No traces were rejected.
