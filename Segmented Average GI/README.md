# Sub01_A segmented GI averages

This folder contains the average of the 40 existing limb-specific GI traces for Sub01_A: 10 trigger-paired physical windows in `jitu_1` and 10 in `jitu_21`, each split into left and right limbs.

Only rows already labelled `GI` by the existing segmentation were used. Each trace was linearly resampled to 300 points spanning 0–100% of its own GI section. The point-by-point mean was then calculated separately for each of the eight FMG channels. No five-repetition or 200-cycle structure was created.

- `average_data/`: one 300-row numerical CSV per channel, containing normalized GI percent, mean, standard deviation, and contributing-trace count.
- `average_graphs/`: one mean waveform PNG per channel.
- `trace_manifest.csv`: all source traces and their GI sample counts.
- `average_summary.json`: machine-readable processing summary.

All 40 traces were valid and used; 0 traces were rejected.
