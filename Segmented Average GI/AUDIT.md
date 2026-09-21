# Sub01_A GI trigger and trial audit

## Raw files

The subject has two FMG batches with matched left and right insole files:

- `jitu_1`, `jitu_1L`, `jitu_1R`
- `jitu_21`, `jitu_21L`, `jitu_21R`

Each stream in each batch has 20 rising trigger pulses.

## Trigger meaning supported by the raw timing

Every batch follows the same alternating pattern:

- Trigger 1 to 2, 3 to 4, and so on: 22.13 to 25.55 seconds.
- Trigger 2 to 3, 4 to 5, and so on: 5.36 to 10.18 seconds.

The long interval has the duration and repeated FMG/insole activity expected of one overground gait bout. The following shorter interval is the between-bout gap. The raw trigger pulse is not explicitly labelled as `start` or `end`; those roles are inferred from this stable timing pattern and the synchronized corresponding pulses in all three streams.

## Current project mapping

The existing pairing logic is consistent with the raw pattern:

- Each batch has 10 start/end trigger pairs, so 10 physical gait-bout windows.
- Both batches therefore contain 20 physical windows.
- Each window contains both left and right limb data, exported as two limb-specific GI traces.
- This produces the existing 40 GI traces: 20 windows times 2 limbs.

## Protocol-document discrepancy

The study plan documents `X_1` as trials 1 to 20 and `X_21` as trials 21 to 40. The raw files contain only 20 start/end pairs total, which does not support 40 independently delimited physical gait-bout windows. The document and the raw trigger timeline therefore use different trial-counting conventions, or one of them is incomplete. There is no raw event or existing project field that safely maps 40 protocol trial numbers to 40 independent GI windows.

## Conclusion

The current trigger-pairing does not appear to collapse 40 physical trial windows into 20. It pairs the start and end pulse of each supported gait bout. We cannot correctly recover 40 independent GI trial segments before left/right splitting from the current raw files alone.

The CSV files in this folder provide the complete event timeline and the 20 paired-window mapping. No average was calculated and no existing phase folder was changed.
