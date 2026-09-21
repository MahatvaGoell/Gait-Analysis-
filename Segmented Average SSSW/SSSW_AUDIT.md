# Sub01_A SSSW structure audit

No averages were calculated. This audit uses the existing `tempgraphs/sub01_a` numerical exports only.

## Available grouping

- There are 20 physical trigger-paired windows: 10 in `jitu_1` and 10 in `jitu_21`.
- Each physical window has the same trial-level phase boundaries for left and right, producing 40 limb-specific SSSW traces.
- Each source trace has one continuous sequence of rows labelled `SSSW`; it has no cycle number or repetition field.

## SSSW definition

The saved segmentation uses the provisional sequence QS → GI → SSSW → SLT → SSLW → GT. SSSW begins at the first recurring load-peak candidate used as the operational end of GI. It ends at the selected load-peak boundary immediately before the single SLT region. The selection comes from a trial-wide two-regime cycle-feature fit using the 16 FMG channels and both insoles. These load peaks are explicitly candidate landmarks, not verified heel strikes or ground-truth cycle labels.

## Can 200 existing SSSW segments be recovered?

No. The data supports 40 continuous limb-specific SSSW sections, not 40 traces each carrying five identified repetitions. The number of candidate interpeak intervals inside SSSW varies across trials and limbs. No per-cycle segmentation, repetition identity, or five-cycle grouping is stored. Splitting every trace into five would create artificial segments and is not supported by the present data.

See `sssw_trace_inventory.csv` for the exact 40 trace boundaries and `sssw_cycle_landmark_audit.csv` for the existing candidate pressure-landmark counts.
