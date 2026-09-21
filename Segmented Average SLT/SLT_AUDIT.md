# Sub01_A SLT structure audit

No averages were calculated. This audit uses only the existing numerical exports in `tempgraphs/sub01_a`.

## Available grouping

- 20 trigger-paired physical windows are present: 10 in `jitu_1` and 10 in `jitu_21`.
- Each physical window produces left and right limb-specific data, yielding 40 SLT traces.
- Each trace has one continuous region already labelled `SLT`; it does not include a repetition or cycle-number field.

## SLT definition

The provisional phase order is QS → GI → SSSW → SLT → SSLW → GT. The existing code assigns SLT to the one interval from selected load-peak boundary `peaks[k]` to the next boundary `peaks[k+1]`. This is the single transition stride selected by a trial-wide two-regime cycle-feature fit. The same trial-level timing is applied to both limb traces. The pressure peaks are candidate landmarks, not validated heel strikes or ground-truth cycle labels.

## Can 200 existing SLT segments be recovered?

No. The valid stored grouping is 20 physical SLT transition intervals represented as 40 limb-specific traces. There are no five stored repetitions within each trace, and no per-cycle identifiers. Dividing an SLT trace into five parts would create artificial segments.

See `slt_trace_inventory.csv` for all 40 numerical trace boundaries and `slt_landmark_audit.csv` for landmark support at each of the 20 transition intervals.
