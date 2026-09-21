# Sub01_A GT structure audit

No averages were calculated. This audit uses only the existing numerical exports in `tempgraphs/sub01_a`.

## Available grouping

- 20 trigger-paired physical windows are present: 10 in `jitu_1` and 10 in `jitu_21`.
- Each physical window has left and right limb-specific data, yielding 40 GT traces.
- Each trace contains one continuous terminal region labelled `GT`; there is no repetition or cycle-number field.

## GT definition

The provisional phase order is QS → GI → SSSW → SLT → SSLW → GT. GT starts at the accepted terminal load-peak boundary at the end of SSLW and continues through the final available sample of the trigger-paired recording. Where a bilateral terminal mismatch met cadence and FMG-activity checks, GT can begin at the later supported peak. This is an operational stopping-region estimate. GT deliberately retains settling or postural movement, so it is not proof that motion ceased at its first sample.

## Can 200 existing GT segments be recovered?

No. The valid stored grouping is 20 terminal GT physical-window regions represented as 40 limb-specific traces. GT is not stored as a walking-cycle series, and it has no five-repetition or per-cycle structure. Splitting it into five parts would create artificial segments.

See `gt_trace_inventory.csv` for all 40 source boundaries and `gt_landmark_audit.csv` for the remaining candidate pressure landmarks after the GT start boundary.
