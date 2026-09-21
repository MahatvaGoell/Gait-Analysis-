# Sub01_A SSLW structure audit

No averages were calculated. This audit uses only the existing numerical exports in `tempgraphs/sub01_a`.

## Available grouping

- 20 trigger-paired physical windows are present: 10 in `jitu_1` and 10 in `jitu_21`.
- Each physical window has left and right limb-specific data, yielding 40 SSLW traces.
- Each trace contains one continuous set of samples labelled `SSLW`; there is no repetition or cycle-number field.

## SSLW definition

The provisional phase order is QS → GI → SSSW → SLT → SSLW → GT. SSLW begins at the selected load-peak boundary immediately after SLT and ends at the final accepted recurring load-peak boundary before GT. Its boundaries follow a trial-wide two-regime cycle-feature fit that selected one SLT transition interval. The pressure extrema are candidate landmarks, not verified heel strikes or ground-truth gait-cycle annotations.

## Can 200 existing SSLW segments be recovered?

No. The valid stored grouping is 20 continuous SSLW physical-window sections represented as 40 limb-specific traces. Candidate interpeak counts vary by window and limb, and no five-repetition structure or per-cycle identities are stored. Selecting or splitting five parts per trace would create artificial segments.

See `sslw_trace_inventory.csv` for all 40 source boundaries and `sslw_cycle_landmark_audit.csv` for the candidate pressure-landmark counts inside each SSLW section.
