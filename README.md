# Gait Analysis — Study 1

This project analyses force myography (FMG) and pressure-insole data recorded during walking. Each recording holds 8 FMG channels plus centre of pressure (CoP) and vertical ground reaction force (vGRF) for the left and the right limb.

The current work cuts every walking trial into six gait phases and saves the stance phase of every step:

| Phase | Meaning |
|---|---|
| QS | quiet standing |
| GI | gait initiation |
| SSSW | steady-state short-step walking (5 steps: S1–S5) |
| SLT | short-to-long transition (1 step: T) |
| SSLW | steady-state long-step walking (5 steps: L1–L5) |
| GT | gait termination |

## How a trial is cut

Everything is read from each limb's own CoP, not from vGRF.

- **Valley**: the CoP minimum at foot contact. For the limb that steps first, the first valley is the low point its CoP climbs out of when the person leaves quiet standing.
- **Peak**: the highest CoP point of the stance, at push-off.
- **Step**: one valley to the next. **Stance** is valley to peak and is what gets saved; the swing after the peak is not saved.
- **Phase cuts** sit on valleys, counted from the first one (v0): GI = rise to v0, SSSW = v0–v5, SLT = v5–v6, SSLW = v6–v11, GT = v11 to the end.

This layout reproduces the reviewer's marks on the one trial that has them (Sub08_H, sir_21, interval 04, left: cuts at samples 1005 / 1119 / 1722). No other recording has reviewer marks, so every other cut is a signal-derived candidate, not a validated gait event. SLT is placed by count (the sixth step); a short-to-long change in step length is not measured.

Two detectors are used. The primary one is in `refine_sub08_h_valleys.py`. A second, independently written one (`refine_subject03_04.py`) proposes its own candidates, which are placed on the same valley and peak convention. They are compared on Subjects 01–05 and the amputee sets. Where the two disagree, the layout whose left and right cuts alternate is used and the graph is marked for review. Nothing is forced: a limb without eleven usable stances stays UNRESOLVED.

## Results

| Set | Limb graphs | Cut | Unresolved | Cut graphs with a review note |
|---|---|---|---|---|
| Sub01_A (amputee) | 40 | 38 | 2 | 27 |
| Sub02_A (amputee) | 44 | 43 | 1 | 26 |
| Sub01_H | 80 | 80 | 0 | 15 |
| Sub02_H | 84 | 84 | 0 | 13 |
| Sub03_H | 100 | 93 | 7 | 12 |
| Sub04_H | 80 | 60 | 20 | 23 |
| Sub05_H | 80 | 68 | 12 | 24 |
| Sub06_H | 80 | 80 | 0 | 12 |
| Sub07_H | 86 | 82 | 4 | 11 |
| Sub08_H | 82 | 80 | 2 | 2 |
| **Total** | **756** | **708** | **48** | **165** |

7,788 stance segments are saved. Things to know before using them:

- The right insole of Sub04_H gives an almost flat load and a noisy CoP; most of its unresolved graphs are right-limb graphs.
- Every set except Sub05_H and Sub08_H contains model-estimated samples. They are pale in the graphs and flagged in the tables; a cut on estimated CoP is never presented as measured.
- In the amputee sets, 20 graphs were reviewed one by one and most of them contain edges entered by hand (`amputee_edge_reviews.json`).
- A review note means the layout deserves a look (detector disagreement, weak CoP climb, unusually long step, cut on estimated CoP, and so on). A matching 5–1–5 count or an unflagged graph is not a validation.

## Repository layout

```text
dataset/                     Raw recordings (never modified)
tempgraphs/
├── Sub08_H/                 Graphs and tables for Subject 08
├── segmented_data/Sub08_H/  100 Hz exports for Subject 08
└── sub01_a/ … sub07_h/      One folder per other set, each with its own segmented_data/
```

Inside a subject folder:

```text
<record>/trial_##_<l|r>_complete.png   Graph of one limb for one recording interval
stance_segments/                       One CSV per stance (CoP valley to CoP peak), values as exported
stance_segments.csv                    Valley, peak and next-valley time and sample of every saved stance
phase_cuts.csv                         Phase cuts, status, detector cross-check and review notes per graph
phase_intervals.csv                    Phase spans in seconds and samples
README.md / README_cop_segmentation.md How this set was cut, and its source limitations
segmented_data/                        100 Hz exports; `provisional_phase` matches phase_cuts.csv
```

The remaining files in a subject folder (`complete_trial_manifest.csv`, `summary.json`, `provenance.json`, synchronization, recording-window and missing-value reports, and the older `README.md`) document how the 100 Hz exports were produced from the raw recordings. Any phase timing in them belongs to an earlier provisional segmentation.

`QS/`, `GI/`, `GT/`, `SLT/`, `SSLW/`, `SSSW/` and the `Segmented Average …` folders hold per-phase plots and averages from an earlier stage. They were made with the earlier provisional phase labels and have not been regenerated from the CoP cuts.

## Scripts

```text
refine_sub08_h_valleys.py         CoP segmentation, graphs, stance segments and tables for every set
export_second_engine_trains.py    Records the second detector's candidates in second_engine_trains.json
refine_subject03_04.py            Second detector (with refine_subject01_02.py and refine_subject01_02_amputee.py)

subject08_segmentation.py, generate_subjects01_to04.py, generate_additional_subjects.py,
tempgraphs/sub05_h/generate_subject05.py
                                  Build the 100 Hz exports from the raw recordings (with their helper modules)
generate_qs_*.py, generate_gi_*.py, generate_sssw_*.py, generate_subject_averages.py
                                  Earlier per-phase plotting workflows
```

Run from the project root:

```bash
python export_second_engine_trains.py      # once, or after the exports change
python refine_sub08_h_valleys.py all       # or one set, e.g. Sub08_H
```

Required packages: `numpy`, `scipy` and `Pillow`.

The export-generation scripts redraw the graphs in their old form, so rerun `refine_sub08_h_valleys.py` after them. In `segmented_data` only the two phase-label columns are ever rewritten; waveform values are written back exactly as read.

## Data-use notice

The MIT License applies to the code only. The participant recordings and derived research data must be used according to the relevant consent, privacy, institutional, and research-governance requirements.
