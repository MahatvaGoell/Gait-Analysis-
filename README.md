# Gait Analysis — Study 1

This project analyses force myography (FMG) and insole data collected during walking. The current work prepares the recordings for automatic gait-phase segmentation.

## Completed work

- Kept the original recordings unchanged in `dataset/` for all 10 dataset folders.
- Read and aligned the FMG and insole streams with their trigger signals.
- Identified malformed rows, trigger inconsistencies, timestamp resets, missing streams, and missing samples.
- Generated complete per-recording graphs in `tempgraphs/`. Each graph contains 8 FMG channels, CoP, and vGRF for one limb.
- Segmented recordings into the requested sequence: QS → GI → SSSW → SLT → SSLW → GT.
  - QS: quiet standing
  - GI: gait initiation
  - SSSW: steady-state short-step walking
  - SLT: short-to-long-step transition
  - SSLW: steady-state long-step walking
  - GT: gait termination
- Used each recording’s own signal behaviour to estimate timing; no common fixed gait timing was imposed on every subject.
- Produced 100 Hz segmented numeric exports alongside the figures, with per-signal flags for estimated samples.
- Preserved original finite source-grid values. Where data were missing, the completed plots use documented temporal or within-recording cross-sensor estimates. Estimated portions are shown with pale lines and are flagged in the exported CSV files.

The phase labels currently saved with the figures are signal-derived, provisional labels. They provide training candidates and a structured review set; they are not yet independently verified clinical or ground-truth annotations.

## Current results

The completed graph folders are under `tempgraphs/`:

```text
tempgraphs/
├── sub01_a/ and sub01_h/
├── sub02_a/ and sub02_h/
├── sub03_h/
├── sub04_h/
├── sub05_h/
├── sub06_h/
├── sub07_h/
└── Sub08_H/
```

Within each completed subject folder, the main files are:

```text
<subject-folder>/
├── <recording>/trial_##_l_complete.png   # Left-limb figure
├── <recording>/trial_##_r_complete.png   # Right-limb figure
├── segmented_data/                       # 100 Hz signals, labels, and imputation flags
├── phase_boundaries_provisional.csv       # Estimated QS, GI, SSSW, SLT, SSLW, GT boundaries
├── phase_intervals_provisional.csv        # Phase intervals
├── load_landmarks_provisional.csv         # Pressure-cycle candidates
├── missing_value_report.csv               # How missing samples were completed
├── synchronization_report.csv             # Trigger-clock alignment details
├── recording_windows.csv                  # Recording interval boundaries
└── review_sheets/                         # Contact sheets for visual review
```

Subjects 01–04 contain a `summary.json` and `verification.json`. The verification confirms that the numeric exports reproduce the saved graphs, original available samples remain unchanged, estimates are flagged, and the raw data were not modified.

## Processing approach

Each trial keeps its original elapsed recording time. FMG and insole streams are aligned using their trigger events. The segmentation uses the sustained change from standing, repeating pressure-cycle landmarks, FMG activity, and a change in cycle features to estimate the short-to-long-step transition.

Some recordings have incomplete triggers or long missing sections. Those cases are retained and documented instead of being treated as fully observed data. A retained unpaired context is labelled `UNKNOWN` when the true trial boundary cannot be supported by the recording.

## Next step: train a segmentation model

The next stage is to train a model that learns the signal pattern and predicts the six gait phases for unseen recordings.

1. Review and correct the provisional phase boundaries where necessary, creating the final training labels.
2. Build trial-level features from the 8 FMG channels, CoP, vGRF, and useful timing or pressure-cycle features.
3. Split data by participant or recording before training, so data from the same person do not appear in both training and test sets.
4. Train a sequence-segmentation model to predict QS, GI, SSSW, SLT, SSLW, and GT at each sample or short time window.
5. Evaluate predicted boundaries and phase labels against the held-out reviewed labels, then inspect errors using the saved graphs.

The missing-value flags must be included in the training workflow. Any imputation used for machine learning should be fitted inside each training split so information from test recordings does not influence training.

## Main scripts

```text
generate_subjects01_to04.py     # Graphs, exports, synchronization, and verification for Subjects 01–04
generate_additional_subjects.py # Processing for the additional subject folders
complete_reconstruction.py      # Missing-value completion and phase helpers
subject08_segmentation.py       # Segmentation utilities and Subject 08 workflow
generate_qs_plots.py            # Earlier QS plotting workflow
generate_gi_outputs.py          # Earlier GI plotting workflow
```

Run a Subject 01–04 workflow from the project root, for example:

```powershell
python -B generate_subjects01_to04.py --subject Sub01_A
python -B generate_subjects01_to04.py --subject Sub01_A --verify
```

Required packages: `numpy` and `Pillow`.

## Data-use notice

The MIT License applies to the code only. The participant recordings and derived research data must be used according to the relevant consent, privacy, institutional, and research-governance requirements.
