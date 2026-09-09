# Gait Analysis

This repository contains the dataset, processing scripts, and generated results for a gait-analysis project using force myography (FMG) and insole measurements.

## Current scope

The current work covers the quiet-standing (QS) section of Study 1. FMG and insole recordings are aligned using their trigger signals, malformed rows are excluded during processing, and the QS interval is cut before detected movement onset.

Each QS graph contains:

- Eight FMG channels
- Centre of pressure (CoP)
- Vertical ground-reaction force (vGRF)

The displayed signals use a 0.25-second moving average to reduce sensor quantisation noise. Fixed vertical offsets place CoP and vGRF above the FMG channels without changing the raw source files.

## Repository structure

```text
Gait Analysis/
├── dataset/                         # Original recordings for all 10 subjects
├── output/
│   └── qsoutput/
│       ├── per trial/               # Individual QS trial graphs by subject and recording
│       └── avg/
│           ├── per trialavg/        # Mean QS graph for each recording and limb
│           └── per subject/         # One whole-subject mean QS graph per subject
├── generate_qs_plots.py             # Generates individual QS trial graphs
├── generate_qs_averages.py          # Generates recording-level averages
└── generate_subject_averages.py     # Generates whole-subject averages
```

## Generated results

- 729 individual QS trial graphs
- 127 recording-and-limb average graphs
- 10 whole-subject average graphs

Average curves are calculated from the numerical QS data, not from PNG images. Trials are resampled to a common 0–100% QS phase before point-by-point averaging.

## Running the scripts

Run the scripts from the repository root using Python:

```powershell
python generate_qs_plots.py --output output/qsoutput --overwrite
python generate_qs_averages.py
python generate_subject_averages.py
```

Required Python packages:

```text
numpy
Pillow
```

## Data and synchronization notes

The raw dataset is retained unchanged. Some recordings contain missing, additional, or inconsistent trigger events. Review `output/qsoutput/per trial/synchronization_report.csv` before using FMG/insole combinations for statistical modelling.

## Data-use notice

The MIT License in this repository applies only to the source code. It does **not** grant permission to redistribute, publish, or reuse the participant recordings or derived research data. Access to the dataset must follow the applicable consent, institutional, privacy, and research-governance requirements.

