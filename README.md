# Gait Analysis

This repository contains the dataset, processing scripts, and generated results for a gait-analysis project using force myography (FMG) and insole measurements.

## Current scope

The current work covers the quiet-standing (QS) and gait-initiation (GI) sections of Study 1. FMG and insole recordings are aligned using their trigger signals, and malformed rows are excluded during processing.

Each QS graph contains:

- Eight FMG channels
- Centre of pressure (CoP)
- Vertical ground-reaction force (vGRF)

The displayed signals use a 0.25-second moving average to reduce sensor quantisation noise. Fixed vertical offsets place CoP and vGRF above the FMG channels without changing the raw source files.

GI is the fixed 3.0–4.0 second interval after the matched trial-start trigger. Each GI figure includes the preceding 0.5 seconds of QS as a standing reference, followed by the one-second GI interval; a dashed boundary marks the start of GI. This preserves the real FMG, CoP, and vGRF change from standing into movement while excluding later steady-walking samples.

## Repository structure

```text
Gait Analysis/
├── dataset/                         # Original recordings for all 10 subjects
├── output/
│   ├── qsoutput/
│       ├── per trial/               # Individual QS trial graphs by subject and recording
│       └── avg/
│           ├── per trialavg/        # Mean QS graph for each recording and limb
│           └── per subject/         # One whole-subject mean QS graph per subject
│   └── gioutput/                    # Matching GI-only graphs and averages
│       ├── per trial/
│       └── avg/
│           ├── per trialavg/
│           └── per subject/
├── generate_qs_plots.py             # Generates individual QS trial graphs
├── generate_qs_averages.py          # Generates recording-level averages
├── generate_subject_averages.py     # Generates whole-subject averages
└── generate_gi_outputs.py           # Generates GI trial and average graphs
```

## Generated results

- 729 individual QS trial graphs
- 127 recording-and-limb average graphs
- 10 whole-subject average graphs
- 687 individual GI graphs, each with a 0.5-second QS reference plus the fixed 3.0–4.0 second GI protocol window
- 127 GI recording-and-limb average graphs
- 10 GI whole-subject average graphs

Average curves are calculated from the numerical phase data, not from PNG images. Trials are resampled to a common 0–100% phase before point-by-point averaging. Forty-two trial/limb pairs are not shown in GI because their raw recording does not contain the full fixed 3.0–4.0 second interval; they are listed in the GI plot manifest.

## Running the scripts

Run the scripts from the repository root using Python:

```powershell
python generate_qs_plots.py --output output/qsoutput --overwrite
python generate_qs_averages.py
python generate_subject_averages.py
python generate_gi_outputs.py
```

Required Python packages:

```text
numpy
Pillow
```

## Data and synchronization notes

The raw dataset is retained unchanged. Some recordings contain missing, additional, or inconsistent trigger events. Review the `synchronization_report.csv` file in the relevant QS or GI `per trial` output folder before using FMG/insole combinations for statistical modelling.

## Data-use notice

The MIT License in this repository applies only to the source code. It does **not** grant permission to redistribute, publish, or reuse the participant recordings or derived research data. Access to the dataset must follow the applicable consent, institutional, privacy, and research-governance requirements.
