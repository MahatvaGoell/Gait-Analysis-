# Study 1 training plan: who does what

Ojaswi works with Claude Code and Mahatva works with Codex. Both agents read `AGENTS.md` (Claude through `CLAUDE.md`) for the repository rules. To start a task, tell the agent: *"Read AGENTS.md and TRAINING_PLAN.md, then do task O1"* (or M1, ...).

## Where we are (12 Nov 2026)

These numbers are what the conference paper reports. They come from 8 healthy subjects, 7,196 measured windows and 6 classes: GI, short, long, S→L, L→S and GT.

| Run | Result |
|---|---|
| Per gait cycle, SVM, 128 features + duration, LOSO | balanced accuracy (BA) 57.5 ± 8.1 |
| Same SVM, 128 features, within-subject | BA 80.5 ± 6.5 |
| Weakest classes (LOSO) | S→L and L→S transitions, recall about 21% |
| Calibration + context (`run_cycle_context.py`) | stopped; only the `baseline` config ran |
| Real time (`run_realtime.py`) | stopped; only the 300 ms LDA run exists |
| Deep models | not started; PyTorch is not installed |

Labels are still provisional, because sir has not confirmed the walking direction. Normal-step recordings have not been located.

## Step 0: before anyone starts (Ojaswi, once)

1. Commit and push the model code, which is still uncommitted: `common/`, `study1/`, `results/*.csv`, `paper/`, `.gitignore`, `README.md`, `AGENTS.md`, `CLAUDE.md` and this file. Do not commit `NewSegmentedGraphsFinal/` (577 MB) or `datasets/`.
2. Mahatva pulls `main` and rebuilds the data set with `python common/build_dataset.py`. This writes `datasets/healthy_graphs.pkl` with 627 limb graphs.
3. Mahatva runs `python -m study1.cycle.run_cycle` once. It must print the same LOSO SVM numbers as `results/cycle_summary.csv`, which shows that both machines have the same data.

## Ownership

| Owner | Files they may change |
|---|---|
| Ojaswi | `common/labels.py`, `common/evaluate.py`, `study1/cycle/` and `results/cycle_*` |
| Mahatva | `common/build_dataset.py`, `study1/realtime/`, `study1/deep/` (new), `requirements.txt` (new), and `results/realtime_*` and `results/deep_*` |

To change a file the other person owns, ask them first. `common/labels.py` and `common/evaluate.py` define the labels and the splits for both tracks, so a change there means both tracks re-run.

## Ojaswi (Claude): per-cycle track and labels

### O1: Finish calibration + context

Run `python -m study1.cycle.run_cycle_context`.

Done when `results/cycle_context_summary.csv` holds all four configs. Note whether calibration or context helps under LOSO.

### O2: Transitions (the weakest classes)

Transition recall is about 21% under LOSO. Try these in order and report each one under LOSO:
1. Features of the previous 1–2 steps of the same walk (past data only).
2. A two-stage classifier: steady step vs transition, then the direction of the transition.
3. Decoding along the walk with the protocol order GI → steady → transition → steady → GT.

The third option uses knowledge of the protocol, so report it separately and label it as such.

Done when there is a table of transition recall and overall BA for each method.

### O3: Tuned baselines

So far every model used fixed settings. Tune SVM (C, γ) and HGB (depth, learning rate) with an inner LOSO on the 7 training subjects, never on the held-out subject.

Done when tuned and fixed results are reported side by side.

### O4: Final labels

When sir answers (walking direction, normal-step recordings, GI/GT definitions), update `common/labels.py` and re-run every cycle script. Tell Mahatva, because his runs must be repeated too. This task is blocked on sir.

### O5: Journal, amputee labels

Label Sub01_A and Sub02_A: walking direction, and a check of the hand-entered valleys. Mark the amputated and intact limbs.

## Mahatva (Codex): real-time track, deep models, data set

### M1: Finish the real-time baselines

Run `python -m study1.realtime.run_realtime` (300 ms LDA + HGB, 200 ms LDA). Add one metric: false switches per trial. Count the changes in the predicted class after the 3-window debounce, minus the true class changes in that trial, and report the median over scored trials.

Done when `results/realtime_summary.csv` holds BA, detection rate and median delay for GI, the transition and GT.

### M2: Real-time ablations

Vary the window length (100, 200, 300 and 400 ms). Vary the channel subsets: the thigh compartments of `study1/cycle/run_extra.py`, each one alone and the band without it.

Done when there is a table of BA and median delay against window length and channels.

### M3: Deep models (new `study1/deep/`)

Add PyTorch to `requirements.txt` and train two models:
- **Per cycle:** a small 1D-CNN on the same windows as `run_cycle.py`. Input is 8 channels × 100 points, baseline-subtracted, using `common.labels.cycle_windows` and `resample` from `study1/cycle/run_cycle.py`.
- **Real time:** a causal TCN on the 200–300 ms windows of `run_realtime.py`.

Training rules:
- Use the splits in `common.evaluate`.
- Use class-weighted loss.
- Fix the number of epochs before testing, or early-stop on one inner validation subject, never the test subject.
- Use 3 seeds and report mean ± SD.

Done when `results/deep_summary.csv` sits next to the SVM (57.5 LOSO / 80.5 within-subject) and the M1 real-time numbers.

### M4: Journal, amputee data set

Extend `common/build_dataset.py` to Sub01_A and Sub02_A, writing `datasets/amputee_graphs.pkl` with an amputated/intact flag per limb. Use O5's labels.

## Together

- Each person reviews the other's diff once a week. Ojaswi asks Claude to review Codex's commits, and Mahatva asks Codex to review Claude's.
- Every two weeks, one comparison table goes to sir: per cycle (O1–O3, M3) and real time (M1–M3).
- Amputee experiments are done by both: the healthy-trained model tested on amputees, and subject-specific models for the amputated and intact limbs.
- Writing: Ojaswi writes the labels and per-cycle results, Mahatva writes the real-time and deep-model results, and both write the discussion.

## Timeline (proposed, weeks start on Monday)

| Week | Ojaswi | Mahatva |
|---|---|---|
| 16 Nov | Step 0, O1 | Step 0 check, M1 |
| 23 Nov – 6 Dec | O2, O3 | M2, M3 (per cycle) |
| 7 – 20 Dec | O2/O3 final, O4 if sir has answered | M3 (real time) |
| 20 Dec | Comparison table to sir (both) | |
| 4 – 31 Jan | O5, amputee experiments | M4, amputee experiments |
| 14 Feb 2027 | Journal draft to sir (both) | |

## Still open with sir

1. Walking direction per trial: are alternate walks short → long and long → short?
2. Which recordings hold normal-step walking? Until they are found, the normal-step class is missing.
3. GI and GT definitions: from standing to the first contact, or including the first step(s)?
4. Are the band and insole on both sides, and does channel 1–8 equal S1–S8?
