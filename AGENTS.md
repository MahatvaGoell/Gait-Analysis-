# Rules for coding agents (Codex and Claude)

Two people work on this repository at the same time: Ojaswi with Claude Code and Mahatva with Codex. The task list and the owner of each file are in `TRAINING_PLAN.md`; read it before starting a task.

## Git
- Only the `main` branch exists. Run `git pull --rebase` before starting and before every push. Commit small and often.
- Change only the files your person owns (see `TRAINING_PLAN.md`); ask before touching the other person's files.
- Never commit `datasets/` (rebuilt by `python common/build_dataset.py`) or `NewSegmentedGraphsFinal/`.

## Data and evaluation (Study 1)
- Model input is thigh FMG only. CoP and vGRF make the labels and are never a model input.
- Labels come from `common/labels.py` and splits and metrics from `common/evaluate.py`; do not write your own.
- The main result is leave-one-subject-out (LOSO); the second is within-subject with whole trials held out.
- Scaling, feature selection and hyper-parameters are fitted on the training part only, never on the held-out subject.
- Windows that contain estimated (`*_imputed`) FMG samples are left out. Calibration walks are never scored.
- Report balanced accuracy and macro-F1 as mean ± SD over subjects, per-class recall and the confusion matrix. Use fixed seeds.
- Results are CSV files in `results/` written by the script, with a prefix per track (`cycle_`, `realtime_`, `deep_`). Never edit result files by hand and never report a number that a run did not produce.
- Labels are provisional until the supervisor confirms the walking direction; say so when reporting.
