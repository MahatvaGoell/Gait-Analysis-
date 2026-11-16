# Real-time baselines (M1)

From the repository root:

```text
python -m pip install scikit-learn pillow
python -m common.build_dataset
python -m unittest study1.realtime.test_run_realtime
python -m study1.realtime.run_realtime
```

The dataset builder also uses NumPy, SciPy and Pillow through the segmentation code. The run writes
`results/realtime_summary.csv`, per-subject metrics and confusion matrices. It compares 300 ms
LDA and histogram gradient boosting (HGB) with 200 ms LDA. It holds out every recording and both
limbs of one participant at a time. Balanced accuracy and macro-F1 are mean ± population SD
across held-out participants; recall in the summary is pooled across scored windows.

The eight thigh FMG channels are the only predictors. Trailing windows step by 20 ms; model
fitting subsamples at 60 ms. Windows with non-finite or imputed FMG are excluded. Calibration
walks are neither fitted nor scored. Feature scaling is fitted in each training fold. The
scored stream does not use per-trial centering based on annotated gait onset. These results
supersede the earlier calibration-scaled run; they are not a like-for-like comparison.

Event detection requires three consecutive predictions of the target class in its labeled
interval, with no missing window between them. Delay is time from the reference event to the
third prediction, in seconds. The median is over detected events; detection rate uses all
scored limb trials. These are recognition delays in offline replay, not hardware latency.

False switches are counted separately in each limb-trial stream: changes after the three-window
causal debounce minus reference changes across the same adjacent windows. Gaps do not count as
switches. The reported median is a signed net excess; a negative value means fewer switches
than the reference, not negative false detections. Window classification scores use the raw
predictions, independently of debounce.

The run uses provisional walking-direction labels. The recording interval is selected using
reference annotations for offline benchmarking; this is not a demonstrated live deployment.
