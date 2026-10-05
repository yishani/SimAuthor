# Generalization

Three tests, each changing one thing relative to the search setting.

| Test | Tasks | Where |
|---|---|---|
| New recordings: the search evaluator on held-out recordings (Table 4) | all six | `simauthor evaluate` on the held-out sets; datasets are not redistributed |
| New evaluator: encoders never used during search (Table 2a) | audio and PPG; ECG search already uses ECGFounder | [`new_evaluator/`](new_evaluator) |
| New use: ECG classification with synthetic positives (Table 2b) | LQT and WPW | [`ecg_classification/`](ecg_classification) |

Stored endpoint results are checked by `python experiments/reproduce.py`.
Re-running the experiments requires the external datasets and encoder
checkpoints described in each folder.
