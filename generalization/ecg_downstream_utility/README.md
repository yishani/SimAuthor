# ECG downstream utility

This directory contains the code and reported outputs for the frozen-ECGFounder
downstream experiment. The final simulators are selected by their
representation-search score:

- LQT: seed 3, node 88 (`experiments/core_runs/LQT/seed3/candidates/script_088.py`)
- WPW: seed 3, node 72 (`experiments/core_runs/WPW/seed3/candidates/script_072.py`)

Each experiment adds 300 synthetic positive examples to the real training set,
fits a class-weighted logistic regression on frozen ECGFounder features, and
evaluates on patient-disjoint PTB-XL (ID) and Chapman (OOD) cohorts. The five
classifier seeds are `0,1,2,3,4`.

Set `SIMAUTHOR_ECG_ROOT` to a directory containing the prepared `NORM`,
`LNGQT`, and `WPW` partitions, `data/ptbxl_database.csv`, the Chapman source
headers, and `ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth`. Then run from
the release root:

```bash
export SIMAUTHOR_ECG_ROOT=/path/to/prepared/ECG
python generalization/ecg_downstream_utility/code/run_lqt.py
python generalization/ecg_downstream_utility/code/run_wpw.py
```

Generated samples, embedding caches, and new outputs are written under
`generalization/ecg_downstream_utility/work` by default. Set
`SIMAUTHOR_ECG_WORKDIR` to override that location. The immutable outputs used
in the paper are retained in `results/`.
