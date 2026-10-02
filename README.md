# SimAuthor supplementary code and audit artifacts

This directory contains the compact code-and-trace release accompanying the
anonymous submission. It is intended to make the authoring harness, the main
search trajectories, and the paper's validation analyses inspectable without
shipping reference recordings, generated waveform pools, model weights, or
cluster logs.

## Start here

- Open `interactive_example/index.html` locally for a step-by-step COPD run.
- Read `Authoring/` for the search harness, prompts, evaluators, visual
  feedback, and mechanism-library implementation.
- Inspect `experiments/core_runs/` for all programs produced by the 18 main
  runs (six conditions, three seeds).

## Contents

### `Authoring/`

Core implementation, prompt templates, signal-space evaluators, frozen
representation evaluators, visualization code, and command-line entry points.
Python bytecode and local caches are excluded.

### `experiments/core_runs/`

The 18 primary runs. Each run contains:

```text
tree.json
mechanism_library.json
candidates/script_*.py
```

VSD, AS, COPD, and AF use the signal-space search runs. LQT and WPW use the
final ECGFounder representation-space search runs. Candidate source programs
are included only for these core runs.

### `experiments/baselines/`

Primary three-seed trajectories for Score-only PUCT and Text-Opt.
Independent one-shot sampling is a flat pool rather than a tree, so its ordered
380-proposal score files are stored under `one_shot/`; the first 100 entries are
the pool used in the main comparison, and all 380 support the token-matched
appendix analysis. Candidate programs are intentionally omitted from baseline
traces.

### `experiments/component_ablations/`

Three-seed focused ablations of the discrepancy report (`no_report`) and
mechanism library (`no_mech_lib`) on all six conditions. Only `tree.json` is
included. The ECG conditions use representation-space search.

### `experiments/robustness/`

Final three-seed PUCT-sensitivity and DeepSeek-V4 Pro cross-backbone
trajectories. For each run, the search tree and mechanism library are retained;
candidate programs and local run configurations are omitted.

### `generalization/representation_evaluation/`

Code and compact endpoint results for the held-out representation
evaluation. Audio uses the COLA/OPERA-CE encoder, and PPG uses PaPaGei. Cached
embeddings and generated waveform pools are not included.

### `generalization/ecg_downstream_utility/`

Code, cohort metadata, classifier settings, and final five-classifier-seed
results for the scarce-label PTB-XL/Chapman experiment with frozen ECGFounder
representations. Dataset files, ECGFounder weights, and embedding caches are
not included.

### `analysis/`

The audit prompt and final labelled results used for the
structural-versus-parameter revision analysis. Intermediate extraction records,
model responses, and raw diffs are omitted because they can be regenerated from
the core candidates.

## Deliberate omissions

This preliminary bundle does not include `run_config.json`, non-core candidate
programs, patient/reference recordings, generated signal pools, learned-model
weights, embedding caches, virtual environments, API credentials, or Slurm
`.out`/`.err` files. External model checkpoints and data access instructions
will be documented separately.

Absolute paths in stored artifacts have been anonymized with placeholders such
as `[PROJECT_ROOT]`, `[DATA_ROOT]`, and `[OPERA_ROOT]`. Candidate programs are
preserved as authored audit artifacts; replace their output paths before
executing them in a new environment.
