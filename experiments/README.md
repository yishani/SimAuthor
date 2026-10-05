# Experiments

Everything behind the paper's results, organised like the paper. Every run
uses 100 authoring attempts, 100 generated records per evaluation and
c_puct = 0.25 unless stated otherwise.

```bash
python experiments/reproduce.py    # recompute Tables 1, 2, 3, 5 and the audit; compare with the paper
```

| Folder | Paper | Contents per run |
|---|---|---|
| `main/<TASK>/seed{1,2,3}` | Table 1, Fig. 3: SimAuthor (Gemini 3.1 Pro) | `tree.json`, `mechanism_library.json`, `candidates/script_*.py` (every program) |
| `main/<TASK>/scientific_blueprint.md` | Appendix: the scientific blueprint written before search, shown to the refiner in all three seeds | one per task |
| `baselines/puct_score_search/` | Table 1: same search, refiner sees only the score | `tree.json` |
| `baselines/text_opt/` | Table 1: textual strategy optimization | `tree.json` (with `text_strategy` per node) |
| `baselines/sampling/<TASK>/results.csv` | Table 1, token-matched analysis: 380 independent proposals; the first 100 are best-of-100 | one row per proposal |
| `ablations/no_report/`, `ablations/no_library/` | Table 3: − Report, − Library | `tree.json` |
| `robustness/c_puct_0.10/`, `c_puct_0.50/` | Table 5: exploration constant | `tree.json`, `mechanism_library.json` |
| `robustness/deepseek_v4_pro/` | Table 5: DeepSeek-V4 Pro backbone | `tree.json`, `mechanism_library.json` |
| `generalization/` | Table 2: independent evaluators and ECG classification | code and stored results |
| `audit/` | structural vs parameter revisions (111/138) | prompt and labelled revisions |

Evaluators: VSD, AS and COPD use MFCC/ZCR overlap; AF uses PPG pulse
morphology; LQT and WPW use ECGFounder energy distance (recorded as island
`ecg_founder`). For the sampling baseline the relevant column is `rep_score`
for ECG tasks and `signal_score` otherwise.

## `tree.json`

```json
{
  "condition": "COPD", "signal_modality": "audio", "island": "mfcc",
  "num_samples": 100, "search_algorithm": "era_rank_flat_puct",
  "nodes": [
    {"script_idx": 0, "score": 0.1650, "visits": 100, "parent": null,
     "report": "## Audio Signal-Space Discrepancy Report ...",
     "visual_feedback_path": "[PROJECT_ROOT]/.../visual/node_000.png"},
    ...
  ]
}
```

* Node `i` is `candidates/script_{i:03d}.py`; node 0 is the root S(0).
* Nodes are in archive order. Attempts that produced no valid program are not
  archived, so a run can have fewer than 101 nodes.
* `report` is the discrepancy report the refiner saw when revising that node.
* `[PROJECT_ROOT]` replaces machine-specific paths. Comparison figures are not
  included because they plot reference recordings.

## `mechanism_library.json`

A list of `{name, mechanism, code_snippet, source_node, delta,
observation_count, island}`; `source_node` is the improved child the entry was
extracted from and `delta` its gain over the parent.

## Running a stored program

```bash
simauthor generate experiments/main/COPD/seed1/candidates/script_041.py --out samples/copd_41
```

`generate` rewrites the program's output folder, so any stored program can be
run as is.
