# Lineage step labelling instructions (audit label pass)

You are labelling the parent→child revisions on the **best lineage** of Full
SimAuthor runs. Work strictly from the files listed below; do not invent code.

## Files you will use
- The run JSON step-records (one per seed):
  `Results/lineage_audit/work/<COND>_seed<k>_steps.json`
  Every element of `steps` is one lineage revision and carries mechanical
  facts already extracted: `parent_script`/`child_script`, `parent_score`/
  `child_score`/`delta`, `attempt`, `param_only`, `structural_add_lines`,
  `structural_del_lines`, `n_param_pairs`, `mechanisms_triggered`,
  `visible_mechanisms`, `parent_report` / `parent_report_compact`.
- Per-step context diffs (2 lines of context):
  `Results/lineage_audit/work/diffs/<COND>_<run>_<child_script:03d>.txt`
  The parent and child scripts are at
  `<run_dir>/candidates/script_<idx:03d>.py` if you ever need more context.

## What to decide for each step
1. **category** — one primary structural category of the *behavioural change*:
   - `mechanism` — new/replaced/removed signal-generation component or physical
     modelling rule (new noise colour, added murmur/heart-sound component, an
     added resonance/morphology rule like a notch or split, a coupling of one
     component to another, an envelope *shape* family change, a new transfer
     function or generator). New code that *defines* a generator/coupling.
   - `temporal` — timing/rhythm/duration/schedule changes: HR & RR intervals,
     systolic ratio, onset/duration of components, QT/ST/PR timing, envelope
     length (but a plain number tweak to an existing constant is
     `calibration`).
   - `heterogeneity` — adds or widens *between-subject / across-sample*
     diversity: fixed scalar → per-sample draw, widened `uniform` ranges for
     population spread, added subtype/variant distribution, severity-based
     branching that makes subjects differ.
   - `stochasticity` — randomness *magnitude/process* not tied to subject
     structure: noise amplitude/SNR, jitter magnitude, number of random draws,
     random phase, adding randomness where there was none.
   - `recording process` — the sensor→file pipeline: output sample rate /
     resampling, output low-pass or quantisation, normalisation, baseline
     wander / mains hum / ambient recording noise, filename or sample-count
     contract (number of generated files).  Room/background noise is
     recording process; the murmur's own spectral content is `mechanism`.
   - `calibration` — essentially pure scalar/numeric tuning (amplitude,
     frequency, coefficient, single range-endpoint values) with no new logic.
   - `other` — boilerplate, refactors, docstring/comment-only, imports,
     fixes that change no physiology.
   You may add a secondary tag (a second category that is genuinely present,
   comma-free; use `|` to join several). Do not over-tag: only if the change
   is substantial in that dimension too.
2. **is_param_tuning** — `true` iff the *behavioural* change is only scalar
   value tuning (numbers and comments), no new/removed logic, function,
   branch, or coupling.  Structural_add/del that differ only in numeric
   literals ⇒ true.  Any real structural change ⇒ false.
3. **added_summary / removed_summary** — the *smallest meaningful semantic
   change*, 1–2 sentences total, describing what generation behaviour was
   added and what was removed/replaced.  Skip cosmetic/comment noise.
4. **top_added_lines / top_removed_lines** — the 3–10 *most informative*
   actual code lines from the diff (the ones that carry the semantics; pick
   from the diff file).  No leading `+`/`-`; drop indentation-only
   differences; strip trailing whitespace.
5. **param_changes** — only if there were scalar/parameter edits *alongside*
   the structural change (or alone): a compact `name=old -> new` list, e.g.
   `alpha=0.2->0.1; s1_dur=0.1->0.06; s2_cutoff=150->100`.  Leave empty if
   no numeric tuning happened.
6. **step_note** — one sentence on why this child plausibly scored better
   (tie to the discrepancy it addresses). Skip if a pure-tuning step.

## Mechanism correlation (per run)
Each run JSON also has a top-level `reuse` dict: for every mechanism born on
the lineage it lists `source_node`, `desc`, `code_snippet`, `delta`,
`in_later_prompt` (lineage nodes where the mechanism was literally in the
refiner's "Reusable Mechanisms" section) and `in_later_code` (token/verbatim
hits in later lineage scripts, with matching `lines` for you to judge).

For **every** mechanism in `reuse`, decide `code_reuse` = whether the later
lineage code *genuinely applies that mechanism* (verbatim snippet match is
strong; token matches are weak — treat as reuse only if the matched lines
clearly implement the same idea), and write a ≤1-line `code_note`.  A
mechanism listed with an empty `in_later_code` array is *not* reused in code.

## Output contract
Write ONE JSONL file `Results/lineage_audit/work/labels/<COND>.jsonl`
with one line per step AND one line per mechanism reuse verdict:

Step line:
```json
{"kind":"step","id":"<COND>_<run>_<child_script:03d>","category":"mechanism",
 "category_secondary":"calibration|stochasticity","is_param_tuning":false,
 "added_summary":"...","removed_summary":"...",
 "top_added_lines":["..."],"top_removed_lines":["..."],
 "param_changes":"alpha=0.2->0.1","step_note":"..."}
```
`category_secondary` = "" when none. `is_param_tuning` must be true/false.

Reuse line (one per mechanism present in a run's `reuse` dict):
```json
{"kind":"reuse","run":"<seed1>","mechanism":"<name>","source_node":11,
 "reuse_prompt":true,"prompt_nodes":[65,81,89],
 "code_reuse":true,"code_nodes":[89],"code_note":"..."}
```
`reuse_prompt`/`prompt_nodes` come straight from `in_later_prompt` in the run
JSON (copy them — do not reinterpret).  `code_nodes` = lineage nodes whose
code you judged to genuinely reuse it; [] if none.  If a mechanism is *not*
in a run's `reuse` dict, emit nothing for it.

Constraints: ids must match exactly; one step line per step in the three seed
files; one reuse line per mechanism per run in which it appears.  When done,
reply with a terse summary: counts per seed (steps, categories present,
param-tuning steps, mechanisms reused in code).
