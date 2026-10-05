# New evaluator: independent encoders

`results.json` holds the endpoint numbers of Table 2a: the program selected by
the search score (`r_sel`) and the best program in its pool (`r_max`), scored in
encoders never used during search: OPERA-CE (COLA) for audio, PaPaGei-S for
PPG.

`code/eval_audio_cola_pools.py` is the audio evaluator. It needs

* a clone of [OPERA](https://github.com/evelyn0414/OPERA) with
  `checkpoints/audio_encoder_only.pt` (`export OPERA_ROOT=/path/to/OPERA`),
* the search reference sets as `$SIMAUTHOR_REF_ROOT/<TASK>/search_ref/*.wav`,
* generated candidate pools as
  `$SIMAUTHOR_POOLS/<TASK>/<method>/candidate_scores.csv` and
  `$SIMAUTHOR_POOLS/<TASK>/<method>/generated/candidate_XXX_*/` (regenerate with
  `simauthor generate experiments/main/<TASK>/seed*/candidates/script_XXX.py`).

The PPG side uses the `papagei` evaluator in
`simauthor/search_feedback/representation/ppg/papagei.py`.
