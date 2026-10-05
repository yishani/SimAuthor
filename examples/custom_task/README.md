# Author a simulator for your own task

SimAuthor needs three things from you: a name for what you want to simulate,
a small folder of real recordings, and an evaluator that says how the
simulated recordings differ from the real ones.

## 1. Prepare a small reference set

Put real recordings in one flat folder. A few dozen is enough; the paper used
37–96 per task.

| Signal | Format |
|---|---|
| audio | mono `.wav`, any sample rate (scored at 16 kHz) |
| ECG | `.npy`, shape `(5000, 2)`: leads I and II, 10 s at 500 Hz |
| PPG | `.npy`, shape `(3750,)`: 30 s at 125 Hz |

## 2. Write an evaluator

Copy [`my_evaluator.py`](my_evaluator.py) and edit `FEATURES`. An evaluator
returns a **score** in [0, 1], which decides which program is revised next,
and a **report** in markdown, which tells the model what to change. Reports
that name concrete, measurable differences work best.

Check it by comparing the reference set with itself; the score should be
close to 1:

```bash
simauthor evaluate --modality audio --ref data/my_ref --gen data/my_ref --evaluator my_evaluator.py
```

The built-in evaluators (`mfcc` for audio, `morphology` for PPG,
`ecg_founder_12lead` for ECG) can be used instead with `--feedback-agent`.

## 3. Run

```bash
export GEMINI_API_KEY=...          # or --model deepseek-v4-pro / any OpenAI-compatible model
simauthor run --condition "Mitral regurgitation" --modality audio \
    --ref data/my_ref --evaluator my_evaluator.py --out runs/mr --iterations 100
```

`runs/mr/initialization/` holds the scientific blueprint and the root program;
`runs/mr/run/` holds the search tree, every candidate program, the evaluator
reports and the mechanism library. The best program is printed at the end.

## Dry run without data or an API key

```bash
simauthor generate VSD -n 20 --out data/toy_ref     # stand-in reference set
simauthor run --condition "VSD" --modality audio --ref data/toy_ref \
    --evaluator my_evaluator.py --out runs/dry --model mock --iterations 3 --num-samples 8
```

The `mock` model returns canned programs, so this only checks the plumbing.

## Other signal types

Generated programs must write audio, two-lead ECG or PPG files in the formats
above. For a different signal, add a modality specification next to
`simauthor/prompts/initialization/*/modalities/audio.md` and a matching preset
in `simauthor/presets.py`.
