# Released simulators

For each task: `root.py` is the zero-shot root S(0) and `authored.py` is the
best program of the median-seed run (the run behind the median in Table 1).
They are copied verbatim from `experiments/main` except for `OUTPUT_DIR`, and
need only NumPy and SciPy. They are installed with the package.

| Task | Signal | Output | Source | Score root → authored |
|---|---|---|---|---|
| VSD | heart sounds, ventricular septal defect | 100 × WAV, 10 s, 16 kHz | seed1 #89 | .258 → .587 |
| AS | heart sounds, aortic stenosis | 100 × WAV, 10 s, 16 kHz | seed3 #31 | .139 → .557 |
| COPD | lung sounds | 100 × WAV, 10 s, 16 kHz | seed1 #96 | .165 → .672 |
| AF | PPG, atrial fibrillation | 100 × NPY (3750,), 30 s, 125 Hz | seed2 #26 | .216 → .738 |
| LQT | ECG leads I/II, long-QT syndrome | 100 × NPY (5000, 2), 10 s, 500 Hz | seed2 #53 | .286 → .568 |
| WPW | ECG leads I/II, Wolff–Parkinson–White | 100 × NPY (5000, 2), 10 s, 500 Hz | seed1 #11 | .364 → .581 |

```bash
simauthor generate COPD                    # -> samples/COPD_authored/
simauthor generate LQT --which root -n 10  # 10 records from S(0)
simauthor generate AF --out my_af_samples  # choose the output folder
```

Root programs predate the evaluator's resampling step for some tasks (for
example the COPD root writes 20 s at 44.1 kHz); the evaluator resamples, so
scores are unaffected. Scores are search scores against the paper's search
reference sets, not measures of physiological fidelity.
