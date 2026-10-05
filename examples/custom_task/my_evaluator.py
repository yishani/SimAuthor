"""Template evaluator for your own task.

An evaluator compares a folder of real recordings with a folder of simulator
outputs and returns two things:

  * score  -- a number in [0, 1]; the search uses it to decide what to revise
  * report -- markdown text; the refiner reads it to decide *how* to revise

This example computes a few per-record audio features with NumPy/SciPy and
scores how close the generated feature distributions are to the real ones.
Replace FEATURES with whatever matters for your signal.

Use it with:
  simauthor evaluate --modality audio --ref data/my_ref --gen samples/ --evaluator my_evaluator.py
  simauthor run --condition "..." --modality audio --ref data/my_ref --evaluator my_evaluator.py --out runs/my_task
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from simauthor.search_feedback import SearchFeedback, SearchFeedbackAgent

SR = 16000  # analysis rate


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return x / (np.max(np.abs(x)) + 1e-9)


def rms_envelope(x, win=0.05):
    n = int(win * SR)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return np.sqrt((frames ** 2).mean(axis=1))


FEATURES = {
    # name: function(record) -> float
    "RMS level": lambda x: float(np.sqrt(np.mean(x ** 2))),
    "Envelope variability": lambda x: float(np.std(rms_envelope(x)) / (np.mean(rms_envelope(x)) + 1e-9)),
    "Spectral centroid (Hz)": lambda x: float(
        np.sum(np.fft.rfftfreq(len(x), 1 / SR) * np.abs(np.fft.rfft(x))) / (np.sum(np.abs(np.fft.rfft(x))) + 1e-9)),
    "Zero-crossing rate": lambda x: float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
}


class MyEvaluator(SearchFeedbackAgent):
    agent_id = "my_evaluator"   # recorded in tree.json
    modality = "audio"

    def features(self, folder):
        files = sorted(Path(folder).glob("*.wav"))
        return np.array([[f(load(p)) for f in FEATURES.values()] for p in files])

    def analyze(self, reference_dir, generated_dir, *, include_report=True):
        real, gen = self.features(reference_dir), self.features(generated_dir)
        if len(gen) < 3:
            return SearchFeedback(score=0.0, report="Fewer than 3 generated records.",
                                  metadata={"n_generated": len(gen)})
        mu_r, mu_g = real.mean(0), gen.mean(0)
        sd = np.sqrt((real.var(0) + gen.var(0)) / 2) + 1e-9
        similarity = np.exp(-np.abs(mu_g - mu_r) / sd)      # 1 = identical means
        score = float(similarity.mean())

        rows = "\n".join(
            f"| {name} | {r:.4g} | {g:.4g} | {(g - r) / s:+.2f} σ | {q:.3f} |"
            for name, r, g, s, q in zip(FEATURES, mu_r, mu_g, sd, similarity))
        report = (f"## Discrepancy report\n\nReference: {len(real)} files | Generated: {len(gen)} files\n"
                  f"Overall score: {score:.4f}\n\n"
                  "| Feature | Reference mean | Generated mean | Shift | Similarity |\n"
                  "|---|---|---|---|---|\n" + rows)
        return SearchFeedback(score=score, summary=f"score {score:.3f}", report=report,
                              metadata={"n_generated": len(gen)})
