"""Canned responses for :class:`simauthor.model_client.MockClient`.

The mock backend lets the full harness (initialization → search → mechanism
extraction) run offline.  It recognises the prompt type from its template
header and returns a blueprint, a small valid simulator that honours the
output contract stated in the prompt, or a mechanism JSON.
"""

import json
import re

_BLUEPRINT = """# Scientific Blueprint (mock)

## 1. Condition Overview
Offline placeholder blueprint produced by the mock model backend.

## 2. Signal Manifestation
A quasi-periodic carrier with condition-dependent noise bursts.

## 3. Severity and Variability
Severity scales burst amplitude and rate.

## 4. Key Discriminators
Burst rate, spectral centroid.

## 5. Synthesis Considerations
Sum of a periodic component and band-limited noise.

## 7. Reference Values
None (mock).
"""

_SIM_TEMPLATE = '''```python
"""Mock simulator (variant {variant}) generated offline by MockClient."""
import os
import numpy as np

OUTPUT_DIR = "{output_dir}"
N_SAMPLES = {n}
SEED = 42
DURATION = {duration}
SR = {sr}
MODALITY = "{modality}"
BURST_GAIN = {gain}
RATE_HZ = {rate}


def synth_one(rng):
    t = np.arange(int(DURATION * SR)) / SR
    rate = RATE_HZ * rng.uniform(0.8, 1.2)
    carrier = np.sin(2 * np.pi * rate * t) ** 15
    noise = rng.standard_normal(t.size) * BURST_GAIN * rng.uniform(0.5, 1.5)
    x = carrier + noise
    return x / (np.max(np.abs(x)) + 1e-9)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(N_SAMPLES):
        x = synth_one(rng)
        if MODALITY == "audio":
            from scipy.io import wavfile
            wavfile.write(os.path.join(OUTPUT_DIR, f"mock_{{i:03d}}.wav"),
                          SR, x.astype(np.float32))
        elif MODALITY == "ecg":
            np.save(os.path.join(OUTPUT_DIR, f"mock_{{i:03d}}.npy"),
                    np.stack([x, 1.5 * x], axis=1).astype(np.float32))
        else:
            np.save(os.path.join(OUTPUT_DIR, f"mock_{{i:03d}}.npy"),
                    x.astype(np.float32))


if __name__ == "__main__":
    main()
```'''


def _find(pattern, text, default):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def _simulator(prompt: str, call: int) -> str:
    if "File format: WAV" in prompt:
        modality, sr = "audio", int(_find(r"resample to (\d+) Hz", prompt, 16000))
    else:
        modality = "ecg" if "two leads" in prompt else "ppg"
        sr = int(_find(r"Sampling rate: (\d+) Hz", prompt, 500))
    return _SIM_TEMPLATE.format(
        variant=call,
        output_dir=_find(r"Save samples to: `([^`]*?)/?`", prompt, "generated"),
        n=int(_find(r"generate exactly \**(\d+)", prompt, 10)),
        duration=float(_find(r"Duration per sample: ([\d.]+)", prompt, 10.0)),
        sr=sr,
        modality=modality,
        gain=round(0.05 + 0.1 * (call % 5), 3),
        rate=round(1.0 + 0.2 * (call % 3), 2),
    )


def respond(prompt: str, call: int) -> str:
    if prompt.startswith("# Scientific Blueprint"):
        return _BLUEPRINT
    if prompt.startswith("Compare two versions"):
        return json.dumps({
            "name": f"mock_burst_gain_change_{call}",
            "mechanism": "Adjusts noise burst gain (mock extraction).",
            "code_snippet": "BURST_GAIN = ...",
        })
    return _simulator(prompt, call)
