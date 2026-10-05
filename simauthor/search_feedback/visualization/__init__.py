"""
Visualization submodule — optional multimodal visual feedback for the Refiner.

Renders a single real-vs-generated comparison figure per modality. Enabled
only when explicitly configured (default OFF, preserving current text-only
behavior).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

_SUPPORTED = frozenset({"audio", "ecg", "ppg"})


def select_samples(files: List[str], n: int, seed: int) -> List[str]:
    """Deterministically select *n* samples from a sorted file list."""
    files = sorted(files)
    if n >= len(files):
        return files
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=n, replace=False))
    return [files[i] for i in idx]


def create_visual_feedback(modality: str,
                           real_dir: str,
                           generated_dir: str,
                           output_path: str,
                           n_samples: int = 3,
                           seed: int = 42,
                           ) -> dict:
    """Render a real-vs-generated comparison figure for *modality*.

    Returns a metadata dict (modality, plot_type, real_samples,
    generated_samples, seed, output_path, created_at). Raises ValueError for
    unsupported modalities.
    """
    if modality not in _SUPPORTED:
        raise ValueError(f"Unsupported modality for visual feedback: {modality!r}")

    real_files = select_samples(
        [f for f in os.listdir(real_dir) if not f.startswith('.')],
        n_samples, seed)
    gen_files = select_samples(
        [f for f in os.listdir(generated_dir) if not f.startswith('.')],
        n_samples, seed)

    if modality == "audio":
        from .audio import audio_visual_feedback
        plot_type = "mel_spectrogram"
        audio_visual_feedback(real_dir, generated_dir,
                              real_files, gen_files, output_path)
    elif modality == "ecg":
        from .ecg import ecg_visual_feedback
        plot_type = "waveform"
        ecg_visual_feedback(real_dir, generated_dir,
                            real_files, gen_files, output_path)
    elif modality == "ppg":
        from .ppg import ppg_visual_feedback
        plot_type = "waveform"
        ppg_visual_feedback(real_dir, generated_dir,
                            real_files, gen_files, output_path)

    metadata = {
        "modality": modality,
        "plot_type": plot_type,
        "real_samples": real_files,
        "generated_samples": gen_files,
        "seed": seed,
        "output_path": output_path,
        "created_at": datetime.now().isoformat(),
    }
    return metadata


def save_metadata(metadata: dict, metadata_path: str):
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


__all__ = ["create_visual_feedback", "save_metadata", "select_samples"]
