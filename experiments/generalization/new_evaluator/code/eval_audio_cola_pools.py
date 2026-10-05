#!/usr/bin/env python3
"""Evaluate audio candidate pools with the medical COLA encoder.

This is the evaluator used for the held-out audio representation results.
Generated waveform pools and the external OPERA checkpoint are not distributed
in this compact release.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from simauthor.search_feedback.representation.comparison import compare


# Generated candidate pools (one folder per task and method, see README) and the
# search reference sets are not distributed; point these variables at them.
SOURCE = Path(os.environ.get("SIMAUTHOR_POOLS", "pools"))
REF_ROOT = Path(os.environ.get("SIMAUTHOR_REF_ROOT", "data"))
OUT = Path(__file__).resolve().parents[1] / "work"
OPERA = Path(os.environ.get("OPERA_ROOT", "OPERA"))  # clone of github.com/evelyn0414/OPERA
CHECKPOINT = OPERA / "checkpoints" / "audio_encoder_only.pt"
SAMPLE_RATE = 16_000
AUDIO_SECONDS = 10
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

CONDITIONS = {
    "VSD": {
        "directory": "Ventricular_Septal_Defect",
        "reference": "VSD",
    },
    "AS": {
        "directory": "Aortic_Stenosis",
        "reference": "Aortic_Stenosis",
    },
    "COPD": {
        "directory": "COPD",
        "reference": "COPD",
    },
}
METHODS = ("one_shot", "text_op", "full")
TASKS = tuple((condition, method) for condition in CONDITIONS for method in METHODS)


def _mel_from_path(path: str):
    try:
        import librosa
        waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
        target = AUDIO_SECONDS * SAMPLE_RATE
        if len(waveform) > target:
            start = (len(waveform) - target) // 2
            waveform = waveform[start:start + target]
        else:
            waveform = np.pad(waveform, (0, target - len(waveform)))
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS,
            fmin=50, fmax=8000, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        frames = target // HOP_LENGTH
        if mel.shape[1] < frames:
            mel = np.pad(mel, ((0, 0), (0, frames - mel.shape[1])))
        else:
            mel = mel[:, :frames]
        return mel.astype(np.float32)
    except Exception:
        return None


def _load_encoder(device):
    import torch
    sys.path.insert(0, str(OPERA))
    from opera.src.model.models_cola import Cola
    model = Cola(encoder="efficientnet")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    result = model.load_state_dict(
        checkpoint["audio_encoder_state_dict"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"COLA checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}")
    encoder = model.encoder.to(device=device, dtype=torch.float32)
    encoder.eval()
    return encoder


def _embed(directory: Path, cache: Path, encoder, device, workers) -> np.ndarray:
    import torch
    if cache.is_file():
        return np.load(cache, allow_pickle=False)
    paths = sorted(str(path) for path in directory.rglob("*.wav"))
    if not paths:
        return np.empty((0, 1280), dtype=np.float32)
    mels = list(workers.map(_mel_from_path, paths, chunksize=4))
    mels = [mel for mel in mels if mel is not None]
    outputs = []
    for start in range(0, len(mels), 32):
        batch = torch.from_numpy(np.stack(mels[start:start + 32])).to(device)
        with torch.no_grad():
            outputs.append(encoder(batch).cpu().numpy())
    embeddings = (np.concatenate(outputs).astype(np.float32)
                  if outputs else np.empty((0, 1280), dtype=np.float32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    return embeddings


def _candidate_directory(condition: str, method: str, index: int) -> Path:
    matches = sorted((SOURCE / condition / method / "generated").glob(
        f"candidate_{index:03d}_*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one generated directory for {condition}/{method}/{index}, "
            f"found {matches}")
    return matches[0]


def evaluate(condition: str, method: str):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_out = OUT / condition / method
    task_out.mkdir(parents=True, exist_ok=True)
    source_rows = list(csv.DictReader(
        (SOURCE / condition / method / "candidate_scores.csv").open()))
    encoder = _load_encoder(device)
    reference_dir = REF_ROOT / condition / "search_ref"

    rows = []
    with ProcessPoolExecutor(max_workers=8) as workers:
        reference = _embed(
            reference_dir, task_out / "embeddings" / "reference.npy",
            encoder, device, workers)
        for done, source in enumerate(source_rows, 1):
            index = int(source["index"])
            directory = _candidate_directory(condition, method, index)
            embedding = _embed(
                directory,
                task_out / "embeddings" / f"candidate_{index:03d}.npy",
                encoder, device, workers)
            if len(embedding) < 3:
                continue
            stats = compare(reference, embedding)
            rows.append({
                "condition": condition, "method": method,
                "seed": source["seed"], "index": index,
                "signal_score": float(source["signal_score"]),
                "representation_score": float(stats["score"]),
                "energy_distance": float(stats["energy_distance"]),
                "n_embedded": len(embedding),
            })
            if done % 10 == 0 or done == len(source_rows):
                print(f"[{condition}/{method}] COLA {done}/{len(source_rows)}",
                      flush=True)

    if not rows:
        raise RuntimeError(f"No scorable candidates for {condition}/{method}")
    selected = max(rows, key=lambda row: row["signal_score"])
    rep_best = max(rows, key=lambda row: row["representation_score"])
    result = {
        "condition": condition, "method": method,
        "encoder": "OPERA-COLA-EfficientNet",
        "checkpoint": str(CHECKPOINT), "reference": "search_ref",
        "aggregation": "L2-normalized unbiased energy distance",
        "n_candidates": len(rows),
        "signal_selected_index": selected["index"],
        "r_sel": selected["representation_score"],
        "representation_best_index": rep_best["index"],
        "r_max": rep_best["representation_score"],
    }
    with (task_out / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (task_out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    condition, method = TASKS[args.task_id]
    evaluate(condition, method)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
