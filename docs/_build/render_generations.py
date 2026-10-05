#!/usr/bin/env python3
"""Render synthetic outputs along each task's best lineage for the project page.

For the released (median-seed) run of every task, each program on the path
root -> ... -> best is executed from ``experiments/main`` with a fixed seed.
Only synthetic signals are written — no reference recording is touched.

Outputs:
  docs/media/<TASK>/node_XXX_sK.mp3   (audio tasks, 16 kHz mono)
  docs/data/generations.json          waveforms / mel images / ECG + PPG traces

The hero comparison on the page also shows real COPD recordings as mel
spectrograms and waveform envelopes only (no audio).  Pass the three files used
in the paper's comparison figure with ``--hero-real``; filenames are not stored.

Usage:  python docs/_build/render_generations.py [--samples 8] [--only COPD]
                                             [--hero-real a.wav b.wav c.wav]
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from simauthor.utils import patch_output_dir, patch_sample_count  # noqa: E402

DOCS = REPO / "docs"
KEEP_END = 3      # samples kept for the root and the best program
KEEP_MID = 1      # samples kept for intermediate lineage programs


def run_program(src: Path, n: int, timeout: int = 600) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="simauthor_render_"))
    out = tmp / "out"
    script = tmp / "prog.py"
    shutil.copy(src, script)
    patch_output_dir(str(script), str(out))
    patch_sample_count(str(script), n)
    r = subprocess.run(["nice", "-n", "10", sys.executable, str(script)], cwd=tmp,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{src}: {r.stderr[-800:]}")
    return out


def mel_b64(y, sr=16000, n_mels=64, frames=240):
    import librosa
    hop = max(1, len(y) // frames)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop, n_fft=1024)
    db = librosa.power_to_db(m, ref=np.max, top_db=80.0)[:, :frames]
    q = np.clip((db + 80.0) / 80.0 * 255, 0, 255).astype(np.uint8)[::-1]  # high freq first
    return {"w": int(q.shape[1]), "h": int(q.shape[0]),
            "b64": base64.b64encode(q.tobytes()).decode()}


def envelope(y, bins=600):
    y = y[: len(y) - len(y) % bins] if len(y) >= bins else np.pad(y, (0, bins - len(y)))
    seg = y.reshape(bins, -1)
    return [round(float(v), 3) for v in np.stack([seg.min(1), seg.max(1)], 1).ravel()]


def audio_item(path: Path, dst: Path):
    import librosa
    import soundfile as sf
    y, _ = librosa.load(str(path), sr=16000, mono=True, duration=10.0)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, 16000, format="MP3")
    return {"audio": str(dst.relative_to(DOCS)), "env": envelope(y), "mel": mel_b64(y)}


def ecg_item(path: Path):
    x = np.load(path)
    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.T
    x = x[:5000:2]                                      # 500 Hz -> 250 Hz
    return {"sr": 250, "leads": [[round(float(v), 3) for v in x[:, k]] for k in range(2)]}


def ppg_item(path: Path):
    x = np.asarray(np.load(path), dtype=float).ravel()[:3750]
    x = x / (np.max(np.abs(x)) + 1e-9)
    return {"sr": 125, "sig": [round(float(v), 3) for v in x]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8, help="records generated per program")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--hero-real", nargs="*", help="real COPD recordings for the hero (visual only)")
    args = ap.parse_args()
    index = json.load(open(DOCS / "data" / "index.json"))
    out_path = DOCS / "data" / "generations.json"
    result = json.load(open(out_path)) if out_path.exists() else {}
    if args.hero_real:
        import librosa
        real = []
        for f in args.hero_real:
            y, _ = librosa.load(f, sr=16000, mono=True, duration=10.0)
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
            real.append({"env": envelope(y), "mel": mel_b64(y)})
        result["_hero_real"] = {"task": "COPD", "dataset": "ICBHI 2017", "samples": real}
        print(f"  hero: {len(real)} real recordings (visual only)")
        if args.only == []:
            args.only = ["__none__"]
    for task in index:
        key = task["key"]
        if args.only and key not in args.only:
            continue
        run = json.load(open(DOCS / "data" / "runs" / f"{key}.json"))
        cdir = REPO / "experiments" / "main" / key / f"seed{task['seed']}" / "candidates"
        media = DOCS / "media" / key
        if media.exists():
            shutil.rmtree(media)
        nodes = {}
        lin = task["lineage"]
        for pos, nid in enumerate(lin):
            keep = KEEP_END if pos in (0, len(lin) - 1) else KEEP_MID
            try:
                out = run_program(cdir / f"script_{nid:03d}.py", args.samples)
            except Exception as e:
                print(f"  {key} node {nid}: FAILED {e}")
                continue
            ext = ".wav" if task["modality"] == "audio" else ".npy"
            files = sorted(p for p in out.rglob(f"*{ext}"))[:keep]
            items = []
            for k, f in enumerate(files):
                if task["modality"] == "audio":
                    items.append(audio_item(f, media / f"node_{nid:03d}_s{k}.mp3"))
                elif task["modality"] == "ecg":
                    items.append(ecg_item(f))
                else:
                    items.append(ppg_item(f))
            shutil.rmtree(out.parent, ignore_errors=True)
            nodes[str(nid)] = {"score": run["nodes"][nid]["score"], "samples": items}
            print(f"  {key} node {nid:3d}: {len(items)} sample(s)")
        result[key] = {"modality": task["modality"], "lineage": lin, "nodes": nodes}
    out_path.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
