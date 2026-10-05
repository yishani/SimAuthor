#!/usr/bin/env python3
"""Render the real / root / authored comparison at the top of the project page.

Each program is run with its own seed to generate ``--n`` records.  For every
real reference recording we take the root output and the authored output that
are closest to it in the evaluator's standardized MFCC+ZCR feature space, and
keep the recording where the authored output is much closer than the root
output.  The real recording is stored as a spectrogram and waveform envelope
only; no audio and no filename is written.

For each task the k recordings with the largest gap are kept.

Usage:
  python docs/_build/render_hero.py --refs VSD=/data/VSD/search_ref COPD=/data/COPD/search_ref \
      --keep VSD:2 COPD:3
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_generations import audio_item, envelope, mel_b64, DOCS  # noqa: E402
from simauthor.search_feedback.signal.audio.mfcc import MFCCFeedback  # noqa: E402


def generate(script: Path, n: int, out: Path):
    subprocess.run([sys.executable, "-m", "simauthor.cli", "generate", str(script), "-n", str(n),
                    "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    return sorted(out.glob("*.wav"))


TASKS = {  # task: (seed, best node, display name) -- the median-seed runs of Table 1
    "VSD": (1, 89, "Ventricular septal defect"),
    "COPD": (1, 96, "COPD"),
}


def examples_for(task, ref, n, k):
    seed, best, name = TASKS[task]
    run = REPO / "experiments" / "main" / task / f"seed{seed}"
    nodes = json.load(open(run / "tree.json"))["nodes"]
    fb = MFCCFeedback()
    R, _ = fb._load_features(ref)
    real_files = sorted(Path(ref).glob("*.wav"))
    mu, sd = R.mean(0), R.std(0) + 1e-8
    tmp = Path(tempfile.mkdtemp(prefix="simauthor_hero_"))
    feats, files = {}, {}
    for tag, idx in (("root", 0), ("best", best)):
        files[tag] = generate(run / "candidates" / f"script_{idx:03d}.py", n, tmp / tag)
        F, _ = fb._load_features(str(tmp / tag))
        feats[tag] = (F - mu) / sd
    Rn = (R - mu) / sd
    D = {t: np.linalg.norm(feats[t][:, None, :] - Rn[None], axis=2) for t in feats}  # (n_gen, n_ref)
    picked, used = [], {"root": set(), "best": set()}
    left = set(range(len(Rn)))
    while left and len(picked) < k:
        # nearest unused generated record of each program, per real recording
        def nearest(t, i):
            order = [g for g in np.argsort(D[t][:, i]) if g not in used[t]]
            return int(order[0]), float(D[t][order[0], i])
        scored = []
        for i in left:
            a, da = nearest("root", i); b, db = nearest("best", i)
            scored.append((da - db, i, a, b, da, db))
        gap, i, a, b, da, db = max(scored)
        if db >= da:
            break
        picked.append((gap, i, a, b, da, db)); used["root"].add(a); used["best"].add(b); left.discard(i)
    import librosa
    media = DOCS / "media" / "hero" / task
    shutil.rmtree(media, ignore_errors=True)
    items = []
    for j, (_, i, a, b, d_root, d_best) in enumerate(picked):
        y, _ = librosa.load(str(real_files[i]), sr=16000, mono=True, duration=10.0)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
        items.append({
            "real": {"env": envelope(y), "mel": mel_b64(y)},
            "root": audio_item(files["root"][a], media / f"root_{j}.mp3"),
            "authored": audio_item(files["best"][b], media / f"authored_{j}.mp3"),
            "distance": {"root": round(float(d_root), 2), "authored": round(float(d_best), 2)},
        })
        print(f"  {task} example {j + 1}: distance root {d_root:.2f}, authored {d_best:.2f}")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"task": task, "name": name, "seed": seed, "best": best, "n_ref": len(real_files),
            "root_score": round(nodes[0]["score"], 3), "best_score": round(nodes[best]["score"], 3),
            "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, metavar="TASK=DIR",
                    help="search reference directory per task, e.g. VSD=/data/VSD/search_ref")
    ap.add_argument("--n", type=int, default=20, help="records generated per program")
    ap.add_argument("--k", type=int, default=3, help="candidate examples per task")
    ap.add_argument("--keep", nargs="*", default=[], metavar="TASK:N",
                    help="publish only candidate N (1-based) of a task, e.g. VSD:2 COPD:3")
    args = ap.parse_args()
    shutil.rmtree(DOCS / "media" / "hero", ignore_errors=True)
    examples = []
    for spec in args.refs:
        task, ref = spec.split("=", 1)
        examples.append(examples_for(task, ref, args.n, args.k))
    path = DOCS / "data" / "generations.json"
    data = json.load(open(path))
    keep = {t["key"] for t in json.load(open(DOCS / "data" / "index.json"))}
    data = {k: v for k, v in data.items() if k in keep}
    for spec in args.keep:
        task, n = spec.split(":")
        for ex in examples:
            if ex["task"] == task:
                it = ex["items"][int(n) - 1]
                for k in ("root", "authored"):
                    src = DOCS / it[k]["audio"]
                    dst = src.with_name(f"{k}.mp3")
                    shutil.copy(src, dst)
                    it[k]["audio"] = str(dst.relative_to(DOCS))
                it.pop("distance", None)
                ex["items"] = [it]
    for f in (DOCS / "media" / "hero").rglob("*_[0-9].mp3"):
        f.unlink()
    data["_hero"] = {"examples": examples}
    path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {sum(len(e['items']) for e in examples)} hero examples to {path}")


if __name__ == "__main__":
    main()
