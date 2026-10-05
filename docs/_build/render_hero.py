#!/usr/bin/env python3
"""Render the real / root / authored comparison at the top of the project page.

For each task, the root program and the best program of the median-seed run
each generate ``--n`` records with their own seeds.  For every real reference
recording we take the root output and the authored output closest to it in a
standardized feature space, and keep the recordings where the authored output
is much closer than the root output (``--k`` candidates per task).

Feature spaces: audio uses the search evaluator's MFCC+ZCR features, PPG the
search evaluator's pulse statistics, and ECG a set of simple waveform
statistics (heart rate, amplitude distribution, spectral shape).

Real recordings are published as images only (spectrograms for audio, plotted
traces for ECG and PPG); no audio, raw values or filenames are written.

Usage:
  python docs/_build/render_hero.py --refs AS=/data/AS/search_ref AF=/data/AF/search_ref \\
      LQT=/data/LQT/search_ref WPW=/data/WPW/search_ref            # candidates, merged into the page
  python docs/_build/render_hero.py --refs VSD=... COPD=... --keep VSD:2 COPD:3   # publish chosen ones
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

# task: (seed, best node, modality, display name). Audio and PPG use the
# median-seed runs of Table 1; ECG uses the best seed (also shown in the explorer).
TASKS = {
    "VSD": (1, 89, "audio", "Ventricular septal defect"),
    "AS": (3, 31, "audio", "Aortic stenosis"),
    "COPD": (1, 96, "audio", "COPD"),
    "AF": (2, 26, "ppg", "Atrial fibrillation"),
    "LQT": (3, 88, "ecg", "Long-QT syndrome"),
    "WPW": (3, 72, "ecg", "Wolff-Parkinson-White syndrome"),
}
EXT = {"audio": "*.wav", "ppg": "*.npy", "ecg": "*.npy"}


def generate(script: Path, n: int, out: Path, ext: str):
    subprocess.run([sys.executable, "-m", "simauthor.cli", "generate", str(script), "-n", str(n),
                    "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    return sorted(out.glob(ext))


# ── features ────────────────────────────────────────────────────────────
ECG_DISPLAY_SR = 100   # ECG is shown at 100 Hz; 500 Hz traces are too dense for a small panel


def ecg_display(x):
    """(5000, 2) at 500 Hz -> (1000, 2) at 100 Hz with an anti-aliasing filter."""
    from scipy.signal import decimate
    return decimate(x, 500 // ECG_DISPLAY_SR, axis=0, zero_phase=True)


def load_ecg(path):
    x = np.load(path).astype(np.float64)
    if x.shape[0] == 2 and x.shape[-1] != 2:
        x = x.T
    return x[:5000]                                      # (5000, 2), 500 Hz


def ecg_features(path):
    from scipy.signal import find_peaks, welch
    x = load_ecg(path)
    f = []
    for k in range(2):
        s = x[:, k] - np.median(x[:, k])
        peaks, _ = find_peaks(np.abs(s), distance=150, height=np.percentile(np.abs(s), 98) * 0.5)
        rr = np.diff(peaks) / 500 if len(peaks) > 2 else np.array([1.0])
        fr, p = welch(s, fs=500, nperseg=1024)
        p = p / (p.sum() + 1e-12)
        f += [60 / np.mean(rr), np.std(rr), *np.percentile(s, [1, 10, 50, 90, 99]), np.std(s),
              float(np.sum(fr * p)), float(np.sum(p[fr < 5]))]
    return f


def features(modality, files, folder):
    if modality == "audio":
        from simauthor.search_feedback.signal.audio.mfcc import MFCCFeedback
        F, _ = MFCCFeedback()._load_features(str(folder))
        return F, files
    if modality == "ppg":
        from simauthor.search_feedback.signal.ppg.morphology import PPGSignal
        fb, rows, keep = PPGSignal(), [], []
        for f in files:
            seg = np.load(f).astype(np.float64).ravel()[:3750]
            r = fb._segment_features(seg) if len(seg) == 3750 else None
            if r is not None:
                rows.append(r)
                keep.append(f)
        return np.array(rows), keep
    rows = [ecg_features(f) for f in files]
    return np.array(rows), files


# ── rendering ───────────────────────────────────────────────────────────
def real_image(modality, path, dst):
    """Plot a real ECG/PPG recording on the scope's dark panel (image only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.3, 3.4), dpi=150)  # same aspect as the page panel
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    if modality == "ecg":
        x = ecg_display(load_ecg(path))
        t = np.arange(len(x)) / ECG_DISPLAY_SR
        for k in range(2):
            s = x[:, k]
            s = (s - np.median(s)) / (np.ptp(s) + 1e-9)
            ax.plot(t, s * 0.85 + (1 - k) * 1.0 + 0.5, color="#6fd0ff", lw=0.9)
        ax.set_ylim(0, 2)
        ax.set_xlim(0, 10)
    else:
        s = np.load(path).astype(np.float64).ravel()[:3750]
        s = (s - s.min()) / (np.ptp(s) + 1e-9)
        ax.plot(np.arange(len(s)) / 125, s * 0.8 + 0.1, color="#6fd0ff", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 30)
    for v in np.linspace(*ax.get_xlim(), 11)[1:-1]:
        ax.axvline(v, color="white", alpha=0.14, lw=0.6)
    for v in np.linspace(*ax.get_ylim(), 5)[1:-1]:
        ax.axhline(v, color="white", alpha=0.14, lw=0.6)
    ax.axis("off")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, facecolor="black")
    plt.close(fig)
    return {"img": str(dst.relative_to(DOCS))}


def sim_item(modality, path, dst):
    if modality == "audio":
        return audio_item(path, dst.with_suffix(".mp3"))
    if modality == "ecg":
        x = ecg_display(load_ecg(path))
        return {"sr": ECG_DISPLAY_SR, "leads": [[round(float(v), 3) for v in x[:, k]] for k in range(2)]}
    s = np.load(path).astype(np.float64).ravel()[:3750]
    s = s / (np.max(np.abs(s)) + 1e-9)
    return {"sr": 125, "sig": [round(float(v), 3) for v in s]}


def real_item(modality, path, dst):
    if modality == "audio":
        import librosa
        y, _ = librosa.load(str(path), sr=16000, mono=True, duration=10.0)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
        return {"env": envelope(y), "mel": mel_b64(y)}
    return real_image(modality, path, dst.with_suffix(".png"))


def examples_for(task, ref, n, k):
    seed, best, modality, name = TASKS[task]
    run = REPO / "experiments" / "main" / task / f"seed{seed}"
    nodes = json.load(open(run / "tree.json"))["nodes"]
    ext = EXT[modality]
    real_files = sorted(Path(ref).glob(ext))
    R, real_files = features(modality, real_files, ref)
    mu, sd = R.mean(0), R.std(0) + 1e-8
    tmp = Path(tempfile.mkdtemp(prefix="simauthor_hero_"))
    feats, files = {}, {}
    for tag, idx in (("root", 0), ("best", best)):
        gen = generate(run / "candidates" / f"script_{idx:03d}.py", n, tmp / tag, ext)
        F, files[tag] = features(modality, gen, tmp / tag)
        feats[tag] = (F - mu) / sd
    Rn = (R - mu) / sd
    D = {t: np.linalg.norm(feats[t][:, None, :] - Rn[None], axis=2) for t in feats}  # (n_gen, n_ref)
    picked, used = [], {"root": set(), "best": set()}
    left = set(range(len(Rn)))

    def nearest(t, i):  # nearest unused generated record of program t to real recording i
        order = [g for g in np.argsort(D[t][:, i]) if g not in used[t]]
        return int(order[0]), float(D[t][order[0], i])

    while left and len(picked) < k:
        scored = []
        for i in left:
            a, da = nearest("root", i)
            b, db = nearest("best", i)
            scored.append((da - db, i, a, b, da, db))
        gap, i, a, b, da, db = max(scored)
        if db >= da:
            break
        picked.append((i, a, b, da, db))
        used["root"].add(a); used["best"].add(b); left.discard(i)

    media = DOCS / "media" / "hero" / task
    shutil.rmtree(media, ignore_errors=True)
    items = []
    for j, (i, a, b, d_root, d_best) in enumerate(picked):
        items.append({
            "real": real_item(modality, real_files[i], media / f"real_{j}"),
            "root": sim_item(modality, files["root"][a], media / f"root_{j}"),
            "authored": sim_item(modality, files["best"][b], media / f"authored_{j}"),
            "distance": {"root": round(d_root, 2), "authored": round(d_best, 2)},
        })
        print(f"  {task} candidate {j + 1}: distance root {d_root:.2f}, authored {d_best:.2f}")
    shutil.rmtree(tmp, ignore_errors=True)
    return {"task": task, "name": name, "modality": modality, "seed": seed, "best": best,
            "n_ref": len(real_files), "root_score": round(nodes[0]["score"], 3),
            "best_score": round(nodes[best]["score"], 3), "items": items}


def keep_one(ex, n):
    """Publish only candidate n (1-based) of an example; rename its media files."""
    it = ex["items"][n - 1]
    for key in ("real", "root", "authored"):
        for field in ("audio", "img"):
            if field in it[key]:
                src = DOCS / it[key][field]
                dst = src.with_name(f"{key}{src.suffix}")
                shutil.copy(src, dst)
                it[key][field] = str(dst.relative_to(DOCS))
    it.pop("distance", None)
    ex["items"] = [it]
    for f in (DOCS / "media" / "hero" / ex["task"]).glob("*_[0-9].*"):
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="*", default=[], metavar="TASK=DIR",
                    help="search reference directory per task to (re)render")
    ap.add_argument("--n", type=int, default=20, help="records generated per program")
    ap.add_argument("--k", type=int, default=3, help="candidates per task")
    ap.add_argument("--keep", nargs="*", default=[], metavar="TASK:N",
                    help="publish only candidate N of a task, e.g. VSD:2")
    args = ap.parse_args()

    path = DOCS / "data" / "generations.json"
    data = json.load(open(path))
    hero = {e["task"]: e for e in data.get("_hero", {}).get("examples", [])}
    for spec in args.refs:
        task, ref = spec.split("=", 1)
        hero[task] = examples_for(task, ref, args.n, args.k)
    for spec in args.keep:
        task, n = spec.split(":")
        keep_one(hero[task], int(n))
    data["_hero"] = {"examples": [hero[t] for t in TASKS if t in hero]}
    path.write_text(json.dumps(data, separators=(",", ":")))
    print("hero tasks:", {t: len(e["items"]) for t, e in hero.items()})


if __name__ == "__main__":
    main()
