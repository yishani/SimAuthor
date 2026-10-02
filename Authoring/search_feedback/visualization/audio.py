"""Audio visual feedback: real vs generated mel-spectrogram grid."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 20
# Heart/lung sounds carry essentially no energy above ~1-2 kHz (AS murmur
# 150-400 Hz, S1/S2 50-150 Hz, wheeze <2 kHz).  At the old FMAX=8000 Hz the
# informative content collapsed into the bottom 2-3% of each spectrogram,
# leaving mostly-empty panels that made the visual useless to the Refiner.
FMAX = 2000
SR = 16000


def _mel_db(wav_path):
    import librosa
    y, _ = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return librosa.power_to_db(mel, ref=np.max)


def audio_visual_feedback(real_dir, generated_dir,
                          real_files, generated_files, output_path):
    n = len(real_files)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)

    # Determine shared color scale across all panels
    all_db = []
    for f in real_files + generated_files:
        d = real_dir if f in real_files else generated_dir
        all_db.append(_mel_db(os.path.join(d, f)))
    vmin = min(m.min() for m in all_db)
    vmax = max(m.max() for m in all_db)

    for col, f in enumerate(real_files):
        db = _mel_db(os.path.join(real_dir, f))
        axes[0, col].imshow(db, aspect="auto", origin="lower", cmap="magma",
                            vmin=vmin, vmax=vmax,
                            extent=[0, db.shape[1] * HOP / SR, 0, FMAX / 1000])
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel("Real\n(kHz)", fontsize=9)

    for col, f in enumerate(generated_files):
        db = _mel_db(os.path.join(generated_dir, f))
        axes[1, col].imshow(db, aspect="auto", origin="lower", cmap="magma",
                            vmin=vmin, vmax=vmax,
                            extent=[0, db.shape[1] * HOP / SR, 0, FMAX / 1000])
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel("Generated\n(kHz)", fontsize=9)

    fig.suptitle("Real (top) vs Generated (bottom) — mel-spectrograms",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
