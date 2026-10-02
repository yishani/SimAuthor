"""ECG visual feedback: real vs generated waveform grid (2-lead)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 500  # ECG synthesis/output rate used by the pipeline
N_SHOW = 2000  # first 4 seconds


def _load_ecg(path):
    seg = np.load(path).astype(np.float64)
    if seg.ndim == 2 and seg.shape[1] == 2:
        seg = seg.T  # (N,2) -> (2,N)
    elif seg.ndim == 2 and seg.shape[0] == 2:
        pass  # already (2,N)
    else:
        seg = np.stack([seg, seg])  # 1D -> 2 identical leads
    return seg[:, :N_SHOW]


def ecg_visual_feedback(real_dir, generated_dir,
                        real_files, generated_files, output_path):
    n = len(real_files)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    t = np.arange(N_SHOW) / SR

    for col, f in enumerate(real_files):
        seg = _load_ecg(os.path.join(real_dir, f))
        axes[0, col].plot(t, seg[0], "k-", linewidth=0.4)
        if seg.shape[0] > 1:
            axes[0, col].plot(t, seg[1] + 3, "b-", linewidth=0.3, alpha=0.6)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        axes[0, col].set_ylim(-2, 6)
        if col == 0:
            axes[0, col].set_ylabel("Real", fontsize=9)

    for col, f in enumerate(generated_files):
        seg = _load_ecg(os.path.join(generated_dir, f))
        axes[1, col].plot(t, seg[0], "k-", linewidth=0.4)
        if seg.shape[0] > 1:
            axes[1, col].plot(t, seg[1] + 3, "b-", linewidth=0.3, alpha=0.6)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        axes[1, col].set_ylim(-2, 6)
        if col == 0:
            axes[1, col].set_ylabel("Generated", fontsize=9)

    fig.suptitle("Real (top) vs Generated (bottom) — ECG waveforms",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
