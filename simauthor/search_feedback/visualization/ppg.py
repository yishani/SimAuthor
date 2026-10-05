"""PPG visual feedback: real vs generated waveform grid."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 125  # PPG output rate
N_SHOW = 1250  # first 10 seconds


def ppg_visual_feedback(real_dir, generated_dir,
                        real_files, generated_files, output_path):
    n = len(real_files)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    t = np.arange(N_SHOW) / SR

    for col, f in enumerate(real_files):
        seg = np.load(os.path.join(real_dir, f)).astype(np.float64).flatten()
        seg = seg[:N_SHOW]
        axes[0, col].plot(t, seg, "k-", linewidth=0.5)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel("Real", fontsize=9)

    for col, f in enumerate(generated_files):
        seg = np.load(os.path.join(generated_dir, f)).astype(np.float64).flatten()
        seg = seg[:N_SHOW]
        axes[1, col].plot(t, seg, "k-", linewidth=0.5)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel("Generated", fontsize=9)

    fig.suptitle("Real (top) vs Generated (bottom) — PPG waveforms",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
