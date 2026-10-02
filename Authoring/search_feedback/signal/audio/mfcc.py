"""
audio/signal/mfcc.py — MFCC + ZCR Distribution Overlap (Audio Signal Space)
=============================================================================
Compares reference and generated audio sets using per-recording MFCC
feature extraction and one-dimensional histogram overlap.

Comparison space:  signal
Modality:           audio
Condition:          agnostic — defined by the supplied reference set

Score
-----
1. Load each WAV file, peak-normalise, extract 13 MFCCs + ZCR (14 features).
2. Z-normalise using reference-set mean and standard deviation.
3. Compute per-feature one-dimensional histogram overlap (30 bins).
4. Score = mean of 14 per-feature overlaps  ∈ [0, 1].

The score measures MFCC-feature distribution overlap in Audio Signal Space.
Per-recording peak-normalisation is applied, so absolute amplitude is not
evaluated.  ZCR is retained as a 14th feature.

MFCC coefficient labels are approximate interpretive guides, not precise
physiological descriptions.
"""

import numpy as np
from pathlib import Path

from ...base import SearchFeedbackAgent, SearchFeedback
from ... import register_feedback


def _feature_name(idx: int, n_mfcc: int) -> str:
    """Return ``MFCC-{idx}`` or ``ZCR`` for the last index."""
    if idx == n_mfcc:
        return "ZCR"
    return f"MFCC-{idx}"


class MFCCFeedback(SearchFeedbackAgent):
    """MFCC + ZCR histogram-overlap comparison for audio."""

    agent_id = "mfcc"
    comparison_space = "signal"
    modality = "audio"

    _MFCC_LABELS = [
        "MFCC-0 (spectral energy — note: peak-norm removes absolute amplitude)",
        "MFCC-1 (spectral tilt, approx.)",
        "MFCC-2 (broad spectral shape, approx.)",
        "MFCC-3 (mid-low detail, approx.)",
        "MFCC-4 (mid detail, approx.)",
        "MFCC-5 (mid-high detail, approx.)",
        "MFCC-6 (fine texture — low, approx.)",
        "MFCC-7 (fine texture — mid-low, approx.)",
        "MFCC-8 (fine texture — mid, approx.)",
        "MFCC-9 (fine texture — mid-high, approx.)",
        "MFCC-10 (fine texture — high, approx.)",
        "MFCC-11 (highest-order — low, approx.)",
        "MFCC-12 (highest-order — high, approx.)",
    ]

    def __init__(self, name: str = "MFCC + ZCR Overlap",
                 sr: int = 16000, n_mfcc: int = 13):
        super().__init__(name=name)
        self.sr = sr
        self.n_mfcc = n_mfcc

    # ── Feature extraction ────────────────────────────────────────────────

    def _load_features(self, directory: str) -> tuple[np.ndarray, dict]:
        """Load and extract MFCC+ZCR feature vectors from all WAV files.

        Returns
        -------
        (features, stats) where *features* is (n_valid, 14) and *stats*
        records file counts and failures.
        """
        import librosa
        feats = []
        n_total = 0
        n_failed = 0
        failures: list[str] = []

        for f in sorted(Path(directory).glob("*.wav")):
            n_total += 1
            try:
                audio, _ = librosa.load(str(f), sr=self.sr)
                # Per-recording peak normalisation — removes absolute
                # amplitude; comparison measures spectral-shape statistics.
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                mfcc = librosa.feature.mfcc(
                    y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                )
                zcr = librosa.feature.zero_crossing_rate(audio)
                feats.append(
                    np.concatenate([np.mean(mfcc, axis=1), [np.mean(zcr)]])
                )
            except Exception:
                n_failed += 1
                failures.append(f.name)

        stats = {
            "n_total": n_total,
            "n_valid": len(feats),
            "n_failed": n_failed,
            "failures": failures[:10],  # keep bounded
        }
        arr = np.array(feats) if feats else np.empty((0, self.n_mfcc + 1))
        return arr, stats

    # ── Score computation ─────────────────────────────────────────────────

    @staticmethod
    def _histogram_overlap(a: np.ndarray, b: np.ndarray,
                           n_bins: int = 30) -> float:
        lo = min(a.min(), b.min()) - 0.5
        hi = max(a.max(), b.max()) + 0.5
        bins = np.linspace(lo, hi, n_bins)
        ha, _ = np.histogram(a, bins=bins, density=True)
        hb, _ = np.histogram(b, bins=bins, density=True)
        return float(np.clip(
            np.sum(np.minimum(ha, hb)) * (bins[1] - bins[0]), 0.0, 1.0,
        ))

    def _compute_score(self, real_feats, gen_feats) -> tuple[float, list[float]]:
        mean = np.mean(real_feats, axis=0)
        std = np.std(real_feats, axis=0) + 1e-8
        real_norm = (real_feats - mean) / std
        gen_norm = (gen_feats - mean) / std
        overlaps = [
            self._histogram_overlap(real_norm[:, i], gen_norm[:, i])
            for i in range(real_norm.shape[1])
        ]
        return float(np.mean(overlaps)), overlaps

    # ── analyze() ──────────────────────────────────────────────────────────

    def analyze(self, reference_dir: str, generated_dir: str, *,
                include_report: bool = True) -> SearchFeedback:
        real_feats, ref_stats = self._load_features(reference_dir)
        gen_feats, gen_stats = self._load_features(generated_dir)

        if real_feats.shape[0] < 3 or gen_feats.shape[0] < 3:
            return SearchFeedback(
                score=0.0, summary="Insufficient valid files (< 3 each).",
                report=(
                    f"Reference: {ref_stats['n_valid']} valid "
                    f"({ref_stats['n_failed']} failed).  "
                    f"Generated: {gen_stats['n_valid']} valid "
                    f"({gen_stats['n_failed']} failed).  "
                    f"Need at least 3 valid files each."
                ),
                metadata={"ref_stats": ref_stats, "gen_stats": gen_stats},
            )

        score, overlaps = self._compute_score(real_feats, gen_feats)

        if not include_report:
            return SearchFeedback(
                score=score, summary=f"Score: {score:.4f}",
                metadata={
                    "per_feature_overlap": [float(o) for o in overlaps],
                    "ref_stats": ref_stats, "gen_stats": gen_stats,
                },
            )

        # ── Per-feature discrepancy table ────────────────────────────────
        mean = np.mean(real_feats, axis=0)
        std = np.std(real_feats, axis=0) + 1e-8
        real_norm = (real_feats - mean) / std
        gen_norm = (gen_feats - mean) / std
        real_means = np.mean(real_norm, axis=0)
        gen_means = np.mean(gen_norm, axis=0)
        real_stds = np.std(real_norm, axis=0)
        gen_stds = np.std(gen_norm, axis=0)

        table = (
            f"| {'Feature':<22} | {'Overlap':>7} | {'Mean Shift':>9} | "
            f"{'Spread Ratio':>11} | Status |\n"
            f"|{'—' * 24}|{'—' * 9}|{'—' * 11}|{'—' * 13}|{'—' * 8}|"
        )
        table_lines = [table]
        under_count = 0
        over_count = 0
        for i in range(self.n_mfcc):
            name = f"MFCC-{i}"
            shift = gen_means[i] - real_means[i]
            sratio = gen_stds[i] / (real_stds[i] + 1e-8)
            if sratio < 0.5:
                status = "under"
                under_count += 1
            elif sratio > 2.0:
                status = "over"
                over_count += 1
            else:
                status = "—"
            table_lines.append(
                f"| {name:<22} | {overlaps[i]:>7.4f} | {shift:>+9.3f} σ | "
                f"{sratio:>11.3f} | {status:<6} |"
            )

        # ZCR
        zcr_i = self.n_mfcc
        zcr_shift = gen_means[zcr_i] - real_means[zcr_i]
        zcr_ratio = gen_stds[zcr_i] / (real_stds[zcr_i] + 1e-8)
        if zcr_ratio < 0.5:
            zcr_status = "under"
            under_count += 1
        elif zcr_ratio > 2.0:
            zcr_status = "over"
            over_count += 1
        else:
            zcr_status = "—"
        table_lines.append(
            f"| {'ZCR':<22} | {overlaps[zcr_i]:>7.4f} | {zcr_shift:>+9.3f} σ | "
            f"{zcr_ratio:>11.3f} | {zcr_status:<6} |"
        )

        feature_table = "\n".join(table_lines)

        # ── Observations ─────────────────────────────────────────────────
        ranked = np.argsort(overlaps)
        worst_idx = int(ranked[0])
        worst_name = _feature_name(worst_idx, self.n_mfcc)

        observations = [
            f"Reference: {ref_stats['n_valid']} valid"
            + (f" ({ref_stats['n_failed']} failed)" if ref_stats['n_failed'] else ""),
            f"Generated: {gen_stats['n_valid']} valid"
            + (f" ({gen_stats['n_failed']} failed)" if gen_stats['n_failed'] else ""),
            f"Worst overlap: {worst_name} (overlap = {overlaps[worst_idx]:.3f}).",
        ]
        if under_count or over_count:
            parts = []
            if under_count: parts.append(f"{under_count} under-diverse")
            if over_count: parts.append(f"{over_count} over-diverse")
            observations.append(", ".join(parts) + " (outside 0.5×–2.0× range).")
        else:
            observations.append("All features within 0.5×–2.0× diversity range.")

        report = (
            f"## Audio Signal-Space Discrepancy Report (MFCC + ZCR)\n\n"
            f"Reference (empirical sample): {ref_stats['n_valid']} files  |  "
            f"Generated: {gen_stats['n_valid']} files  |  "
            f"SR: {self.sr} Hz  |  MFCCs: {self.n_mfcc}  |  "
            f"ZCR in score: yes\n"
            f"Overall overlap score: {score:.4f}  "
            f"(1.0 = identical distributions)\n\n"
            f"Per-recording peak-normalisation is applied; MFCC-0 reflects "
            f"spectral shape after amplitude removal, not absolute loudness.\n\n"
            f"{feature_table}\n\n"
            f"### Observations\n"
            + "\n".join(f"- {o}" for o in observations)
        )

        return SearchFeedback(
            score=score,
            summary=(
                f"MFCC+ZCR overlap: {score:.3f}.  "
                f"Worst: {worst_name} (overlap={overlaps[worst_idx]:.3f})."
            ),
            report=report,
            metadata={
                "per_feature_overlap": [float(o) for o in overlaps],
                "n_reference": int(real_feats.shape[0]),
                "n_generated": int(gen_feats.shape[0]),
                "ref_stats": ref_stats,
                "gen_stats": gen_stats,
                "n_mfcc": self.n_mfcc,
                "zcr_in_score": True,
                "sample_rate": self.sr,
                "normalization": "per-recording peak",
            },
        )


register_feedback("audio", "mfcc", MFCCFeedback)
