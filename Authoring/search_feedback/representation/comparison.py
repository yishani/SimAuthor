"""
representation/comparison.py — Shared Embedding Comparison
=============================================================
Standardised set-to-set comparison for Representation-Space agents.

Primary score:  scaled unbiased Energy Distance (U-statistic) between
L2-normalised reference and generated embeddings.

Energy Distance is a parameter-free proper metric on distributions that
penalises mode collapse, unlike centroid-only similarity.

Every representation agent (ECGFounder and PaPaGei) calls
``compare()`` for the math and ``build_feedback()`` for the report.
Agents only provide ``_embed(directory) → ndarray``.

Design
------
- **Unbiased U-statistic ED**: denominators use n*(n-1) for within-set
  terms, producing an unbiased estimator of population Energy Distance.
  Under the null (same distribution) the expectation is ≈ 0 at all
  sample sizes; finite-sample estimates can be slightly negative and are
  clamped at zero.

- **Fixed-scale score mapping**: ``score = 1 / (1 + ED_eff / scale)``
  where ``ED_eff = max(ED_unbiased, 0)``.  The scale constant (default
  0.15) is a fixed calibration value chosen to provide useful score
  dynamic range across embedding geometries; it is NOT derived from
  pairwise embedding distances.

- **No bootstrap**: the score is deterministic given reference and
  generated embeddings — no random real-vs-real splits at inference
  time.
"""

import numpy as np

from ..base import SearchFeedback
from ..report import DistributionDiscrepancy, render_report

DEFAULT_SCALE = 0.15


def _l2n(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased (U-statistic) Energy Distance between two sets of vectors.

    Computed on L2-normalised embeddings.  Uses n*(n-1) denominators for
    within-set pairwise-distance terms, producing an unbiased estimator of
    the population Energy Distance.

    Finite-sample estimates can be slightly negative under the null
    distribution (same underlying population).  Callers should clamp at
    zero when constructing a [0,1] score.
    """
    X = _l2n(X)
    Y = _l2n(Y)
    n, m = X.shape[0], Y.shape[0]
    d_xy = np.linalg.norm(X[:, None] - Y[None, :], axis=-1).sum() / (n * m)
    d_xx = (np.linalg.norm(X[:, None] - X[None, :], axis=-1).sum()
            / (n * (n - 1))) if n > 1 else 0.0
    d_yy = (np.linalg.norm(Y[:, None] - Y[None, :], axis=-1).sum()
            / (m * (m - 1))) if m > 1 else 0.0
    return float(2.0 * d_xy - d_xx - d_yy)


def compare(real_emb: np.ndarray, gen_emb: np.ndarray, *,
            scale: float = DEFAULT_SCALE) -> dict:
    """Compute set-to-set representation-space statistics.

    Primary score: scaled unbiased Energy Distance in [0, 1] (higher = better).

    ``score = 1 / (1 + ED_eff / scale)`` where ``ED_eff = max(ED_unbiased, 0)``.

    - Same distribution → ED ≈ 0 → score ≈ 1.0.
    - Moderate shift → ED ≈ scale → score ≈ 0.5.
    - Severe collapse → ED ≫ scale → score → 0.

    Parameters
    ----------
    real_emb: (n_real, d) reference embeddings.
    gen_emb:  (n_gen, d) generated embeddings.
    scale:    Score-scale constant (default 0.15).

    Returns
    -------
    dict with keys ``score``, ``energy_distance``, ``energy_distance_eff``,
    ``scale``, ``real_mean``, ``gen_mean``, ``real_std``, ``gen_std``,
    ``gap``, ``std_ratio``, ``inter_sample_sim``, ``real_pairwise_sim``.
    """
    real = _l2n(real_emb)
    gen = _l2n(gen_emb)
    n_gen = gen.shape[0]
    n_real = real.shape[0]

    # ── Unbiased Energy Distance ──────────────────────────────────────────
    ed_raw = energy_distance(real_emb, gen_emb)
    ed_eff = max(ed_raw, 0.0)  # clamp negative finite-sample estimates
    score = float(1.0 / (1.0 + ed_eff / max(scale, 1e-8)))

    # ── Centroid alignment (report only) ──────────────────────────────────
    centroid = real.mean(axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    real_sims = (real @ centroid.T).ravel()
    gen_sims = (gen @ centroid.T).ravel()

    real_mean = float(np.mean(real_sims))
    gen_mean = float(np.mean(gen_sims))
    real_std = float(np.std(real_sims))
    gen_std = float(np.std(gen_sims))
    std_ratio = gen_std / (real_std + 1e-8)

    # ── Inter-sample similarity (mode-collapse detection) ─────────────────
    if gen.shape[0] > 1:
        inter = gen @ gen.T
        np.fill_diagonal(inter, 0)
        inter_sim = float(inter.sum() / (gen.shape[0] * (gen.shape[0] - 1)))
    else:
        inter_sim = 1.0

    # ── Real pairwise similarity for reference ────────────────────────────
    if real.shape[0] > 1:
        r_inter = real @ real.T
        np.fill_diagonal(r_inter, 0)
        real_pairwise = float(
            r_inter.sum() / (real.shape[0] * (real.shape[0] - 1))
        )
    else:
        real_pairwise = 1.0

    return {
        "score": score,
        "energy_distance": ed_raw,
        "energy_distance_eff": ed_eff,
        "scale": scale,
        "real_mean": real_mean, "gen_mean": gen_mean,
        "real_std": real_std, "gen_std": gen_std,
        "gap": real_mean - gen_mean,
        "std_ratio": std_ratio,
        "inter_sample_sim": inter_sim,
        "real_pairwise_sim": real_pairwise,
    }


def build_feedback(stats: dict, *,
                   title: str,
                   encoder_name: str,
                   n_real: int, n_gen: int,
                   embedding_dim: int,
                   ) -> SearchFeedback:
    """Build a SearchFeedback with a representation-space report."""
    s = stats
    ed_raw = s.get("energy_distance", 0.0)
    ed_eff = s.get("energy_distance_eff", max(ed_raw, 0.0))
    scale = s.get("scale", DEFAULT_SCALE)

    discrepancy_name = "Energy Distance (unbiased, set-to-set)"
    if ed_raw < 0:
        discrepancy_name += " [clamped at 0]"

    discrepancies = [
        DistributionDiscrepancy(
            name=discrepancy_name,
            value=1.0 - s["score"], benchmark=0.0,
            interpretation=(
                f"ED_unbiased = {ed_raw:.6f} (eff = {ed_eff:.6f}).  "
                f"Scale = {scale:.4f}.  "
                f"Score = 1/(1+ED_eff/scale).  "
                f"Lower ED = more similar."
            ),
        ),
        DistributionDiscrepancy(
            name="Mean centroid similarity",
            value=s["gen_mean"], benchmark=s["real_mean"],
            interpretation=(
                f"Generated embeddings are {s['gap']:+.3f} from the "
                f"reference centroid."
            ),
        ),
        DistributionDiscrepancy(
            name="Spread ratio (gen / ref)",
            value=s["std_ratio"], benchmark=1.0,
            interpretation=(
                f"Gen spread is {s['std_ratio']:.2f}× reference spread."
            ),
        ),
        DistributionDiscrepancy(
            name="Gen inter-sample similarity",
            value=s["inter_sample_sim"],
            benchmark=s.get("real_pairwise_sim", 0.0),
            interpretation=(
                f"Reference pairwise similarity = "
                f"{s.get('real_pairwise_sim', 0):.4f}.  "
                f"Near 1.0 indicates mode collapse."
            ),
        ),
    ]

    observations = []
    if s["score"] < 0.3:
        observations.append(
            f"Large distribution discrepancy (score = {s['score']:.3f})."
        )
    elif s["score"] < 0.7:
        observations.append(
            f"Moderate distribution discrepancy (score = {s['score']:.3f})."
        )
    else:
        observations.append(
            f"Small distribution discrepancy (score = {s['score']:.3f})."
        )

    if ed_raw < 0:
        observations.append(
            f"ED_unbiased = {ed_raw:.6f} < 0 — finite-sample U-statistic "
            f"estimate below zero (clamped to 0 for scoring)."
        )

    if s["inter_sample_sim"] > s.get("real_pairwise_sim", 0) * 1.5:
        observations.append(
            f"Mode collapse: gen inter-sample similarity "
            f"({s['inter_sample_sim']:.4f}) ≫ reference "
            f"({s.get('real_pairwise_sim', 0):.4f})."
        )
    elif s["inter_sample_sim"] > 0.98:
        observations.append(
            f"Mode collapse: inter-sample similarity = "
            f"{s['inter_sample_sim']:.4f}."
        )

    report = render_report(
        title=title,
        header=[
            f"Segments: {n_real} reference (empirical sample)  |  "
            f"{n_gen} generated  |  "
            f"Embedding dim: {embedding_dim}",
            f"Encoder: {encoder_name}  |  "
            f"Score: {s['score']:.4f}  |  "
            f"ED (unbiased): {ed_raw:.6f}  |  "
            f"ED_eff: {ed_eff:.6f}  |  "
            f"Scale: {scale:.4f}",
        ],
        discrepancies=discrepancies,
        observations=observations,
    )

    return SearchFeedback(
        score=s["score"],
        summary=(
            f"{encoder_name} ED score: {s['score']:.3f}.  "
            f"ED_unbiased = {ed_raw:.6f}.  "
            f"{'Mode collapse!' if s['inter_sample_sim'] > 0.98 else ''}"
        ),
        report=report,
        metadata={
            "score": s["score"],
            "energy_distance": ed_raw,
            "energy_distance_eff": ed_eff,
            "scale": scale,
            "real_mean_sim": s["real_mean"],
            "gen_mean_sim": s["gen_mean"],
            "inter_sample_sim": s["inter_sample_sim"],
            "real_pairwise_sim": s.get("real_pairwise_sim", 0),
            "std_ratio": float(s["std_ratio"]),
        },
    )
