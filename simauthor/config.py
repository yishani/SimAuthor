"""
config.py — Shared configuration for the SimAuthor framework
==============================================================
Type aliases and search defaults.
"""

from typing import Literal

# ── Canonical signal modalities ──────────────────────────────────────────

SignalModality = Literal["audio", "ecg", "ppg"]

# ── Model and search defaults ────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_C_PUCT = 0.25
DEFAULT_ITERATIONS = 100
MIN_DELTA_FOR_MECHANISM = 0.02

# ── Representation-space scoring ─────────────────────────────────────────

# Fixed scale constant for the ED_unbiased → [0,1] score transformation.
# score = 1 / (1 + max(ED_unbiased, 0) / DEFAULT_REPRESENTATION_SCALE)
DEFAULT_REPRESENTATION_SCALE = 0.15
