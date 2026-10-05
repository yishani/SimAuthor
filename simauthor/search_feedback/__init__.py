"""
search_feedback — Search Feedback Framework
=============================================
Each Search Feedback Agent compares a Search Reference Set against a
Generated Sample Set and returns a scalar score, a structured discrepancy
report, and optional metadata.

Framework
---------
  base.py              — SearchFeedback dataclass, SearchFeedbackAgent ABC
  report.py            — Discrepancy objects + shared markdown renderer
  registry.py          — Agent registration and factory

Comparison Spaces
-----------------
  representation/      — Learned-encoder comparison (ECGFounder, PaPaGei)
  signal/              — Handcrafted comparison (audio MFCC/ZCR, PPG morphology)
"""

# Framework types
from .base import SearchFeedback, SearchFeedbackAgent  # noqa: F401

# Registry API
from .registry import (                                 # noqa: F401
    register_feedback, create_feedback, list_registered, load_evaluator,
)

# Trigger registration of all built-in instances.
from . import representation as _rep   # noqa: F401
from . import signal as _sig           # noqa: F401
