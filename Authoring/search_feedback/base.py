"""
base.py — Search Feedback Interface
=====================================
Framework-level types for the Search Feedback system.  Modality-independent.

Every Search Feedback Agent compares a Search Reference Set against a
Generated Sample Set and returns:

  1. A scalar **score** [0, 1] — used by the Orchestrator for PUCT node selection.
  2. A structured **discrepancy report** (markdown) — used by the Refiner.
  3. Optional structured **metadata** for logging and analysis.

The report describes *what* differs and by *how much*.  It does not prescribe
simulator modifications — those decisions belong to the Refiner.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchFeedback:
    """Unified output from ``analyze()``.

    Attributes
    ----------
    score:
        Scalar quality score in [0, 1] (higher = better).  Used by the
        Orchestrator for PUCT node selection.
    summary:
        One-line summary for logging and display.
    report:
        Structured discrepancy report (markdown).  Injected into the
        Refiner's prompt.  Describes what differs and by how much —
        no repair instructions.
    metadata:
        Machine-readable structured data.  Not injected into prompts.
    """
    score: float
    summary: str = ""
    report: str = ""
    metadata: dict = field(default_factory=dict)


class SearchFeedbackAgent(ABC):
    """Abstract interface for all Search Feedback Agents.

    Subclasses MUST implement
    ``analyze(reference_dir, generated_dir, *, include_report) → SearchFeedback``.

    Class-level attributes (set by subclasses or by ``create_feedback()``):

    ``agent_id``:
        Registry key (e.g. ``"representation"``, ``"signal"``).
    ``comparison_space``:
        ``"representation"`` or ``"signal"``.
    ``modality``:
        ``"audio"``, ``"ecg"``, ``"ppg"``, etc.
    """

    agent_id: str = ""
    comparison_space: str = ""
    modality: str = ""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def analyze(self,
                reference_dir: str,
                generated_dir: str,
                *,
                include_report: bool = True,
                ) -> SearchFeedback:
        """Measure discrepancy between reference and generated signals.

        Single authoritative entry point.  Computes features once and returns
        both a scalar score and (when *include_report* is True) a structured
        discrepancy report.
        """
        ...

    # ── Convenience methods ────────────────────────────────────────────────

    def evaluate(self, reference_dir: str, generated_dir: str) -> float:
        """Score-only.  Equivalent to ``analyze(include_report=False).score``."""
        return self.analyze(reference_dir, generated_dir,
                            include_report=False).score

    def __repr__(self) -> str:
        return f"{self.name}(modality={self.modality}, space={self.comparison_space})"
