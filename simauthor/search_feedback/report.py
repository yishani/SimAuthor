"""
report.py — Structured Discrepancy Objects and Report Rendering
=================================================================
Framework-level utilities for building Numerical Discrepancy Reports.

Every Search Feedback Agent constructs ``Discrepancy`` objects from its
measurements and calls ``render_report()`` to produce the markdown report
that is passed to the Refiner.  Report formatting lives once, here.
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class ScalarDiscrepancy:
    """A discrepancy where both sides are single scalar values.

    Example: QRS duration — reference 96 ms, generated 112 ms.
    """
    name: str
    reference: float
    generated: float
    unit: str = ""
    interpretation: str = ""

    @property
    def difference(self) -> float:
        return self.generated - self.reference


@dataclass
class DistributionDiscrepancy:
    """A discrepancy comparing two distributions.

    Example: MFCC-0 histogram overlap = 0.47 (benchmark 1.0, gap −0.53).
    """
    name: str
    value: float
    benchmark: float = 1.0
    interpretation: str = ""

    @property
    def gap(self) -> float:
        return self.benchmark - self.value


# ── Report rendering ─────────────────────────────────────────────────────────

def render_report(title: str,
                  header: list[str],
                  discrepancies: list,
                  observations: list[str] = (),
                  ) -> str:
    """Render a complete Numerical Discrepancy Report as markdown.

    Parameters
    ----------
    title:
        Report heading (e.g. ``"Signal-Space Discrepancy Report"``).
    header:
        Lines inserted between the title and the discrepancy section
        (file counts, sample rates, overall score, etc.).
    discrepancies:
        List of :class:`ScalarDiscrepancy` or :class:`DistributionDiscrepancy`.
    observations:
        Plain-language observations rendered as a bulleted list.
    """
    parts = [f"## {title}", ""]
    parts.extend(header)
    parts.append("")

    if discrepancies:
        parts.append(_render_items(discrepancies))

    if observations:
        parts.append("")
        parts.append("### Observations")
        for obs in observations:
            parts.append(f"- {obs}")

    return "\n".join(parts)


def _render_items(items: list) -> str:
    scalars = [i for i in items if isinstance(i, ScalarDiscrepancy)]
    dists = [i for i in items if isinstance(i, DistributionDiscrepancy)]

    parts = []
    if scalars:
        parts.append(_scalar_table(scalars))
    if dists:
        parts.append(_distribution_table(dists))
    return "\n\n".join(parts)


def _distribution_table(items: list[DistributionDiscrepancy]) -> str:
    hdr = f"| {'Feature':<28} | {'Value':>8} | {'Benchmark':>10} | {'Gap':>8} |"
    sep = f"|{'—' * 30}|{'—' * 10}|{'—' * 12}|{'—' * 10}|"
    rows = [hdr, sep]
    for d in items:
        rows.append(
            f"| {d.name:<28} | {d.value:>8.4f} | {d.benchmark:>10.4f} | "
            f"{d.gap:>+8.4f} |"
        )
        if d.interpretation:
            rows.append(f"  *{d.interpretation}*")
    return "\n".join(rows)

def _scalar_table(items: list[ScalarDiscrepancy]) -> str:
    hdr = f"| {'Feature':<28} | {'Reference':>10} | {'Generated':>10} | {'Δ':>10} |"
    sep = f"|{'—' * 30}|{'—' * 12}|{'—' * 12}|{'—' * 12}|"
    rows = [hdr, sep]
    for d in items:
        u = f" {d.unit}" if d.unit else ""
        rows.append(
            f"| {d.name:<28} | {d.reference:>9.3f}{u} | "
            f"{d.generated:>9.3f}{u} | {d.difference:>+9.3f}{u} |"
        )
        if d.interpretation:
            rows.append(f"  *{d.interpretation}*")
    return "\n".join(rows)
