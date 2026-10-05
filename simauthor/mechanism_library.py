"""
mechanism_library.py — Mechanism Library
=========================================
Stores, formats, and extracts reusable simulator-design mechanisms
observed during successful parent-to-child search transitions.

A mechanism is a named, reusable change to signal generation,
physiological modeling, parameterization, temporal or spectral
structure, waveform morphology, or stochastic variation that is
plausibly associated with an improved child candidate.

Key operations:
  extract_from_code()    — LLM-based comparative analysis via unified diff
  add()                  — deduplicated addition (normalized names)
  load() / save()        — JSON persistence
  format_for_prompt()    — compact Refiner prompt section
  merge()                — cross-library combination

Each Search Feedback island maintains its own mechanism library.
"""

import difflib
import json
import re
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

from .prompts import load_prompt, render_prompt

# ── Prompt template (loaded once at import time) ───────────────────────────

_EXTRACT_TEMPLATE = load_prompt("mechanism_library", "extract_mechanism.md")

# ── Normalization ──────────────────────────────────────────────────────────

def _normalize_name(raw: str) -> str:
    """Normalize a mechanism name to lowercase snake_case."""
    name = raw.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:60]


def _compute_diff(parent: str, child: str, max_lines: int = 200) -> str:
    """Return a unified diff between *parent* and *child* scripts."""
    diff = list(difflib.unified_diff(
        parent.splitlines(keepends=True),
        child.splitlines(keepends=True),
        fromfile="parent", tofile="child",
    ))
    if len(diff) > max_lines:
        diff = diff[:max_lines]
        diff.append("... (diff truncated)\n")
    return "".join(diff)


# ── Mechanism dataclass ────────────────────────────────────────────────────

@dataclass
class Mechanism:
    """A candidate reusable simulator-design mechanism.

    Observed during a successful parent-to-child search transition;
    plausibly, but not necessarily causally, associated with the
    improvement.

    Attributes
    ----------
    name: Normalised snake_case identifier.
    mechanism: One-sentence description of the signal-generation change.
    code_snippet: Key function name or brief code fragment.
    source_node: Index of the child node where this was discovered.
    delta: Score improvement (child − parent) — metadata only.
    observation_count: How many times this mechanism has been observed.
    timestamp: ISO-format discovery time.
    island: Search Feedback island that discovered this mechanism.
    """
    name: str
    mechanism: str
    code_snippet: str
    source_node: int
    delta: float
    observation_count: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    island: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    # ── Historical field renames ─────────────────────────────────────────
    _FIELD_ALIASES = {
        "gene_name": "name",
        "validation_count": "observation_count",
    }

    @classmethod
    def from_dict(cls, d: dict) -> "Mechanism":
        normalised = dict(d)
        for legacy, canonical in cls._FIELD_ALIASES.items():
            if legacy in normalised and canonical not in normalised:
                normalised[canonical] = normalised.pop(legacy)
        return cls(**{k: v for k, v in normalised.items()
                      if k in cls.__dataclass_fields__})


# ── Mechanism Library ──────────────────────────────────────────────────────

class MechanismLibrary:
    """Persistent collection of candidate mechanisms for one island.

    Supports JSON load/save, normalised-name deduplication, compact
    Refiner prompt rendering, LLM-based extraction from diffs, and
    cross-library merging.
    """

    def __init__(self, path: str, island_key: str = ""):
        self.path = path
        self.island_key = island_key
        self.mechanisms: list[Mechanism] = []

    # ── Persistence ──────────────────────────────────────────────────────

    def load(self) -> list[Mechanism]:
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.mechanisms = [Mechanism.from_dict(g) for g in data]
        return self.mechanisms

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump([m.to_dict() for m in self.mechanisms], f, indent=2)

    # ── Collection management ────────────────────────────────────────────

    def add(self, mechanism: Mechanism) -> None:
        """Add a mechanism.  Deduplicates by normalised name."""
        mechanism.island = self.island_key
        key = _normalize_name(mechanism.name)
        mechanism.name = key

        for existing in self.mechanisms:
            if _normalize_name(existing.name) == key:
                existing.observation_count += 1
                if mechanism.delta > existing.delta:
                    existing.delta = mechanism.delta
                existing.timestamp = datetime.now().isoformat()
                return
        self.mechanisms.append(mechanism)

    def __len__(self) -> int:
        return len(self.mechanisms)

    def __bool__(self) -> bool:
        return len(self.mechanisms) > 0

    # ── Refiner prompt formatting ────────────────────────────────────────

    def format_for_prompt(self, max_entries: int = 8) -> str:
        """Render a compact mechanism library section for the Refiner prompt.

        Returns an empty string if the library is empty.
        """
        if not self.mechanisms:
            return ""

        lines = [
            "## Reusable Mechanisms",
            "These candidate mechanisms were observed in previously "
            "improved simulator transitions.  Reuse or adapt them only "
            "when relevant to the current discrepancy report and parent "
            "simulator.",
            "",
        ]
        entries = self.mechanisms[-max_entries:]  # most recent
        for m in entries:
            snippet = ""
            if m.code_snippet:
                snippet = f" (`{m.code_snippet}`)"
            lines.append(
                f"- `{m.name}`: {m.mechanism}{snippet}"
            )
        return "\n".join(lines)

    # ── Mechanism extraction ──────────────────────────────────────────────

    @staticmethod
    def extract_from_code(parent_code: str, child_code: str,
                          delta: float, node_idx: int,
                          model_client,
                          trace_logger=None,
                          iteration: int = -1,
                          parent_idx: int = -1) -> Optional[Mechanism]:
        """Extract a candidate mechanism via LLM analysis of the unified diff.

        *delta* is stored as metadata only — it is NOT passed to the
        extraction LLM.
        """
        diff = _compute_diff(parent_code, child_code)
        prompt = render_prompt(_EXTRACT_TEMPLATE,
            diff=diff,
        )

        try:
            text = model_client.generate_text(
                prompt, temperature=0.0, max_tokens=2000,
            )

            if trace_logger:
                trace_logger.log(
                    agent="mechanism_library",
                    step="extract",
                    iteration=iteration, parent_idx=parent_idx,
                    prompt=prompt, response=text,
                    temperature=0.0, max_tokens=2000,
                    model_name=model_client.model_name,
                )

            data = _parse_response(text)
            if data is None:
                return None

            return Mechanism(
                name=data["name"],
                mechanism=data["mechanism"],
                code_snippet=data.get("code_snippet", ""),
                source_node=node_idx,
                delta=round(delta, 5),
            )

        except Exception as e:
            print(f"  [mechanism extract] failed: {e}")
            return None

    # ── Merge ─────────────────────────────────────────────────────────────

    @classmethod
    def merge(cls, *libraries: "MechanismLibrary") -> "MechanismLibrary":
        """Merge libraries, deduplicating by normalised name.

        Source libraries are not mutated.  Each unique mechanism is
        copied into the merged result.
        """
        merged = cls(path="", island_key="merged")
        seen: dict[str, Mechanism] = {}
        for lib in libraries:
            for m in lib.mechanisms:
                key = _normalize_name(m.name)
                if key in seen:
                    seen[key].observation_count += m.observation_count
                    if m.delta > seen[key].delta:
                        seen[key].delta = m.delta
                else:
                    seen[key] = Mechanism(
                        name=key,
                        mechanism=m.mechanism,
                        code_snippet=m.code_snippet,
                        source_node=m.source_node,
                        delta=m.delta,
                        observation_count=m.observation_count,
                        island=m.island,
                    )
        merged.mechanisms = list(seen.values())
        return merged


# ── Response parsing ───────────────────────────────────────────────────────

def _parse_response(text: str) -> Optional[dict]:
    """Parse and validate an LLM extraction response.

    Returns a dict with keys ``name``, ``mechanism``, and optionally
    ``code_snippet``, or ``None`` if the response is malformed.
    """
    # Clean markdown fencing
    text = re.sub(r'^```[a-z]*\n?|\n?```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # 1. Strict JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 2. Conservative regex fallback
        name_m = re.search(r'"name"\s*:\s*"([^"\n,}]+)"', text)
        mech_m = re.search(r'"mechanism"\s*:\s*"([^"]+)"', text)
        snip_m = re.search(r'"code_snippet"\s*:\s*"([^"]*)"', text)
        if not (name_m and mech_m):
            print(f"  [mechanism extract] parse failed: {text[:120]!r}")
            return None
        data = {
            "name": name_m.group(1).strip(),
            "mechanism": mech_m.group(1).strip(),
            "code_snippet": snip_m.group(1).strip() if snip_m else "",
        }

    # Validate required fields
    if not data.get("name") or not data.get("mechanism"):
        print(f"  [mechanism extract] missing required field: {data!r}")
        return None

    # Normalize name
    data["name"] = _normalize_name(data["name"])
    if not data["name"]:
        print(f"  [mechanism extract] name empty after normalisation")
        return None

    # Clean mechanism description
    data["mechanism"] = re.sub(r"\s+", " ", data["mechanism"]).strip()

    # Clean code snippet
    if data.get("code_snippet"):
        data["code_snippet"] = re.sub(r"\s+", " ",
                                       data["code_snippet"]).strip()[:200]

    return data
