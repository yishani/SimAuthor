"""
utils.py — Shared utilities for the SimAuthor framework
=============================================================
Trace logging, file I/O helpers, subprocess management, and code extraction.
"""

import os
import re
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Trace logging ──────────────────────────────────────────────────────────────

class TraceLogger:
    """JSONL trace logger for debugging and reproducibility.

    Usage:
        trace = TraceLogger("logs/run_20250101.jsonl")
        trace.log(agent="refiner", step="llm_call", prompt="...", response="...")
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, agent: str, step: str,
            iteration: int = -1, parent_idx: int = -1, child_idx: int = -1,
            score: Optional[float] = None, delta: Optional[float] = None,
            # ── LLM-specific (only written when prompt is present) ────
            prompt: str = "", response: str = "",
            temperature: float = 0.0, max_tokens: int = 0,
            model_name: str = "", tools: str = "",
            api_attempts: int = 0, api_duration_s: float = 0.0,
            extra: Optional[dict] = None) -> None:
        if not self.path:
            return

        # Core event fields — always present.
        entry: dict = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "step": step,
            "iteration": iteration,
            "parent_idx": parent_idx,
            "child_idx": child_idx,
        }
        if score is not None:
            entry["score"] = score
        if delta is not None:
            entry["delta"] = delta

        # LLM-specific fields — only for actual model calls.
        if prompt:
            entry.update({
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "prompt_chars": len(prompt),
                "response_chars": len(response),
                "prompt": prompt,
                "response": response,
            })
            if api_attempts:
                entry["api_attempts"] = api_attempts
                entry["api_duration_s"] = api_duration_s

        if extra:
            entry.update(extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Code extraction ────────────────────────────────────────────────────────────

def extract_code_block(text: str, language: str = "python") -> tuple[str, str]:
    """Extract a fenced code block from LLM response.

    Returns:
        (code, preamble) — code is the block content, preamble is text before it.
    """
    marker = f"```{language}"
    if marker in text:
        parts = text.split(marker, 1)
        preamble = parts[0].strip()
        after = parts[1]
        if "```" in after:
            code = after.split("```", 1)[0].strip()
        else:
            code = after.strip()
        return code, preamble
    return text.strip(), ""


def is_code_complete(code: str) -> bool:
    """Heuristic check: does the code look like a complete simulator script?"""
    has_save = any(kw in code for kw in ["wavfile.write", "sf.write", "soundfile.write", ".to_wav", "wav.write", "np.save"])
    has_loop = any(kw in code for kw in ["for ", "range(", "NUM_SAMPLES", "num_samples"])
    return has_save and has_loop and len(code) > 500


# ── Script naming ──────────────────────────────────────────────────────────────

def get_script_path(candidates_dir: str, idx: int) -> str:
    """Return ``<candidates_dir>/script_{idx:03d}.py``."""
    return os.path.join(candidates_dir, f"script_{idx:03d}.py")


def get_next_script_index(candidates_dir: str) -> int:
    """Return the next available script index in *candidates_dir*."""
    pattern = re.compile(r"^script_(\d{3})\.py$")
    nums = []
    if os.path.isdir(candidates_dir):
        for f in os.listdir(candidates_dir):
            m = pattern.match(f)
            if m:
                nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


# ── Script gen_dir patching ────────────────────────────────────────────────────

def patch_output_dir(script_path: str, target_dir: str) -> bool:
    """Replace the output-directory assignment in *script_path* with *target_dir*.

    Finds ``output_dir = "..."`` or ``OUTPUT_DIR = "..."`` and replaces the
    quoted path.  Modality-agnostic — works for audio, ECG, and PPG.
    Returns True if the file was modified.
    """
    with open(script_path) as f:
        code = f.read()

    # Preserve a trailing path separator so that scripts which build output
    # filenames by string concatenation (e.g. ``f'{OUTPUT_DIR}file.wav'``)
    # still produce correct paths.  Scripts using ``os.path.join`` are
    # unaffected by the extra separator.
    patched_dir = target_dir
    if not patched_dir.endswith(os.sep):
        patched_dir += os.sep

    patched = re.sub(
        r'(output_dir\s*=\s*)(["\'])(.*?)(\2)',
        rf'\1\2{patched_dir}\2',
        code, count=1, flags=re.IGNORECASE,
    )
    if patched != code:
        with open(script_path, "w") as f:
            f.write(patched)
        print(f"  Patched output_dir → {patched_dir}")
        return True
    return False


# Recognised sample-count variable names in LLM-generated simulator scripts.
_SAMPLE_COUNT_NAMES = ("N_SAMPLES", "NUM_SAMPLES", "num_samples", "n_samples")
_SAMPLE_COUNT_RE = re.compile(
    r'(?P<prefix>\b(?:N_SAMPLES|NUM_SAMPLES|num_samples|n_samples)\b'
    r'(?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)?\s*=\s*)'
    r'(?P<value>\d+)'
)


def patch_sample_count(script_path: str, num_samples: int) -> bool:
    """Patch recognised sample-count constants in *script_path* to *num_samples*.

    Matches ``N_SAMPLES = 472``, ``NUM_SAMPLES = 472``, ``num_samples = 472``,
    ``n_samples: int = 472`` (annotated) and ``config.n_samples = 472``
    (attribute access on the left side is preserved).

    Returns True if the file was modified.
    """
    with open(script_path) as f:
        code = f.read()

    patched, n = _SAMPLE_COUNT_RE.subn(
        lambda m: m.group("prefix") + str(num_samples), code)

    if n and patched != code:
        with open(script_path, "w") as f:
            f.write(patched)
        print(f"  Patched sample-count constants → {num_samples} ({n} site(s))")
        return True
    return False


def count_signal_artifacts(gen_dir: str, modality: str) -> int:
    """Count only modality-specific signal artifacts in *gen_dir*.

    Audio → ``.wav``; ECG/PPG → ``.npy``.  Metadata, JSON, logs and plots are
    ignored so they never trigger a false output-count violation.
    """
    ext = ".wav" if modality == "audio" else ".npy"
    if not os.path.isdir(gen_dir):
        return 0
    count = 0
    for name in os.listdir(gen_dir):
        if name.lower().endswith(ext) and os.path.isfile(os.path.join(gen_dir, name)):
            count += 1
    return count


# ── Subprocess helpers ─────────────────────────────────────────────────────────

def run_script(script_path: str, timeout: int = 600) -> tuple[bool, str, str]:
    """Run a Python simulator script.

    Returns:
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Script timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def clear_and_ensure_dir(dir_path: str) -> None:
    """Remove directory if it exists, then recreate it empty."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)


# ── Retry wrapper ──────────────────────────────────────────────────────────────

def retry_with_backoff(fn, max_attempts: int = 5, base_wait: float = 30.0,
                       label: str = ""):
    """Call fn() with exponential backoff on exception."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            wait = base_wait * (attempt + 1)
            print(f"  [{label}] Attempt {attempt+1}/{max_attempts} failed: {e}. "
                  f"Retrying in {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError(f"[{label}] Failed after {max_attempts} attempts")
