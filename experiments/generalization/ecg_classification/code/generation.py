"""Utilities for executing a released simulator with a fixed sample budget."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def generate_samples(script, n_samples, seed=42, cwd=None):
    """Run ``script`` after redirecting its output and fixing its RNG seed."""
    script = Path(script).resolve()
    code = script.read_text()
    output_dir = Path(tempfile.mkdtemp(prefix="simauthor_ecg_"))

    code, replacements = re.subn(
        r'(output_dir\s*=\s*)(["\'])(.*?)(\2)',
        rf'\1\2{output_dir}\2',
        code,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements != 1:
        raise ValueError(f"Could not redirect output_dir in {script}")

    code = re.sub(r'(RANDOM_SEED\s*=\s*)\d+', rf'\g<1>{seed}', code)
    code = re.sub(r'(np\.random\.seed\()\d+\)', rf'\g<1>{seed})', code)
    code = re.sub(r'(SEED\s*=\s*)\d+', rf'\g<1>{seed}', code)
    for name in ("N_SAMPLES", "NUM_SAMPLES", "num_samples", "n_samples"):
        code = re.sub(rf'({name}\s*=\s*)\d+', rf'\g<1>{n_samples}', code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(code)
        temporary_script = Path(fh.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(temporary_script)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(cwd or script.parent),
        )
    finally:
        temporary_script.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Simulator failed ({script}):\n{completed.stderr[-4000:]}"
        )

    files = sorted(output_dir.glob("*.npy"))
    if len(files) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} generated files from {script}, got {len(files)}"
        )
    return [str(path) for path in files]
