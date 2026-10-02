"""
_simulator_runner.py — Script execution and output-directory management.
=======================================================================
Thin wrappers around utils.py for all simulator-script I/O and execution.

Scripts live under ``<run_dir>/candidates/`` with the naming convention
``script_000.py``, ``script_001.py``, …  Script 000 is the initial simulator
produced by Initialization; it is treated identically to later candidates.
"""

from ..utils import (
    get_script_path, get_next_script_index,
    patch_output_dir, clear_and_ensure_dir, run_script,
)


def save_candidate(candidates_dir: str, idx: int, code: str,
                   gen_dir: str) -> str:
    """Write *code* to ``<candidates_dir>/script_{idx}.py``, patch gen_dir."""
    path = get_script_path(candidates_dir, idx)
    with open(path, "w") as f:
        f.write(code)
    patch_output_dir(path, gen_dir)
    return path


def candidate_path(candidates_dir: str, idx: int) -> str:
    return get_script_path(candidates_dir, idx)


def next_candidate_index(candidates_dir: str) -> int:
    return get_next_script_index(candidates_dir)


def prepare_gen_dir(gen_dir: str) -> None:
    clear_and_ensure_dir(gen_dir)


def execute(path: str, gen_dir: str) -> tuple[bool, str, str]:
    """Run a simulator script.  Returns (success, stdout, stderr)."""
    clear_and_ensure_dir(gen_dir)
    return run_script(path)


# Re-export for cold-start use.
patch_output_dir = patch_output_dir
