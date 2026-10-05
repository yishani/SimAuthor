"""Shared run-directory preparation for the search CLI and external runners.

Thin reusable logic — no tree-search, no prompt rendering, no search feedback.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def prepare_run_dir(initialization_dir: str, run_dir: str, *,
                    condition_name: str = "",
                    signal_modality: str = "",
                    feedback_agent: str = "",
                    model_name: str = "",
                    # ── Search/runtime hyperparameters (all optional) ──────
                    iterations: int = 0,
                    c_puct: float = 0.0,
                    select_mode: str = "puct",
                    report_mode: str = "full",
                    num_samples: int = 0,
                    duration: float = 0.0,
                    synth_sr: int = 0,
                    output_sr: int = 0,
                    min_delta_for_mechanism: float = 0.0,
                    enable_search: bool = False,
                    ) -> dict:
    """Validate initialization artifacts and scaffold a run directory.

    1. Verify *initialization_dir* contains ``scientific_blueprint.md``
       and ``script_000.py``.
    2. Create *run_dir* with ``candidates/``, ``generated/``, ``logs/``.
    3. Copy ``script_000.py`` → ``<run_dir>/candidates/script_000.py``.
    4. Write ``run_config.json`` including resolved search hyperparameters.

    Returns a dict with keys ``blueprint``, ``script_path``, ``config``.
    """
    init = Path(initialization_dir)
    bp_path = init / "scientific_blueprint.md"
    script_path = init / "script_000.py"

    if not bp_path.exists():
        raise FileNotFoundError(
            f"Blueprint not found: {bp_path}.  Run initialization first."
        )
    if not script_path.exists():
        raise FileNotFoundError(
            f"Initial simulator not found: {script_path}.  "
            f"Run initialization first."
        )

    blueprint = bp_path.read_text()

    rd = Path(run_dir)
    candidates = rd / "candidates"
    gen = rd / "generated"
    logs = rd / "logs"
    candidates.mkdir(parents=True, exist_ok=True)
    gen.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    dest = candidates / "script_000.py"
    shutil.copy2(str(script_path), str(dest))

    config = {
        "condition_name": condition_name,
        "signal_modality": signal_modality,
        "feedback_agent": feedback_agent,
        "model_name": model_name,
        "initialization_dir": str(init.resolve()),
        "run_dir": str(rd.resolve()),
        "created_at": datetime.now().isoformat(),
    }

    # ── Resolved search/runtime hyperparameters ──────────────────────────
    # Only written when explicitly provided (non-zero / non-default).
    if iterations:
        config["iterations"] = iterations
    if c_puct:
        config["c_puct"] = c_puct
    if select_mode != "puct":
        config["select_mode"] = select_mode
    if report_mode != "full":
        config["report_mode"] = report_mode
    if num_samples:
        config["num_samples"] = num_samples
    if duration:
        config["duration"] = duration
    if synth_sr:
        config["synth_sr"] = synth_sr
    if output_sr:
        config["output_sr"] = output_sr
    if min_delta_for_mechanism:
        config["min_delta_for_mechanism"] = min_delta_for_mechanism
    if enable_search:
        config["enable_search"] = True

    (rd / "run_config.json").write_text(json.dumps(config, indent=2))

    return {
        "blueprint": blueprint,
        "script_path": str(dest),
        "config": config,
    }
