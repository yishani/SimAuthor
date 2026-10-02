"""
initialization.py — Scientific Blueprint + Initial Simulator Generation
=========================================================================
Generates two independent initialization artifacts for a condition:

  1. A **Scientific Blueprint** — a structured domain-knowledge document
     that externalises the LLM's knowledge about a condition's biosignal
     characteristics.  This is the clinical foundation for all subsequent
     refinement.

  2. An **Initial Simulator** (script_000.py) — an executable Python script
     that synthesises the target signal.  This is a standalone one-shot
     generation (it does NOT use the Scientific Blueprint) and serves as
     the seed for tree search.

Both are one-shot LLM generations routed through ModelClient.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from .prompts import load_prompt
from .orchestrator._simulator_runner import execute as run_script

# ── Supported modalities ─────────────────────────────────────────────────

_SUPPORTED_MODALITIES = frozenset({"audio", "ecg", "ppg"})

# ── Prompt templates (loaded once at import time) ─────────────────────────

_SIM_MAIN = load_prompt("initialization", "initial_simulator", "main.md")
_BP_MAIN = load_prompt("initialization", "scientific_blueprint", "main.md")


def _load_mod_spec(stage: str, signal_modality: str) -> str:
    if signal_modality not in _SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported signal_modality={signal_modality!r}.  "
            f"Supported: {sorted(_SUPPORTED_MODALITIES)}"
        )
    return load_prompt(
        "initialization", stage, "modalities", f"{signal_modality}.md",
    )


# ── Strict rendering ──────────────────────────────────────────────────────

def _render(template: str, **kwargs) -> str:
    """Render *template* with ``str.format(**kwargs)`` and assert no leftover
    ``{placeholder}`` patterns remain."""
    try:
        result = template.format(**kwargs)
    except KeyError as e:
        raise ValueError(
            f"Missing placeholder value: {e}"
        ) from e
    leftover = re.findall(r"\{(\w+)", result)
    if leftover:
        raise ValueError(
            f"Unfilled placeholders in rendered prompt: {leftover}"
        )
    return result


# ── Code extraction ───────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """Extract a fenced Python code block from *text*.

    Raises RuntimeError if no code block with at least 200 characters
    is found.
    """
    from .utils import extract_code_block
    code, _ = extract_code_block(text)
    if not code or len(code) < 200:
        raise RuntimeError(
            "Initial Simulator generation produced no extractable Python "
            "code block.  The model response did not contain a valid "
            "```python ... ``` fenced block with at least 200 characters."
        )
    return code


# ── Public API ────────────────────────────────────────────────────────────

def generate_blueprint(
    condition_name: str,
    signal_modality: str,
    *,
    model_client,
    enable_search: bool = False,
) -> str:
    """Generate a Scientific Blueprint via one-shot LLM call."""
    spec = _load_mod_spec("scientific_blueprint", signal_modality)
    prompt = _render(_BP_MAIN,
        condition_name=condition_name,
        signal_modality=signal_modality,
        modality_specification=spec,
    )

    tools = None
    if enable_search:
        tools = [model_client.types.Tool(
            google_search=model_client.types.GoogleSearch(),
        )]

    return model_client.generate_text(
        prompt, temperature=0.2, max_tokens=8192, tools=tools,
    )


def generate_initial_simulator(
    condition_name: str,
    signal_modality: str,
    *,
    model_client,
    output_dir: str,
    num_samples: int,
    duration: float,
    synth_sr: int,
    output_sr: int,
    enable_search: bool = False,
) -> str:
    """Generate an Initial Simulator script via one-shot LLM call.

    This is a STANDALONE generation — it does NOT use the Scientific
    Blueprint.  The model draws on its own biomedical and DSP knowledge.
    """
    spec_template = _load_mod_spec("initial_simulator", signal_modality)
    modality_specification = _render(spec_template,
        output_dir=output_dir,
        duration=duration,
        synth_sr=synth_sr,
        output_sr=output_sr,
        num_samples=num_samples,
    )

    prompt = _render(_SIM_MAIN,
        condition_name=condition_name,
        signal_modality=signal_modality,
        modality_specification=modality_specification,
        num_samples=num_samples,
    )

    return model_client.generate_code(
        prompt, temperature=0.3, max_tokens=65536,
        enable_search=enable_search,
    )


def initialize(
    output_dir: str,
    condition_name: str,
    signal_modality: str,
    *,
    model_client,
    num_samples: int,
    duration: float,
    synth_sr: int,
    output_sr: int,
    enable_search: bool = False,
) -> dict:
    """Generate and save a Scientific Blueprint + Initial Simulator.

    Parameters
    ----------
    output_dir:
        Directory receiving *scientific_blueprint.md*, *script_000.py*,
        and *initialization_config.json*.
    condition_name:
        Human-readable label (e.g. ``"Atrial Septal Defect"``).
    signal_modality:
        ``"audio"``, ``"ecg"``, or ``"ppg"``.
    model_client:
        Pre-configured :class:`ModelClient` instance (required).
    num_samples / duration / synth_sr / output_sr:
        Required modality-specific output configuration.
    Returns
    -------
    dict with keys ``blueprint_path``, ``blueprint``, ``script_path``,
    ``script``, ``condition_name``, ``config_path``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_dir = str(out / "generated")

    # 1. Scientific Blueprint
    blueprint = generate_blueprint(
        condition_name, signal_modality,
        model_client=model_client, enable_search=enable_search,
    )
    bp_path = out / "scientific_blueprint.md"
    bp_path.write_text(blueprint)

    # Validate Blueprint: must be substantial, well-formed, and complete.
    bp_issues = []
    if len(blueprint) < 1500:
        bp_issues.append(f"too short ({len(blueprint)} chars)")
    if not blueprint.rstrip().endswith((".", ")", "]", "---")):
        bp_issues.append("may be truncated (ends mid-sentence)")
    for heading in ("## 1. Condition Overview", "## 7. Reference Values"):
        if heading not in blueprint:
            bp_issues.append(f"missing required section: {heading}")
    if bp_issues:
        print(f"  ⚠ Blueprint validation warnings: {'; '.join(bp_issues)}")

    # 2. Initial Simulator (independent of the blueprint)
    script = generate_initial_simulator(
        condition_name, signal_modality,
        model_client=model_client, output_dir=gen_dir,
        num_samples=num_samples, duration=duration,
        synth_sr=synth_sr, output_sr=output_sr,
        enable_search=enable_search,
    )
    script_path = out / "script_000.py"
    script_path.write_text(script)

    # 3. Execute and validate the initial simulator.
    ok, stdout, stderr = run_script(str(script_path), gen_dir)
    if not ok:
        raise RuntimeError(
            f"script_000.py execution failed.\n\nSTDERR:\n{stderr[:1000]}"
        )
    artifacts = sorted(Path(gen_dir).glob("*"))
    if len(artifacts) != num_samples:
        raise RuntimeError(
            f"script_000.py produced {len(artifacts)} artifacts, "
            f"expected {num_samples}.  Generated directory: {gen_dir}"
        )
    for a in artifacts:
        if a.stat().st_size == 0:
            raise RuntimeError(f"Empty artifact: {a}")

    # 4. Initialization config
    config = {
        "condition_name": condition_name,
        "signal_modality": signal_modality,
        "num_samples": num_samples,
        "duration": duration,
        "synth_sr": synth_sr,
        "output_sr": output_sr,
        "enable_search": enable_search,
        "model_name": getattr(model_client, "model_name", ""),
        "blueprint_temperature": 0.2,
        "blueprint_max_tokens": 8192,
        "simulator_temperature": 0.3,
        "simulator_max_tokens": 65536,
        "created_at": datetime.now().isoformat(),
        "blueprint_path": str(bp_path),
        "script_path": str(script_path),
        "gen_dir": gen_dir,
        "expected_num_points": round(duration * output_sr),
        "validated": True,
        "artifact_count": len(artifacts),
    }
    config_path = out / "initialization_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    return {
        "condition_name": condition_name,
        "blueprint_path": str(bp_path),
        "blueprint": blueprint,
        "script_path": str(script_path),
        "script": script,
        "config_path": str(config_path),
    }
