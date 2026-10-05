"""
cli.py — ``simauthor`` command-line interface
===============================================

    simauthor conditions                       list the six paper conditions
    simauthor generate COPD --out samples/     run a released simulator (no API key)
    simauthor evaluate --modality audio --ref REF --gen GEN
    simauthor run --condition "..." --modality audio --ref REF --out runs/x
    simauthor init ... / simauthor search ...  the two stages of `run`, separately

Run ``simauthor <command> --help`` for all options.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import presets
from .config import DEFAULT_MODEL

SIMULATORS_DIR = Path(__file__).resolve().parent / "simulators"


# ── helpers ──────────────────────────────────────────────────────────────

def _contract(args) -> dict:
    """Fill duration / sample rates from the modality preset unless given."""
    p = presets.MODALITIES[args.modality]
    return dict(
        num_samples=args.num_samples,
        duration=args.duration if args.duration is not None else p.duration,
        synth_sr=args.synth_sr if args.synth_sr is not None else p.synth_sr,
        output_sr=args.output_sr if args.output_sr is not None else p.output_sr,
    )


def _feedback(args):
    """Built-in evaluator by key, or a user evaluator file via --evaluator."""
    from .search_feedback import create_feedback, load_evaluator
    kwargs = json.loads(args.feedback_kwargs) if args.feedback_kwargs else {}
    if getattr(args, "evaluator", None):
        fb = load_evaluator(args.evaluator, **kwargs)
        return fb.agent_id, fb
    agent = args.feedback_agent or presets.MODALITIES[args.modality].feedback_agent
    return agent, create_feedback(args.modality, agent, **kwargs)


def _add_contract_args(p):
    p.add_argument("--modality", "--signal-modality", dest="modality",
                   required=True, choices=sorted(presets.MODALITIES))
    p.add_argument("--num-samples", type=int, default=presets.NUM_SAMPLES,
                   help="records generated per evaluation (paper: 100)")
    p.add_argument("--duration", type=float, help="seconds per record "
                   "(default from modality: audio 10, ppg 30, ecg 10)")
    p.add_argument("--synth-sr", type=int, help="internal synthesis rate")
    p.add_argument("--output-sr", type=int, help="saved sampling rate "
                   "(default: audio 16000, ppg 125, ecg 500)")


def _add_model_args(p):
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="gemini-* (GEMINI_API_KEY), deepseek-* / any "
                        "OpenAI-compatible model (OPENAI_API_KEY, "
                        "OPENAI_BASE_URL), or 'mock' for an offline dry run")
    p.add_argument("--enable-search", action="store_true",
                   help="Google Search grounding (Gemini only)")


def _add_search_args(p):
    p.add_argument("--evaluator", metavar="FILE.py",
                   help="your own evaluator (a SearchFeedbackAgent subclass); "
                        "see examples/custom_task/my_evaluator.py")
    p.add_argument("--feedback-agent", help="built-in evaluator key (default from "
                   "modality: mfcc / morphology / ecg_founder_12lead)")
    p.add_argument("--feedback-kwargs", help="JSON kwargs for the evaluator, "
                   "e.g. '{\"checkpoint_path\": \"...\"}'")
    p.add_argument("--iterations", type=int, default=presets.ITERATIONS,
                   help="authoring attempts (paper: 100)")
    p.add_argument("--c-puct", type=float, default=presets.C_PUCT)
    p.add_argument("--select-mode", choices=["puct", "greedy"], default="puct")
    p.add_argument("--report-mode", choices=["full", "none"], default="full",
                   help="'none' = the no-report ablation")
    p.add_argument("--no-visual-feedback", dest="visual_feedback",
                   action="store_false",
                   help="disable the real-vs-generated figure sent to the refiner")
    p.add_argument("--min-delta-for-mechanism", type=float,
                   default=presets.MIN_DELTA_FOR_MECHANISM)


# ── commands ─────────────────────────────────────────────────────────────

def cmd_conditions(args):
    print(presets.describe())
    print(f"\nReleased simulators: {SIMULATORS_DIR}")


def _resolve_simulator(target: str, which: str) -> Path:
    path = Path(target)
    if path.suffix == ".py" and path.is_file():
        return path
    cond = presets.get_condition(target)
    path = SIMULATORS_DIR / cond.key / f"{which}.py"
    if not path.is_file():
        sys.exit(f"Simulator not found: {path}.")
    return path


def cmd_generate(args):
    from .utils import patch_output_dir, patch_sample_count, run_script
    src = _resolve_simulator(args.target, args.which)
    out = Path(args.out or f"samples/{Path(args.target).stem}_{args.which}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / src.name
        shutil.copy(src, script)
        if not patch_output_dir(str(script), str(out)):
            sys.exit(f"{src}: no OUTPUT_DIR / output_dir assignment to patch.")
        if args.num_samples is not None:
            patch_sample_count(str(script), args.num_samples)
        print(f"Running {src} -> {out}")
        ok, stdout, stderr = run_script(str(script), timeout=args.timeout)
    if stdout.strip():
        print(stdout.strip()[-2000:])
    if not ok:
        sys.exit(f"Simulator failed:\n{stderr[-3000:]}")
    n = sum(1 for f in out.iterdir() if f.suffix in (".wav", ".npy"))
    print(f"Wrote {n} records to {out}")


def cmd_evaluate(args):
    agent, fb = _feedback(args)
    result = fb.analyze(args.ref, args.gen, include_report=not args.score_only)
    if args.json:
        print(json.dumps({"agent": agent, "score": result.score,
                          "summary": result.summary, "report": result.report},
                         indent=2))
    else:
        if result.report:
            print(result.report)
        print(f"\nscore ({agent}) = {result.score:.4f}")


def _init(args, out_dir: str) -> dict:
    from .model_client import create_model_client
    from .initialization import initialize
    return initialize(
        output_dir=out_dir,
        condition_name=args.condition,
        signal_modality=args.modality,
        model_client=create_model_client(args.model),
        enable_search=args.enable_search,
        **_contract(args),
    )


def _search(args, init_dir: str, run_dir: str) -> dict:
    from .model_client import create_model_client
    from .orchestrator import Orchestrator
    from .run_dir import prepare_run_dir
    agent, fb = _feedback(args)
    contract = _contract(args)
    prep = prepare_run_dir(
        initialization_dir=init_dir, run_dir=run_dir,
        condition_name=args.condition, signal_modality=args.modality,
        feedback_agent=agent, model_name=args.model,
        iterations=args.iterations, c_puct=args.c_puct,
        select_mode=args.select_mode, report_mode=args.report_mode,
        min_delta_for_mechanism=args.min_delta_for_mechanism,
        enable_search=args.enable_search, **contract,
    )
    orch = Orchestrator(
        condition_name=args.condition, signal_modality=args.modality,
        feedback_agent=fb, island_key=agent, blueprint=prep["blueprint"],
        run_dir=run_dir, search_ref_dir=args.ref,
        model_client=create_model_client(args.model),
        c_puct=args.c_puct, select_mode=args.select_mode,
        report_mode=args.report_mode, iterations=args.iterations,
        enable_search=args.enable_search,
        min_delta_for_mechanism=args.min_delta_for_mechanism,
        visual_feedback=args.visual_feedback, **contract,
    )
    result = orch.run()
    print(f"\nSearch complete.\n  Best score:  {result['best_score']:.4f}"
          f"\n  Best script: {result['best_script']}\n  Nodes:       {result['nodes']}")
    return result


def cmd_init(args):
    r = _init(args, args.out)
    print(f"\nBlueprint: {r['blueprint_path']}\nSimulator: {r['script_path']}")


def cmd_search(args):
    _search(args, args.init_dir, args.run_dir)


def cmd_run(args):
    out = Path(args.out)
    init_dir, run_dir = out / "initialization", out / "run"
    if (init_dir / "script_000.py").is_file() and (init_dir / "scientific_blueprint.md").is_file():
        print(f"Reusing initialization in {init_dir}")
    else:
        _init(args, str(init_dir))
    _search(args, str(init_dir), str(run_dir))


# ── parser ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="simauthor",
        description="SimAuthor: persistent authoring of executable scientific simulators.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("conditions", help="list the six paper conditions")
    p.set_defaults(func=cmd_conditions)

    p = sub.add_parser("generate", help="run a released simulator (no API key needed)")
    p.add_argument("target", help="condition key (VSD, AS, COPD, AF, LQT, WPW) or path to a script")
    p.add_argument("--which", choices=["authored", "root"], default="authored",
                   help="authored = best program of the median seed; root = S(0)")
    p.add_argument("--out", help="output directory (default: samples/<target>_<which>)")
    p.add_argument("-n", "--num-samples", type=int, help="records to generate (default: the script's own, 100)")
    p.add_argument("--timeout", type=int, default=1800)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="score a generated set against a reference set")
    p.add_argument("--modality", required=True, choices=sorted(presets.MODALITIES))
    p.add_argument("--ref", required=True, help="directory of reference .wav / .npy records")
    p.add_argument("--gen", required=True, help="directory of generated records")
    p.add_argument("--evaluator", metavar="FILE.py", help="your own evaluator file")
    p.add_argument("--feedback-agent")
    p.add_argument("--feedback-kwargs")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    common = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p = sub.add_parser("run", help="initialize + search in one go", **common)
    p.add_argument("--condition", required=True, help='e.g. "Aortic Stenosis"')
    p.add_argument("--ref", required=True, help="search reference directory")
    p.add_argument("--out", required=True, help="output directory (initialization/ and run/)")
    _add_contract_args(p); _add_model_args(p); _add_search_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("init", help="generate the scientific blueprint and root simulator", **common)
    p.add_argument("--condition", required=True)
    p.add_argument("--out", required=True)
    _add_contract_args(p); _add_model_args(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("search", help="run the authoring search from an initialization", **common)
    p.add_argument("--condition", required=True)
    p.add_argument("--init-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ref", required=True)
    _add_contract_args(p); _add_model_args(p); _add_search_args(p)
    p.set_defaults(func=cmd_search)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
