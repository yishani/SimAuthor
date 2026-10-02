#!/usr/bin/env python3
"""CLI entry point for running one SimAuthor search trajectory.

Prepares an isolated run directory from initialization artifacts, then
constructs and runs one Orchestrator instance.

Usage::

    python -m Authoring.scripts.run_search \\
        --condition "Atrial Septal Defect" \\
        --signal-modality audio \\
        --initialization-dir artifacts/asd_audio/initialization \\
        --run-dir artifacts/asd_audio/runs/signal/seed_000 \\
        --feedback-agent mfcc \\
        --search-ref-dir conditions/asd_audio/search_reference \\
        --iterations 30
"""

import argparse

# All agents are registered when the package is imported.
from Authoring.search_feedback import create_feedback
from Authoring.model_client import ModelClient
from Authoring.orchestrator import Orchestrator
from ._run_prep import prepare_run_dir


def main():
    parser = argparse.ArgumentParser(
        description="SimAuthor — Flat PUCT Tree Search for one feedback agent",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--signal-modality", required=True,
                        choices=["audio", "ecg", "ppg"])
    parser.add_argument("--initialization-dir", required=True,
                        help="Directory containing scientific_blueprint.md and script_000.py")
    parser.add_argument("--run-dir", required=True,
                        help="Directory for this trajectory (tree.json, candidates/, generated/, logs/)")
    parser.add_argument("--feedback-agent", required=True,
                        help="Agent key (e.g. mfcc, representation)")
    parser.add_argument("--search-ref-dir", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--c-puct", type=float, default=0.25)
    parser.add_argument("--select-mode", choices=["puct", "greedy"],
                        default="puct",
                        help="Parent selection: Flat PUCT (default) or greedy "
                             "argmax-score (ablation control; c_puct unused).")
    parser.add_argument("--report-mode", choices=["full", "none"],
                        default="full",
                        help="Numerical Discrepancy Report: 'full' (default) "
                             "forwards it to the refiner; 'none' is the "
                             "no-report ablation control, which keeps the "
                             "blueprint, mechanism library, visual comparison, "
                             "score and PUCT selection unchanged.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--enable-search", action="store_true")
    parser.add_argument("--visual-feedback", action="store_true",
                        help="Enable multimodal visual feedback (default off)")
    # ── Output / execution configuration ─────────────────────────────────
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--synth-sr", type=int, required=True)
    parser.add_argument("--output-sr", type=int, required=True)
    parser.add_argument("--min-delta-for-mechanism", type=float, default=0.02)
    args = parser.parse_args()

    # 1. Prepare the isolated run directory.
    prep = prepare_run_dir(
        initialization_dir=args.initialization_dir,
        run_dir=args.run_dir,
        condition_name=args.condition,
        signal_modality=args.signal_modality,
        feedback_agent=args.feedback_agent,
        model_name=args.model,
        iterations=args.iterations,
        c_puct=args.c_puct,
        select_mode=args.select_mode,
        report_mode=args.report_mode,
        num_samples=args.num_samples,
        duration=args.duration,
        synth_sr=args.synth_sr,
        output_sr=args.output_sr,
        min_delta_for_mechanism=args.min_delta_for_mechanism,
        enable_search=args.enable_search,
    )

    # 2. Construct the feedback agent and orchestrator.
    mc = ModelClient(model_name=args.model)
    feedback = create_feedback(args.signal_modality, args.feedback_agent)

    orch = Orchestrator(
        condition_name=args.condition,
        signal_modality=args.signal_modality,
        feedback_agent=feedback,
        island_key=args.feedback_agent,
        blueprint=prep["blueprint"],
        run_dir=args.run_dir,
        search_ref_dir=args.search_ref_dir,
        model_client=mc,
        c_puct=args.c_puct,
        select_mode=args.select_mode,
        report_mode=args.report_mode,
        iterations=args.iterations,
        enable_search=args.enable_search,
        min_delta_for_mechanism=args.min_delta_for_mechanism,
        num_samples=args.num_samples,
        duration=args.duration,
        synth_sr=args.synth_sr,
        output_sr=args.output_sr,
        visual_feedback=args.visual_feedback,
    )
    result = orch.run()

    print(f"\nSearch complete.")
    print(f"  Best score:  {result['best_score']:.4f}")
    print(f"  Best script: {result['best_script']}")
    print(f"  Nodes:       {result['nodes']}")


if __name__ == "__main__":
    main()
