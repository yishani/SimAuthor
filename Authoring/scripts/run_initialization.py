#!/usr/bin/env python3
"""CLI entry point for SimAuthor initialization.

Generates a Scientific Blueprint and an Initial Simulator for a given
condition and saves them to *output_dir*.

Usage::

    python -m Authoring.scripts.run_initialization \\
        --condition "Atrial Septal Defect" \\
        --signal-modality audio \\
        --output-dir artifacts/asd_audio/initialization \\
        --duration 10.0 --synth-sr 44100 --output-sr 16000
"""

import argparse

from Authoring.model_client import ModelClient
from Authoring.initialization import initialize


def main():
    parser = argparse.ArgumentParser(
        description="SimAuthor — Generate Scientific Blueprint + Initial Simulator",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--signal-modality", required=True,
                        choices=["audio", "ecg", "ppg"])
    parser.add_argument("--output-dir", required=True,
                        help="Directory for generated artifacts")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--enable-search", action="store_true")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True,
                        help="Duration per sample in seconds")
    parser.add_argument("--synth-sr", type=int, required=True,
                        help="Internal synthesis sample rate (Hz)")
    parser.add_argument("--output-sr", type=int, required=True,
                        help="Output sample rate (Hz)")
    args = parser.parse_args()

    mc = ModelClient(model_name=args.model)
    result = initialize(
        output_dir=args.output_dir,
        condition_name=args.condition,
        signal_modality=args.signal_modality,
        model_client=mc,
        num_samples=args.num_samples,
        duration=args.duration,
        synth_sr=args.synth_sr,
        output_sr=args.output_sr,
        enable_search=args.enable_search,
    )

    print(f"\nInitialization complete for '{args.condition}'")
    print(f"  Blueprint:  {result['blueprint_path']}")
    print(f"  Simulator:  {result['script_path']}")
    print(f"  Config:     {result['config_path']}")


if __name__ == "__main__":
    main()
