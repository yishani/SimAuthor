"""
SimAuthor — Iterative Simulator Authoring Framework
====================================================
Agentic biomedical signal simulator authoring through multi-island tree
search with structured search feedback.  Supports audio (heart sounds,
lung sounds), ECG, and PPG modalities.

Architecture
------------
  initialization.py        — Scientific Blueprint + Initial Simulator generation
  orchestrator/          — Search Orchestrator
  refiner.py               — Context-aware LLM refinement
  search_feedback/         — Evaluators providing scalar scores + discrepancy reports
  mechanism_library.py     — Reusable mechanism extraction and storage

Quick Start
-----------
  from Authoring.model_client import ModelClient
  from Authoring.initialization import initialize
  from Authoring.search_feedback import create_feedback
  from Authoring.orchestrator import Orchestrator

  mc = ModelClient(model_name="gemini-3.1-pro-preview")

  init_result = initialize(
      output_dir="artifacts/ASD",
      condition_name="Atrial Septal Defect",
      signal_modality="audio",
      model_client=mc,
      num_samples=100, duration=10.0,
      synth_sr=44100, output_sr=16000,
  )

  # Run one trajectory — repeat for representation, signal
  feedback = create_feedback("audio", "mfcc")
  orch = Orchestrator(
      condition_name="Atrial Septal Defect",
      signal_modality="audio",
      feedback_agent=feedback, island_key="mfcc",
      blueprint=init_result["blueprint"],
      run_dir="artifacts/ASD/runs/mfcc/seed_000",
      search_ref_dir="conditions/ASD/search_ref",
      model_client=mc,
      num_samples=100,
  )
  result = orch.run(iterations=30)
"""

__version__ = "2.0.0"
__author__ = "SimAuthor Team"
