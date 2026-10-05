"""
SimAuthor — persistent authoring of executable scientific simulators
=====================================================================
A foundation model writes a simulator program, the harness executes it,
compares its outputs with a small set of real recordings, and the model
revises the program again — keeping every version in a search tree and
distilling successful edits into a reusable mechanism library.

Architecture
------------
  initialization.py      Scientific blueprint + root simulator S(0)
  orchestrator/          Flat-PUCT search over simulator programs
  refiner.py             Context-aware, hypothesis-driven LLM revision
  search_feedback/       Evaluators: scalar score + discrepancy report
  mechanism_library.py   Reusable mechanism extraction and storage
  model_client.py        Gemini / OpenAI-compatible / mock backends
  cli.py                 The ``simauthor`` command

Python API
----------
  from simauthor.model_client import create_model_client
  from simauthor.initialization import initialize
  from simauthor.search_feedback import create_feedback
  from simauthor.orchestrator import Orchestrator

  mc = create_model_client("gemini-3.1-pro-preview")
  init = initialize(output_dir="runs/as/initialization",
                    condition_name="Aortic Stenosis", signal_modality="audio",
                    model_client=mc, num_samples=100, duration=10.0,
                    synth_sr=44100, output_sr=16000)
  # copy init["script_path"] to runs/as/run/candidates/script_000.py
  # (simauthor.run_dir.prepare_run_dir does this), then:
  orch = Orchestrator(condition_name="Aortic Stenosis", signal_modality="audio",
                      feedback_agent=create_feedback("audio", "mfcc"),
                      island_key="mfcc", blueprint=init["blueprint"],
                      run_dir="runs/as/run", search_ref_dir="data/as_ref",
                      model_client=mc, num_samples=100, visual_feedback=True)
  result = orch.run(iterations=100)
"""

__version__ = "1.0.0"
__author__ = "Yishan Wang, Ran Piao, Mathias Funk, Aaqib Saeed"
