"""
_orchestrator.py — Flat PUCT (ERA-Style) Tree Search Orchestrator
==================================================================
One Orchestrator instance manages exactly one search trajectory guided by
one Search Feedback Agent.

Per-iteration loop:

  SELECT  — pick a parent node via Flat PUCT (global, rank-based).
  REFINE  — call the Refiner with blueprint, diagnosis report, and
            mechanism library.
  EXECUTE — run the new script to generate signals.
  ANALYZE — run SearchFeedback.analyze() for score + discrepancy report.
  EXTRACT — if score improved enough, extract a reusable mechanism.

Selection is global over all existing nodes (not hierarchical descent).
Every node remains eligible for future expansion.

All artifacts live under one ``run_dir``:

  <run_dir>/
    tree.json
    mechanism_library.json
    candidates/          script_000.py, script_001.py, …
    generated/           simulator output
    logs/                trace_*.jsonl
"""

import hashlib
import os
from datetime import datetime
from typing import Optional

from ..model_client import RetryExhaustedError
from ..search_feedback import SearchFeedbackAgent
from ..refiner import Refiner
from ..mechanism_library import MechanismLibrary
from ..utils import TraceLogger, patch_sample_count, count_signal_artifacts

from . import _search_tree as _st
from . import _simulator_runner as _sim


def _derive_visual_seed(run_dir: str, node_idx: int) -> int:
    """Deterministic per-(run, node) seed for visual sample selection.

    The visual-feedback call used to hard-code ``seed=42``, so every node of
    every run showed the SAME three real reference samples for the whole
    search (confirmed in stored node_*.json metadata).  Instead derive a
    stable seed from the run directory + node index:

      * within a run the reference triple now rotates as the search advances
        (node_idx is unique per archived child), and
      * a re-launch of the same run_dir reproduces the exact same sequence,
        so visual evidence is pinned to the node it was created for — even
        across a resume (archived nodes keep their stored PNG; only new
        children draw new seeds).

    Returns an unsigned 32-bit int, a valid numpy Generator seed.
    """
    digest = hashlib.sha256(f"{run_dir}|{node_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0xFFFFFFFF


class Orchestrator:
    """Flat PUCT (ERA-style) tree search over simulator programs.

    Manages one search trajectory: select → refine → execute → analyze → extract.
    Selection is global over all nodes; every node remains eligible.
    """

    def __init__(self,
                 condition_name: str,
                 signal_modality: str,
                 feedback_agent: SearchFeedbackAgent,
                 island_key: str,
                 blueprint: str,
                 run_dir: str,
                 search_ref_dir: str,
                 *,
                 model_client,               # ModelClient (required)
                 c_puct: float = 0.25,
                 select_mode: str = "puct",
                 report_mode: str = "full",
                 iterations: int = 30,
                 enable_search: bool = False,
                 min_delta_for_mechanism: float = 0.02,
                 num_samples: int,
                 duration: float = 10.0,
                 synth_sr: int = 44100,
                 output_sr: int = 16000,
                 visual_feedback: bool = False,
                 ):
        self.condition_name = condition_name
        self.signal_modality = signal_modality
        self.feedback = feedback_agent
        self.island_key = island_key

        if feedback_agent.agent_id and feedback_agent.agent_id != island_key:
            raise ValueError(
                f"island_key={island_key!r} does not match "
                f"feedback_agent.agent_id={feedback_agent.agent_id!r}"
            )

        self.blueprint = blueprint
        self.run_dir = run_dir
        self.search_ref_dir = search_ref_dir
        os.makedirs(self.run_dir, exist_ok=True)

        if not os.path.isdir(self.search_ref_dir):
            raise FileNotFoundError(
                f"Search reference directory not found: {self.search_ref_dir}"
            )

        self.model_client = model_client
        self.c_puct = c_puct
        if select_mode not in ("puct", "greedy"):
            raise ValueError(
                f"select_mode must be 'puct' or 'greedy', got {select_mode!r}")
        self.select_mode = select_mode
        self.search_algorithm = (_st.SEARCH_ALGORITHM_GREEDY if select_mode == "greedy"
                                 else _st.SEARCH_ALGORITHM)
        if report_mode not in ("full", "none"):
            raise ValueError(
                f"report_mode must be 'full' or 'none', got {report_mode!r}")
        # report_mode="none" is the no-report ablation: the Numerical
        # Discrepancy Report is still computed and stored in every node (so
        # the score, the selection and the lineage audit are unchanged), but
        # the Refiner never puts it in the prompt.  See Refiner.report_mode.
        self.report_mode = report_mode
        self.iterations = iterations
        self.enable_search = enable_search
        self.min_delta_for_mechanism = min_delta_for_mechanism
        self.num_samples = num_samples
        self.duration = duration
        self.synth_sr = synth_sr
        self.output_sr = output_sr
        self.visual_feedback = visual_feedback

        # ── Per-run state (initialised on first run()) ───────────────────
        self.candidates_dir: str = ""
        self.gen_dir: str = ""
        self.tree_path: str = ""
        self.mech_lib_path: str = ""
        self.nodes: list[dict] = []
        self.mech_lib: Optional[MechanismLibrary] = None
        self.refiner: Optional[Refiner] = None
        self.trace: Optional[TraceLogger] = None
        self._last_parent: Optional[dict] = None
        self._initialised = False

        # ── Authoring-attempt accounting ─────────────────────────────────
        # Attempts (refinements) are the scientific budget.  Only valid,
        # evaluated candidates become tree nodes.
        self.stats = {
            "requested_authoring_attempts": 0,
            "completed_authoring_attempts": 0,
            # Attempts completed by an earlier process before a provider or
            # infrastructure interruption.  This is restored from
            # pending.json so a resumed 100-attempt run finishes at 100
            # instead of incorrectly performing 100 additional attempts.
            "resumed_completed_attempts": 0,
            "valid_evaluated_candidates": 0,
            "invalid_code_attempts": 0,
            "script_execution_failures": 0,
            "output_contract_violations": 0,
            "invalid_signal_attempts": 0,
            "provider_retries": 0,
            "infrastructure_failures": 0,
        }

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, iterations: Optional[int] = None) -> dict:
        """Run the search trajectory and return the best result."""
        if iterations is None:
            iterations = self.iterations

        self._init_state()

        # Resume any pending stage from a previous interrupted run.
        self._resume_pending()

        self.stats["requested_authoring_attempts"] = iterations

        # An authoring attempt is consumed ONLY when an actual simulator
        # revision is attempted and reaches a definite outcome (valid or one of
        # the authoring-failure categories).  Infrastructure/provider failures
        # retry the same attempt slot and do NOT advance the budget.
        while self.stats["completed_authoring_attempts"] < iterations:
            attempt_idx = self.stats["completed_authoring_attempts"]
            self._last_parent = None
            try:
                self._run_iteration(attempt_idx, iterations)
            except RetryExhaustedError:
                # Transient provider failure — infrastructure, NOT an authoring
                # attempt.  Persist state for later resume.
                self.stats["provider_retries"] += 1
                self._save_tree()
                self._save_pending(attempt_idx,
                                   self._last_parent["script_idx"]
                                   if self._last_parent else -1)
                raise  # re-raise so the caller knows the run is incomplete
            except Exception as e:
                import traceback
                self.stats["infrastructure_failures"] += 1
                print(f"  Infrastructure failure on authoring attempt "
                      f"{attempt_idx+1} (retrying, not consuming budget): {e}")
                traceback.print_exc()
                self._save_tree()
                continue  # retry the same attempt slot
            self.stats["completed_authoring_attempts"] += 1

        self._finalize_stats()
        best = _st.best_node(self.nodes)
        best_script = _sim.candidate_path(
            self.candidates_dir, best["script_idx"],
        )
        print(f"\n  Search complete.  Best: {best_script}  "
              f"score={best['score']:.4f}")

        return {
            "best_score": best["score"],
            "best_script": best_script,
            "nodes": len(self.nodes),
            "tree_path": self.tree_path,
            "mech_lib_path": self.mech_lib_path,
        }

    # ── State initialisation ─────────────────────────────────────────────

    def _init_state(self) -> None:
        if self._initialised:
            return

        rd = self.run_dir
        self.candidates_dir = os.path.join(rd, "candidates")
        self.gen_dir = os.path.join(rd, "generated")
        self.tree_path = os.path.join(rd, "tree.json")
        self.mech_lib_path = os.path.join(rd, "mechanism_library.json")

        log_dir = os.path.join(rd, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.trace = TraceLogger(os.path.join(
            log_dir, f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        ))

        self.refiner = Refiner(
            condition_name=self.condition_name,
            signal_modality=self.signal_modality,
            blueprint=self.blueprint,
            model_client=self.model_client,
            enable_search=self.enable_search,
            island_key=self.island_key,
            trace_logger=self.trace,
            report_mode=self.report_mode,
        )

        # Pre-render the Immutable Output Contract from the same
        # modality specification used during Initial Simulator generation.
        from ..prompts import load_prompt
        mod_template = load_prompt(
            "initialization", "initial_simulator",
            "modalities", f"{self.signal_modality}.md",
        )
        self.signal_modality_specification = mod_template.format(
            output_dir=self.gen_dir,
            num_samples=self.num_samples,
            duration=self.duration,
            synth_sr=self.synth_sr,
            output_sr=self.output_sr,
        )

        self.mech_lib = MechanismLibrary(self.mech_lib_path,
                                         island_key=self.island_key)
        self.mech_lib.load()

        self._pending_path = os.path.join(rd, "pending.json")

        self.nodes = _st.load_tree(self.tree_path)
        if not self.nodes:
            self._cold_start()
        else:
            stored_n = self._stored_num_samples()
            if stored_n is not None and stored_n != self.num_samples:
                raise RuntimeError(
                    f"Run directory {self.run_dir} has stored generation "
                    f"contract num_samples={stored_n}, but this run requests "
                    f"num_samples={self.num_samples}. Refusing to resume under "
                    f"a mismatched generation contract."
                )
            print(f"  Loaded tree: {len(self.nodes)} nodes")
        self._initialised = True

    def _stored_num_samples(self) -> Optional[int]:
        """Read the generation contract recorded in tree.json (if any)."""
        if not os.path.exists(self.tree_path):
            return None
        try:
            import json
            with open(self.tree_path) as f:
                return json.load(f).get("num_samples")
        except Exception:
            return None

    def _cold_start(self) -> None:
        """Execute script_000.py, evaluate, and create the root node."""
        script_path = _sim.candidate_path(self.candidates_dir, 0)
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Initial simulator not found: {script_path}.  "
                f"Place script_000.py in {self.candidates_dir}/ before running."
            )

        print(f"  Cold start — evaluating {script_path}...")
        # Patch the output directory and enforce the generation contract so a
        # stale root (e.g. authored with N=96/472/20) does not silently run at
        # its old count.
        _sim.patch_output_dir(script_path, self.gen_dir)
        patch_sample_count(script_path, self.num_samples)
        ok, stdout, stderr = _sim.execute(script_path, self.gen_dir)
        if not ok:
            raise RuntimeError(f"script_000.py failed: {stderr[:500]}")

        generated_count = count_signal_artifacts(self.gen_dir, self.signal_modality)
        if generated_count != self.num_samples:
            raise RuntimeError(
                f"Cold start contract violation: script_000.py produced "
                f"{generated_count} signal artifacts, expected {self.num_samples}."
            )

        fb = self.feedback.analyze(self.search_ref_dir, self.gen_dir)

        gen_valid = fb.metadata.get("gen_stats", {}).get("n_valid",
                     fb.metadata.get("n_generated", -1))
        if gen_valid >= 0 and gen_valid < 3:
            raise RuntimeError(
                f"Cold start failed: script_000.py produced only "
                f"{gen_valid} valid output(s).  Need at least 3.  "
                f"Check the initial simulator and modality contract."
            )

        root = _st.make_root(fb.score, fb.report)
        if self.visual_feedback:
            root["visual_feedback_path"] = self._generate_visual(0)
        self.nodes = [root]
        self._save_tree()
        self.trace.log(
            agent="orchestrator", step="cold_start",
            iteration=0, parent_idx=-1, child_idx=0, score=fb.score,
        )
        print(f"  Seed score = {fb.score:.4f}")

    def _generate_visual(self, node_idx: int) -> str:
        """Render a real-vs-generated comparison figure for a node.

        The three real reference samples shown to the refiner are re-drawn per
        node via ``_derive_visual_seed`` (was: hard-coded seed 42 → the same
        three examples every step of every run).  Returns the PNG path, or ""
        on failure (visual feedback is optional and never blocks search).
        """
        try:
            from ..search_feedback.visualization import (
                create_visual_feedback, save_metadata,
            )
            vis_dir = os.path.join(self.run_dir, "visual")
            os.makedirs(vis_dir, exist_ok=True)
            png_path = os.path.join(vis_dir, f"node_{node_idx:03d}.png")
            meta_path = os.path.join(vis_dir, f"node_{node_idx:03d}.json")
            metadata = create_visual_feedback(
                self.signal_modality, self.search_ref_dir, self.gen_dir,
                png_path, n_samples=3,
                seed=_derive_visual_seed(self.run_dir, node_idx),
            )
            save_metadata(metadata, meta_path)
            return png_path
        except Exception as e:
            print(f"  [visual] feedback generation skipped: {e}")
            return ""

    # ── Parent selection ─────────────────────────────────────────────────

    def _select_parent(self) -> dict:
        """Pick the parent node for the next refinement.

        ``select_mode="puct"`` (default) — Flat PUCT over rank scores.
        ``select_mode="greedy"`` — ablation control: always the argmax-score
        node.  Everything else about the iteration (context, extraction,
        visual feedback, budget) is unchanged.
        """
        if self.select_mode == "greedy":
            return _st.greedy_select(self.nodes)
        return _st.puct_select(self.nodes, self.c_puct)

    # ── Single iteration ─────────────────────────────────────────────────

    def _run_iteration(self, i: int, total: int) -> None:
        print(f"\n--- Authoring attempt {i+1}/{total} ({self.island_key}) ---")

        # 1. SELECT
        parent = self._select_parent()
        self._last_parent = parent
        parent_path = _sim.candidate_path(
            self.candidates_dir, parent["script_idx"],
        )
        print(f"  Parent: script_{parent['script_idx']}.py  "
              f"score={parent['score']:.4f}  visits={parent['visits']}")

        with open(parent_path) as f:
            parent_code = f.read()

        # 2. Build context
        mechanism_context = self.mech_lib.format_for_prompt()

        # 3. REFINE
        # The Refiner owns report_mode: with report_mode="none" it ignores
        # this argument and substitutes a notice, so the report cannot reach
        # the prompt by accident.  It is still stored on the node below.
        try:
            new_code, preamble = self.refiner.refine(
                parent_idx=parent["script_idx"],
                parent_code=parent_code,
                feedback_report=parent.get("report", ""),
                modality_specification=self.signal_modality_specification,
                mechanism_context=mechanism_context,
                iteration=i,
                visual_feedback_path=parent.get("visual_feedback_path"),
            )
        except RetryExhaustedError:
            raise  # provider failure — handled by run()
        except Exception as e:
            self.stats["invalid_code_attempts"] += 1
            print(f"  Refiner produced no usable code (authoring failure): {e}")
            self._save_tree()
            return

        if not new_code or len(new_code) < 200:
            self.stats["invalid_code_attempts"] += 1
            print("  Refiner produced empty/invalid code. Skipping.")
            self._save_tree()
            return

        # 4. EXECUTE
        new_idx = _sim.next_candidate_index(self.candidates_dir)
        _sim.save_candidate(self.candidates_dir, new_idx, new_code,
                            self.gen_dir)
        print(f"  Saved: candidates/script_{new_idx}.py "
              f"({len(new_code)} chars)")

        ok, stdout, stderr = _sim.execute(
            _sim.candidate_path(self.candidates_dir, new_idx),
            self.gen_dir,
        )
        if not ok:
            self.stats["script_execution_failures"] += 1
            print(f"  Script execution failed:\n{stderr[:300]}")
            self._save_tree()
            return

        # ── Output-count validation (signal artifacts only) ──────────────
        generated_count = count_signal_artifacts(self.gen_dir, self.signal_modality)
        if generated_count != self.num_samples:
            self.stats["output_contract_violations"] += 1
            print(f"  Generation contract violation: expected "
                  f"{self.num_samples} signal artifacts, found "
                  f"{generated_count}. Discarding candidate.")
            self._save_tree()
            return

        # 5. ANALYZE
        fb = self.feedback.analyze(self.search_ref_dir, self.gen_dir)
        score = fb.score

        gen_valid = fb.metadata.get("gen_stats", {}).get("n_valid",
                     fb.metadata.get("n_generated", -1))
        if gen_valid >= 0 and gen_valid < 3:
            self.stats["invalid_signal_attempts"] += 1
            print(f"  Only {gen_valid} valid output(s) — discarding candidate.")
            self._save_tree()
            return

        delta = score - parent["score"]
        print(f"  score={score:.4f}  delta={delta:+.4f}")

        # 6. ARCHIVE
        _st.add_child(self.nodes, parent,
                      script_idx=new_idx, score=score,
                      report=fb.report)
        self.stats["valid_evaluated_candidates"] += 1
        if self.visual_feedback:
            child = self.nodes[-1]
            child["visual_feedback_path"] = self._generate_visual(new_idx)
        self._save_tree()
        self._last_parent = None

        # 7. EXTRACT mechanism (best-effort).
        if delta >= self.min_delta_for_mechanism:
            print(f"  Extracting mechanism (delta={delta:.4f})...")
            try:
                mechanism = MechanismLibrary.extract_from_code(
                    parent_code, new_code, delta, new_idx,
                    model_client=self.model_client,
                    trace_logger=self.trace,
                    iteration=i, parent_idx=parent["script_idx"],
                )
                if mechanism:
                    self.mech_lib.add(mechanism)
                    self.mech_lib.save()
                    print(f"  Mechanism added: {mechanism.name}")
            except Exception as e:
                print(f"  Mechanism extraction failed: {e}")

        # Trace logging (best-effort).
        try:
            self.trace.log(
                agent="orchestrator", step="iteration",
                iteration=i, parent_idx=parent["script_idx"],
                child_idx=new_idx, score=score, delta=delta,
            )
        except Exception as e:
            print(f"  Trace logging failed: {e}")

    # ── Pending-state persistence ───────────────────────────────────────

    def _save_pending(self, iteration: int, parent_idx: int) -> None:
        import json
        with open(self._pending_path, "w") as f:
            json.dump({
                "pending": True,
                "iteration": iteration,
                "parent_idx": parent_idx,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)

    def _clear_pending(self) -> None:
        if os.path.exists(self._pending_path):
            os.remove(self._pending_path)

    def _resume_pending(self) -> None:
        import json
        if not os.path.exists(self._pending_path):
            return
        with open(self._pending_path) as f:
            p = json.load(f)
        completed = int(p["iteration"])
        if completed < 0 or completed > self.iterations:
            raise RuntimeError(
                f"Invalid pending iteration {completed} for an "
                f"{self.iterations}-attempt run: {self._pending_path}"
            )
        self.stats["resumed_completed_attempts"] = completed
        self.stats["completed_authoring_attempts"] = completed
        print(f"  Resuming pending state: iteration={p['iteration']}, "
              f"parent={p['parent_idx']}")
        self._clear_pending()

    # ── Tree persistence ─────────────────────────────────────────────────

    def _save_tree(self) -> None:
        _st.save_tree(self.tree_path, self.nodes, {
            "condition": self.condition_name,
            "signal_modality": self.signal_modality,
            "island": self.island_key,
            "num_samples": self.num_samples,
            "report_mode": self.report_mode,
            "select_mode": self.select_mode,
        }, search_algorithm=self.search_algorithm)

    def _finalize_stats(self) -> None:
        """Derive completed authoring attempts and persist run statistics."""
        self.stats["completed_authoring_attempts"] = (
            self.stats["resumed_completed_attempts"]
            + self.stats["valid_evaluated_candidates"]
            + self.stats["invalid_code_attempts"]
            + self.stats["script_execution_failures"]
            + self.stats["output_contract_violations"]
            + self.stats["invalid_signal_attempts"]
        )
        import json
        stats_path = os.path.join(self.run_dir, "run_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"  Run stats → {stats_path}")
