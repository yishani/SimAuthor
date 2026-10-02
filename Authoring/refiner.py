"""
refiner.py — Context-Aware Script Refinement
=============================================
Assembles refinement context and calls the LLM to produce one improved
simulator script via a single focused, hypothesis-driven mutation.

The Refiner receives a descriptive Numerical Discrepancy Report and
decides *how* to modify the simulator.  It does not receive raw search
scores, evaluator source code, or direct repair instructions.
"""

from typing import Optional

from .prompts import load_prompt, render_prompt
from .utils import extract_code_block, is_code_complete, TraceLogger

# ── Prompt template (loaded once at import time) ───────────────────────────

_MAIN_TEMPLATE = load_prompt("refiner", "main.md")

#: ``report_mode="none"`` ablation: identical prompt, except the Numerical
#: Discrepancy Report section carries this notice and the policy paragraphs
#: that would otherwise point the model at a report are redirected to the
#: remaining evidence (blueprint, contract, source, visual comparison).
_MAIN_TEMPLATE_NO_REPORT = load_prompt("refiner", "main_no_report.md")

_NO_REPORT_NOTICE = (
    "No numerical discrepancy report is available for this candidate.  "
    "Diagnose from the Scientific Reference, the Immutable Output Contract, "
    "the current simulator source, and the visual evidence when one is "
    "attached."
)

_REPORT_MODES = ("full", "none")


class Refiner:
    """Hypothesis-driven LLM refiner for biomedical signal simulators.

    Receives a parent script, a discrepancy report, and a modality
    specification; produces one improved child script via a single
    focused mutation.
    """

    def __init__(self,
                 condition_name: str,
                 signal_modality: str,
                 blueprint: str,
                 *,
                 model_client,
                 enable_search: bool = False,
                 island_key: str = "",
                 trace_logger: Optional[TraceLogger] = None,
                 report_mode: str = "full",
                 ):
        """
        Args:
            condition_name: e.g. ``"Atrial Septal Defect"``.
            signal_modality: ``"audio"``, ``"ecg"``, or ``"ppg"``.
            blueprint: Scientific Blueprint markdown text.
            model_client: Pre-configured :class:`ModelClient` instance.
            enable_search: Enable Google Search grounding.
            island_key: Optional label for logging.
            trace_logger: Optional :class:`TraceLogger`.
            report_mode: ``"full"`` (default) forwards the Numerical
                Discrepancy Report to the model.  ``"none"`` suppresses it —
                the ablation control that holds the blueprint, the mechanism
                library, the visual comparison, the score and the PUCT
                selection fixed and removes only the structured numerical
                evidence.  The report is still computed and stored in
                ``tree.json``; it simply never reaches the prompt.
        """
        if report_mode not in _REPORT_MODES:
            raise ValueError(
                f"report_mode must be one of {_REPORT_MODES}, "
                f"got {report_mode!r}")
        self.condition_name = condition_name
        self.signal_modality = signal_modality
        self.blueprint = blueprint
        self.island_key = island_key
        self.enable_search = enable_search
        self.trace = trace_logger
        self.model_client = model_client
        self.report_mode = report_mode

    # ── Main entry point ─────────────────────────────────────────────────

    def refine(self, *,
               parent_idx: int,
               parent_code: str,
               feedback_report: str,
               modality_specification: str,
               mechanism_context: str = "",
               iteration: int = -1,
               visual_feedback_path: str = None,
               ) -> tuple[str, str]:
        """Produce one refined simulator script.

        Parameters
        ----------
        parent_idx:
            Index of the parent script (for display only).
        parent_code:
            Complete Python source of the parent simulator.
        feedback_report:
            Structured Numerical Discrepancy Report from SearchFeedback.
        modality_specification:
            Pre-rendered Immutable Output Contract (file format, sampling
            rates, duration, sample count, channel/lead structure, units,
            filename convention).
        mechanism_context:
            Formatted Mechanism Library section (may be empty).

        Returns
        -------
        ``(code, preamble)`` — the refined Python script and any
        LLM reasoning preamble preceding the code block.
        """
        template = _MAIN_TEMPLATE
        if self.report_mode == "none":
            template = _MAIN_TEMPLATE_NO_REPORT
            feedback_report = _NO_REPORT_NOTICE

        prompt = render_prompt(template,
            condition_name=self.condition_name,
            signal_modality=self.signal_modality,
            blueprint=self.blueprint,
            parent_idx=parent_idx,
            parent_code=parent_code,
            feedback_report=feedback_report,
            modality_specification=modality_specification,
            mechanism_context=mechanism_context,
        )

        # Optional multimodal visual feedback — complementary evidence only.
        image_paths = None
        if visual_feedback_path:
            prompt += (
                "\n\n## VISUAL EVIDENCE\n"
                "The attached figure shows representative real reference "
                "samples and samples produced by the current simulator. Use "
                "this visual evidence together with the scientific blueprint "
                + ("and the biomedical specification " if self.report_mode == "none"
                   else "and the quantitative discrepancy report ")
                + "when diagnosing the simulator. Do not optimize solely for "
                "visual resemblance.\n"
            )
            image_paths = [visual_feedback_path]

        mc = self.model_client
        tools = None
        if self.enable_search:
            tools = [mc.types.Tool(google_search=mc.types.GoogleSearch())]

        print(f"  Refiner: calling LLM (island={self.island_key}, "
              f"prompt={len(prompt)} chars)...")

        kwargs = dict(prompt=prompt, temperature=0.3, max_tokens=65536,
                      tools=tools)
        if image_paths:
            # Only pass image_paths to backends that support multimodal input.
            kwargs["image_paths"] = image_paths
        text = mc.generate_text(**kwargs)
        code, preamble = extract_code_block(text)

        if not code or not is_code_complete(code):
            raise RuntimeError(
                "Refiner: model response contained no complete, "
                "executable Python script."
            )

        if self.trace:
            self.trace.log(
                agent="refiner", step="refine",
                iteration=iteration, parent_idx=parent_idx,
                prompt=prompt, response=text,
                temperature=0.3, max_tokens=65536,
                model_name=self.model_client.model_name,
                tools="google_search" if self.enable_search else "",
            )

        return code, preamble
