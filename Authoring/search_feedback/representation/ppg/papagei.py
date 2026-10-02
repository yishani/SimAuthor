"""PaPaGei-S representation evaluator for 125 Hz, 30 s PPG records.

Each record is filtered once, split into three 10-second windows, normalized
per window as in PaPaGei feature extraction, and represented by the mean of
its three embeddings. The exact input contract is enforced so an incorrect
sample rate cannot be rewarded as a short waveform fragment.
"""

import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from ...base import SearchFeedbackAgent, SearchFeedback
from ... import register_feedback
from ..comparison import compare, build_feedback, DEFAULT_SCALE


class PPGRepresentation(SearchFeedbackAgent):
    """Compare PPG sets in PaPaGei-S representation space."""

    agent_id = "papagei"
    comparison_space = "representation"
    modality = "ppg"

    SR = 125
    CHUNK_SAMPLES = 1250
    SEGMENT_SAMPLES = 3750
    N_CHUNKS = 3
    EMBEDDING_DIM = 512

    def __init__(self, name: str = "PPG Representation (PaPaGei-S)",
                 papagei_dir: str = "", weights_path: str = "",
                 scale: float = DEFAULT_SCALE, batch_size: int = 64):
        super().__init__(name=name)
        self._papagei_dir = papagei_dir
        self._weights_path = weights_path
        self.scale = scale
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _init_model(self):
        if self._model is not None:
            return
        if self._papagei_dir and self._papagei_dir not in sys.path:
            sys.path.insert(0, self._papagei_dir)
        if not self._weights_path or not Path(self._weights_path).is_file():
            raise FileNotFoundError(f"PaPaGei-S checkpoint not found: {self._weights_path}")

        from models.resnet import ResNet1DMoE
        from linearprobing.utils import load_model_without_module_prefix
        import torch

        model = ResNet1DMoE(
            in_channels=1, base_filters=32, kernel_size=3, stride=2,
            groups=1, n_block=18, n_classes=self.EMBEDDING_DIM, n_experts=3,
        )
        model = load_model_without_module_prefix(model, self._weights_path)
        model.requires_grad_(False)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device).eval()

    @classmethod
    def _preprocess(cls, segment: np.ndarray) -> np.ndarray:
        """Validate, filter, split, and normalize one 30-second PPG record."""
        from scipy.signal import cheby2, sosfiltfilt

        signal = np.asarray(segment).squeeze()
        if signal.ndim != 1:
            raise ValueError("PPG array must contain exactly one channel")
        if signal.size != cls.SEGMENT_SAMPLES:
            raise ValueError(
                f"PPG must contain exactly {cls.SEGMENT_SAMPLES} samples at "
                f"{cls.SR} Hz; got {signal.size}. Resample explicitly before evaluation."
            )
        signal = signal.astype(np.float64, copy=False)
        if not np.isfinite(signal).all():
            raise ValueError("PPG contains NaN or infinite values")
        if signal.std() < 1e-8:
            raise ValueError("PPG is constant or nearly constant")

        # PaPaGei documents pyPPG's fourth-order Chebyshev-II 0.5--12 Hz
        # preprocessing for raw PPG. SOS form is numerically stable.
        sos = cheby2(4, 20, [0.5, 12.0], btype="bandpass",
                     fs=cls.SR, output="sos")
        signal = sosfiltfilt(sos, signal)
        chunks = signal.reshape(cls.N_CHUNKS, cls.CHUNK_SAMPLES)
        means = chunks.mean(axis=1, keepdims=True)
        stds = chunks.std(axis=1, keepdims=True)
        if np.any(stds < 1e-8) or not np.isfinite(chunks).all():
            raise ValueError("PPG has a constant/invalid 10-second chunk")
        return ((chunks - means) / stds).astype(np.float32, copy=False)

    def _embed_with_audit(self, directory: str):
        self._init_model()
        import torch

        files = sorted(Path(directory).glob("*.npy"))
        records, errors = [], Counter()
        for path in files:
            try:
                records.append(self._preprocess(np.load(path, allow_pickle=False)))
            except Exception as exc:
                errors[f"{type(exc).__name__}: {exc}"] += 1

        chunk_embeddings = []
        if records:
            chunks = np.stack(records).reshape(-1, self.CHUNK_SAMPLES)
            for start in range(0, len(chunks), self.batch_size):
                x = torch.from_numpy(chunks[start:start + self.batch_size])
                x = x.unsqueeze(1).to(self._device)
                with torch.inference_mode():
                    outputs = self._model(x)
                output = outputs[0].detach().cpu().numpy()
                if (output.ndim != 2 or output.shape[1] != self.EMBEDDING_DIM
                        or not np.isfinite(output).all()):
                    raise ValueError(f"invalid PaPaGei embeddings: {output.shape}")
                chunk_embeddings.append(output.astype(np.float32, copy=False))
        if chunk_embeddings:
            chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)
            embeddings = chunk_embeddings.reshape(
                len(records), self.N_CHUNKS, self.EMBEDDING_DIM).mean(axis=1)
        else:
            embeddings = np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        audit = {"total_files": len(files), "embedded_files": len(embeddings),
                 "skipped_files": len(files) - len(embeddings),
                 "errors": dict(errors)}
        if errors:
            warnings.warn(f"PaPaGei skipped {audit['skipped_files']}/{len(files)} "
                          f"files in {directory}: {dict(errors)}", RuntimeWarning)
        array = np.asarray(embeddings, dtype=np.float32)
        return array, audit

    def _embed(self, directory: str) -> np.ndarray:
        """Embed each valid file; retained for existing analysis scripts."""
        return self._embed_with_audit(directory)[0]

    def analyze(self, reference_dir: str, generated_dir: str, *,
                include_report: bool = True) -> SearchFeedback:
        real_emb, real_audit = self._embed_with_audit(reference_dir)
        gen_emb, gen_audit = self._embed_with_audit(generated_dir)
        audit_metadata = {"reference_input": real_audit,
                          "generated_input": gen_audit,
                          "sampling_rate_hz": self.SR,
                          "duration_s": self.SEGMENT_SAMPLES / self.SR,
                          "chunks_per_record": self.N_CHUNKS}

        if real_emb.shape[0] < 3 or gen_emb.shape[0] < 3:
            return SearchFeedback(
                score=0.0, summary="Insufficient valid PPG records (< 3 each).",
                report=(f"Valid reference records: {real_emb.shape[0]}/"
                        f"{real_audit['total_files']}; valid generated records: "
                        f"{gen_emb.shape[0]}/{gen_audit['total_files']}. Need at "
                        "least 3 each. Expected one-channel PPG, 125 Hz, 30 s."),
                metadata=audit_metadata,
            )

        stats = compare(real_emb, gen_emb, scale=self.scale)
        if not include_report:
            return SearchFeedback(score=stats["score"],
                                  summary=f"Score: {stats['score']:.4f}",
                                  metadata={**stats, **audit_metadata})

        feedback = build_feedback(
            stats,
            title="PPG Representation-Space Discrepancy Report (PaPaGei-S)",
            encoder_name="PaPaGei-S (3 x 10 s, record-mean)",
            n_real=real_emb.shape[0], n_gen=gen_emb.shape[0],
            embedding_dim=real_emb.shape[1],
        )
        feedback.report += (
            "\n\n### Input audit\n"
            f"- Contract: one-channel PPG, {self.SR} Hz, 30 s "
            f"({self.SEGMENT_SAMPLES} samples).\n"
            f"- Reference: {real_audit['embedded_files']}/"
            f"{real_audit['total_files']} files embedded.\n"
            f"- Generated: {gen_audit['embedded_files']}/"
            f"{gen_audit['total_files']} files embedded."
        )
        for side, audit in (("Reference", real_audit),
                            ("Generated", gen_audit)):
            for reason, count in audit["errors"].items():
                feedback.report += f"\n- {side} skipped {count}: {reason}"
        feedback.metadata.update(audit_metadata)
        return feedback


register_feedback("ppg", "papagei", PPGRepresentation)
