"""ECGFounder representation evaluator used for the LQT and WPW searches.

The released 12-lead checkpoint receives Lead I and Lead II in channels 0 and
1; the remaining ten channels are zero.  Unlike the historical implementation,
this frozen version rejects anything other than an exact 10 s, 500 Hz,
two-lead record.  This changes only invalid-input handling, not scores for
valid historical records.
"""

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ...base import SearchFeedbackAgent, SearchFeedback
from ... import register_feedback
from ..comparison import compare, build_feedback, DEFAULT_SCALE


class ECGFounder12LeadRepresentation(SearchFeedbackAgent):
    """12-lead ECGFounder with two observed and ten zero leads."""

    agent_id = "ecg_founder_12lead"
    comparison_space = "representation"
    modality = "ecg"

    SR = 500
    N_SAMPLES = 5000
    N_MODEL_LEADS = 12
    EMBEDDING_DIM = 1024

    def __init__(self, name="ECG Representation (ECGFounder 12-lead)",
                 repo_dir="", checkpoint_path="", scale=DEFAULT_SCALE,
                 batch_size=32):
        super().__init__(name=name)
        # Paths may also come from the environment (see README, "Encoders").
        self._repo_dir = repo_dir or os.environ.get("ECGFOUNDER_REPO", "")
        self._checkpoint_path = (checkpoint_path
                                 or os.environ.get("ECGFOUNDER_CKPT", ""))
        self.scale = scale
        self.batch_size = batch_size
        self._model = None
        self._device = None

    def _init_model(self):
        if self._model is not None:
            return
        if self._repo_dir and self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)
        from net1d import Net1D
        import torch

        checkpoint = Path(self._checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ECGFounder checkpoint not found: {str(checkpoint)!r}. Set "
                "ECGFOUNDER_REPO (clone of github.com/PKUDigitalHealth/ECGFounder) "
                "and ECGFOUNDER_CKPT (12_lead_ECGFounder.pth), or pass "
                "--feedback-kwargs '{\"repo_dir\": ..., \"checkpoint_path\": ...}'.")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            verbose=False, use_bn=False, use_do=False,
            n_classes=150, return_features=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        first = state.get("state_dict", {}).get("first_conv.conv.weight")
        if first is None or first.ndim != 3 or first.shape[1] != 12:
            channels = None if first is None else first.shape[1]
            raise ValueError(f"Expected 12-lead ECGFounder checkpoint, got {channels}")
        model.load_state_dict(state["state_dict"], strict=False)
        model.requires_grad_(False)
        self._model = model.to(self._device).eval()

    @classmethod
    def _preprocess(cls, segment):
        from scipy.signal import butter, filtfilt

        segment = np.asarray(segment).squeeze()
        if segment.shape == (cls.N_SAMPLES, 2):
            leads = segment.T
        elif segment.shape == (2, cls.N_SAMPLES):
            leads = segment
        else:
            raise ValueError(
                "Legacy ECGFounder requires exact shape (5000, 2) or (2, 5000)"
            )
        if not np.isfinite(leads).all():
            raise ValueError("ECG contains NaN or infinite values")

        padded = np.zeros((cls.N_MODEL_LEADS, cls.N_SAMPLES), dtype=np.float32)
        b, a = butter(2, 0.5, btype="highpass", fs=cls.SR)
        for index in range(2):
            lead = filtfilt(b, a, np.asarray(leads[index], dtype=np.float32))
            std = float(np.std(lead))
            if not np.isfinite(std) or std < 1e-8:
                raise ValueError(f"ECG lead {index + 1} is constant or invalid")
            padded[index] = (lead - np.mean(lead)) / std
        return padded

    def _embed_with_audit(self, directory):
        self._init_model()
        import torch

        files = sorted(Path(directory).glob("*.npy"))
        records, errors = [], Counter()
        for path in files:
            try:
                records.append(self._preprocess(np.load(path, allow_pickle=False)))
            except Exception as exc:
                errors[f"{type(exc).__name__}: {exc}"] += 1
        if not records:
            embeddings = np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        else:
            outputs = []
            for start in range(0, len(records), self.batch_size):
                x = torch.from_numpy(np.stack(records[start:start + self.batch_size])).to(
                    self._device)
                with torch.inference_mode():
                    _, features = self._model(x)
                outputs.append(features.detach().cpu().numpy().astype(np.float32))
            embeddings = np.concatenate(outputs)
        audit = {
            "total_files": len(files),
            "embedded_files": len(records),
            "skipped_files": len(files) - len(records),
            "errors": dict(errors),
        }
        return embeddings, audit

    def _embed(self, directory):
        return self._embed_with_audit(directory)[0]

    def analyze(self, reference_dir, generated_dir, *, include_report=True):
        real, real_audit = self._embed_with_audit(reference_dir)
        generated, generated_audit = self._embed_with_audit(generated_dir)
        metadata = {
            "reference_input": real_audit,
            "generated_input": generated_audit,
            "sampling_rate_hz": self.SR,
            "duration_s": 10.0,
            "observed_leads": ["I", "II"],
            "zero_padded_model_leads": 10,
            "checkpoint_leads": 12,
        }
        if len(real) < 3 or len(generated) < 3:
            return SearchFeedback(
                score=0.0,
                summary="Insufficient valid ECG records (< 3 each).",
                report=(f"Valid reference: {len(real)}/{real_audit['total_files']}; "
                        f"generated: {len(generated)}/{generated_audit['total_files']}. "
                        "Expected two leads, 500 Hz, 10 s."),
                metadata=metadata,
            )
        stats = compare(real, generated, scale=self.scale)
        if not include_report:
            return SearchFeedback(score=stats["score"],
                                  summary=f"Score: {stats['score']:.4f}",
                                  metadata={**stats, **metadata})
        feedback = build_feedback(
            stats,
            title="ECG Representation-Space Discrepancy Report (ECGFounder)",
            encoder_name="ECGFounder 12-lead (Lead I/II + 10 zero leads)",
            n_real=len(real), n_gen=len(generated),
            embedding_dim=self.EMBEDDING_DIM,
        )
        feedback.metadata.update(metadata)
        return feedback


register_feedback("ecg", "ecg_founder_12lead", ECGFounder12LeadRepresentation)
# Island name recorded in the released LQT/WPW core-run traces.
register_feedback("ecg", "ecg_founder", ECGFounder12LeadRepresentation)
