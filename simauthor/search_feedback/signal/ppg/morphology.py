"""
signal/ppg/morphology.py — PPG Signal-Space Agent (Morphology)
========================================
Compares reference and generated PPG sets using direct signal-domain
statistics: pulse intervals, amplitudes, waveform shape.

Comparison space:  signal
Modality:           ppg
Condition:          agnostic — reusable across AF and normal PPG

Input:  .npy files, 125 Hz, any duration (first 30 s used per segment).

Score:  mean feature-wise similarity (exp-based), aggregated across the
        feature set.
"""

import numpy as np
from pathlib import Path

from ...base import SearchFeedbackAgent, SearchFeedback
from ...report import DistributionDiscrepancy, render_report
from ... import register_feedback


class PPGSignal(SearchFeedbackAgent):
    """Signal-domain PPG evaluator — pulse intervals, amplitudes, waveform."""

    agent_id = "morphology"
    comparison_space = "signal"
    modality = "ppg"

    SR = 125
    SEGMENT_SAMPLES = 3750  # 30 s at 125 Hz

    _FEATURE_SPEC = [
        ("Heart rate (bpm)", "bpm"),
        ("Pulse interval mean", "ms"),
        ("Pulse interval std", "ms"),
        ("RMSSD", "ms"),
        ("Systolic peak amplitude mean", "a.u."),
        ("Systolic peak amplitude std", "a.u."),
        ("Pulse amplitude range", "a.u."),
        ("Rise time mean", "ms"),
        ("Pulse width mean", "ms"),
        ("Pulse count", ""),
    ]

    def __init__(self, name: str = "PPG Signal",
                 sr: int = 125, robust_height: bool = False):
        super().__init__(name=name)
        self.sr = sr
        # Robust height threshold: some real PPG records (e.g. the external AF
        # eval_ref) carry a tall artefact/beat that dominates the segment, so
        # ``height = 0.3 * max(filtered)`` rejects nearly all genuine pulses.
        # When set, the threshold is 0.3 x the 75th-percentile candidate beat
        # height instead (applied to reference and generated identically).
        # Default False keeps the legacy rule, so stored search/median scores
        # (computed with the legacy detector) remain directly comparable.
        self.robust_height = robust_height

    # ── Feature extraction ────────────────────────────────────────────────

    def _extract_features(self, directory: str) -> np.ndarray:
        features = []
        files = sorted(Path(directory).glob("*.npy"))
        for f in files:
            try:
                seg = np.load(f).astype(np.float64).flatten()
                if len(seg) < self.SEGMENT_SAMPLES or np.std(seg) < 1e-6:
                    continue
                seg = seg[:self.SEGMENT_SAMPLES]
                feats = self._segment_features(seg)
                if feats is not None:
                    features.append(feats)
            except Exception:
                pass
        return np.array(features) if features else np.empty((0, 10))

    def _segment_features(self, seg: np.ndarray):
        """Extract features from one PPG segment."""
        peaks = self._detect_peaks(seg)
        if len(peaks) < 3:
            return None

        # Pulse intervals
        rr = np.diff(peaks) / self.sr * 1000  # ms
        rr = rr[(rr > 300) & (rr < 2000)]  # physiological range
        if len(rr) < 2:
            return None

        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        hr = 60000 / rr_mean if rr_mean > 0 else 0.0

        # Amplitudes
        amps = seg[peaks[:len(rr) + 1]]
        amp_mean = float(np.mean(amps))
        amp_std = float(np.std(amps))
        amp_range = float(np.max(amps) - np.min(amps))

        # Rise time: 25%→75% rise per pulse
        rise_times = []
        pulse_widths = []
        for i in range(len(peaks) - 1):
            # Find pulse onset (foot before peak)
            search_start = max(0, peaks[i] - int(0.3 * self.sr))
            window = seg[search_start:peaks[i]]
            if len(window) < 2:
                continue
            foot = search_start + int(np.argmin(window))

            # Rise time: 25%→75% of peak amplitude
            lo = seg[foot] + 0.25 * (seg[peaks[i]] - seg[foot])
            hi = seg[foot] + 0.75 * (seg[peaks[i]] - seg[foot])
            rise_start = foot + int(np.argmax(
                seg[foot:peaks[i]] >= lo
            )) if np.any(seg[foot:peaks[i]] >= lo) else foot
            rise_end = foot + int(np.argmax(
                seg[foot:peaks[i]] >= hi
            )) if np.any(seg[foot:peaks[i]] >= hi) else peaks[i]
            rise_times.append((rise_end - rise_start) / self.sr * 1000)

            # Pulse width at 50% amplitude
            half_amp = seg[foot] + 0.5 * (seg[peaks[i]] - seg[foot])
            above = seg[foot:min(peaks[i] + int(0.3 * self.sr), len(seg))] >= half_amp
            if np.any(above):
                crossings = np.where(np.diff(above.astype(int)))[0]
                if len(crossings) >= 2:
                    pulse_widths.append(
                        (crossings[-1] - crossings[0]) / self.sr * 1000
                    )

        rise_mean = float(np.mean(rise_times)) if rise_times else 0.0
        pw_mean = float(np.mean(pulse_widths)) if pulse_widths else 0.0

        return [
            hr, rr_mean, rr_std, rmssd,
            amp_mean, amp_std, amp_range,
            rise_mean, pw_mean,
            float(len(peaks)),
        ]

    def _detect_peaks(self, seg: np.ndarray) -> np.ndarray:
        """Detect systolic peaks using scipy."""
        import scipy.signal
        # Bandpass 0.5–10 Hz
        nyq = 0.5 * self.sr
        b, a = scipy.signal.butter(2, [0.5/nyq, 10/nyq], btype="band")
        filtered = scipy.signal.filtfilt(b, a, seg)
        # Peak detection
        min_dist = int(0.3 * self.sr)  # min 300 ms between beats
        if self.robust_height:
            # outlier-robust representative height: 75th pctile of the heights
            # of candidate (distance-only) peaks, so a single tall artefact
            # cannot veto the genuine pulses below it.
            cand, _ = scipy.signal.find_peaks(filtered, distance=min_dist)
            if len(cand) < 3:
                return cand
            height = 0.3 * float(np.percentile(filtered[cand], 75))
        else:
            height = 0.3 * np.max(filtered)
        peaks, _ = scipy.signal.find_peaks(
            filtered, distance=min_dist, height=height,
        )
        return peaks

    # ── Score ──────────────────────────────────────────────────────────────

    def _compute_score(self, real_f, gen_f) -> tuple[float, np.ndarray]:
        r_mean = np.mean(real_f, axis=0)
        g_mean = np.mean(gen_f, axis=0)
        pooled = np.sqrt(
            (np.std(real_f, axis=0) ** 2 + np.std(gen_f, axis=0) ** 2) / 2
            + 1e-8
        )
        sims = np.exp(-2.0 * np.abs(r_mean - g_mean) / (pooled + 1e-8))
        return float(np.mean(sims)), sims

    # ── analyze() ──────────────────────────────────────────────────────────

    def analyze(self, reference_dir: str, generated_dir: str, *,
                include_report: bool = True) -> SearchFeedback:
        real_f = self._extract_features(reference_dir)
        gen_f = self._extract_features(generated_dir)

        if real_f.shape[0] < 5 or gen_f.shape[0] < 5:
            return SearchFeedback(
                score=0.0, summary="Insufficient segments (< 5 each).",
                report=(
                    f"Real segments: {real_f.shape[0]}, "
                    f"gen segments: {gen_f.shape[0]}.  Need at least 5 each."
                ),
            )

        score, sims = self._compute_score(real_f, gen_f)

        if not include_report:
            return SearchFeedback(score=score, summary=f"Score: {score:.4f}")

        r_mean = np.mean(real_f, axis=0)
        g_mean = np.mean(gen_f, axis=0)
        discrepancies = [
            DistributionDiscrepancy(
                name=name,
                value=sims[i], benchmark=1.0,
                interpretation=(
                    f"Gen mean: {g_mean[i]:.3f}, ref mean: {r_mean[i]:.3f}"
                    f"{' ' + unit if unit else ''}."
                ),
            )
            for i, (name, unit) in enumerate(self._FEATURE_SPEC)
        ]

        worst = int(np.argmin(sims))
        observations = [
            f"Valid segments: {real_f.shape[0]} reference, "
            f"{gen_f.shape[0]} generated.",
        ]
        for i, sim in enumerate(sims):
            if sim < 0.3:
                name, _ = self._FEATURE_SPEC[i]
                observations.append(
                    f"{name}: low similarity ({sim:.3f}).  "
                    f"Gen {g_mean[i]:.3f} vs ref {r_mean[i]:.3f}."
                )

        report = render_report(
            title="PPG Signal-Space Discrepancy Report",
            header=[
                f"Segments: {real_f.shape[0]} reference  |  "
                f"{gen_f.shape[0]} generated  |  "
                f"SR: {self.sr} Hz",
                f"Overall score: {score:.4f}",
            ],
            discrepancies=discrepancies,
            observations=observations,
        )

        return SearchFeedback(
            score=score,
            summary=(
                f"PPG signal similarity: {score:.3f}.  "
                f"Worst: {self._FEATURE_SPEC[worst][0]} "
                f"(sim={sims[worst]:.3f})."
            ),
            report=report,
            metadata={
                "per_feature_similarity": [float(s) for s in sims],
                "n_reference": int(real_f.shape[0]),
                "n_generated": int(gen_f.shape[0]),
            },
        )


register_feedback("ppg", "morphology", PPGSignal)
