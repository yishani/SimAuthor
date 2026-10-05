"""
presets.py — Output contracts and the six paper conditions
============================================================
Per-modality defaults reproduce the settings used for every run in the paper
(100 samples per evaluation, 100 authoring attempts, c_puct = 0.25,
delta = 0.02, visual feedback on).
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModalityPreset:
    duration: float          # seconds per generated record
    synth_sr: int            # internal synthesis rate suggested to the model
    output_sr: int           # sampling rate of saved records
    feedback_agent: str      # default evaluator key
    file_ext: str            # artifact extension the evaluator reads


MODALITIES = {
    "audio": ModalityPreset(10.0, 44100, 16000, "mfcc", ".wav"),
    "ppg": ModalityPreset(30.0, 500, 125, "morphology", ".npy"),
    "ecg": ModalityPreset(10.0, 500, 500, "ecg_founder_12lead", ".npy"),
}

#: Paper search defaults.
NUM_SAMPLES = 100
ITERATIONS = 100
C_PUCT = 0.25
MIN_DELTA_FOR_MECHANISM = 0.02


@dataclass(frozen=True)
class Condition:
    key: str
    name: str                # condition string given to the model
    modality: str
    feedback_agent: str
    dataset: str             # search-set source (obtain under its own licence)
    median_seed: int         # seed whose best program is released in simauthor/simulators/
    best_node: int
    root_score: float
    best_score: float


CONDITIONS = {c.key: c for c in [
    Condition("VSD", "Ventricular Septal Defect", "audio", "mfcc",
              "ZCHSound", 1, 89, 0.258, 0.587),
    Condition("AS", "Aortic Stenosis", "audio", "mfcc",
              "BMD-HS", 3, 31, 0.139, 0.557),
    Condition("COPD", "COPD", "audio", "mfcc",
              "ICBHI 2017", 1, 96, 0.165, 0.672),
    Condition("AF", "Atrial Fibrillation", "ppg", "morphology",
              "MIMIC PERform AF", 2, 26, 0.216, 0.738),
    Condition("LQT", "Long QT Syndrome", "ecg", "ecg_founder",
              "PTB-XL", 2, 53, 0.286, 0.568),
    Condition("WPW", "WPW Syndrome", "ecg", "ecg_founder",
              "PTB-XL", 1, 11, 0.364, 0.581),
]}


def get_condition(key: str) -> Condition:
    try:
        return CONDITIONS[key.upper()]
    except KeyError:
        raise ValueError(f"Unknown condition {key!r}. "
                         f"Available: {', '.join(CONDITIONS)}") from None


def describe() -> str:
    rows = [f"{'key':5} {'modality':8} {'evaluator':12} {'root':>5} {'best':>5}  name"]
    for c in CONDITIONS.values():
        rows.append(f"{c.key:5} {c.modality:8} {c.feedback_agent:12} "
                    f"{c.root_score:5.3f} {c.best_score:5.3f}  {c.name}")
    return "\n".join(rows)


__all__ = ["ModalityPreset", "MODALITIES", "Condition", "CONDITIONS",
           "get_condition", "describe", "asdict"]
