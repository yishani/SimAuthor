"""Construct the patient-disjoint PTB-XL and Chapman ECG cohorts."""

import ast
import glob
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_DIR = {"LQT": "LNGQT", "WPW": "WPW"}
CONDITION_CODE = {"LQT": "LNGQT", "WPW": "WPW"}


def _record_id(path):
    return int(Path(path).stem.split("_")[1])


def load_cohort_files(ecg_root, condition):
    """Return the exact train, ID, and OOD file lists used in the study."""
    ecg_root = Path(ecg_root)
    database = pd.read_csv(ecg_root / "data/ptbxl_database.csv", index_col="ecg_id")
    database["scp"] = database.scp_codes.apply(ast.literal_eval)

    positive_ids = set(database[
        database.scp.apply(
            lambda codes: CONDITION_CODE[condition] in codes
            and codes[CONDITION_CODE[condition]] > 0
        )
    ].index)
    all_rare_patients = set()
    for code in CONDITION_CODE.values():
        ids = database[
            database.scp.apply(lambda codes: code in codes and codes[code] > 0)
        ].index
        all_rare_patients.update(database.loc[list(ids)].patient_id.dropna())

    def patient(path):
        return database.loc[_record_id(path)].patient_id

    norm_train = [
        path for path in sorted(glob.glob(str(ecg_root / "NORM/real_data/*.npy")))
        if patient(path) not in all_rare_patients
    ]
    norm_id = [
        path for path in sorted(glob.glob(str(ecg_root / "NORM/ptb_test/*.npy")))
        if patient(path) not in all_rare_patients
    ]
    directory = CONDITION_DIR[condition]
    rare_train = [
        path for path in sorted(glob.glob(str(ecg_root / f"{directory}/real_data/*.npy")))
        if _record_id(path) in positive_ids
    ]
    rare_id = [
        path for path in sorted(glob.glob(str(ecg_root / f"{directory}/ptb_test/*.npy")))
        if _record_id(path) in positive_ids
    ]

    strict_sinus = set()
    for header in glob.glob(str(ecg_root / "data/chapman/**/*.hea"), recursive=True):
        try:
            with open(header) as fh:
                for line in fh:
                    if line.startswith("#Dx:"):
                        if line.replace("#Dx:", "").strip().split(",") == ["426783006"]:
                            strict_sinus.add(Path(header).stem)
                        break
        except OSError:
            continue
    normal_ood = [
        str(ecg_root / f"NORM/chapman_test/chapman_{record}.npy")
        for record in sorted(strict_sinus)
        if (ecg_root / f"NORM/chapman_test/chapman_{record}.npy").exists()
    ]
    rare_ood = sorted(glob.glob(str(ecg_root / f"{directory}/chapman_test/*.npy")))
    return {
        "norm_train": norm_train,
        "norm_id": norm_id,
        "rare_train": rare_train,
        "rare_id": rare_id,
        "normal_ood": normal_ood,
        "rare_ood": rare_ood,
    }


def embed_files(encoder, files, cache_dir, cache_key):
    """Embed files with ECGFounder and cache features plus source filenames."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists():
        return np.load(cache_path)["embs"]

    temporary_dir = Path(tempfile.mkdtemp(prefix="ecgfounder_"))
    try:
        for index, source in enumerate(files):
            os.symlink(
                os.path.abspath(source),
                temporary_dir / f"{index:05d}_{Path(source).name}",
            )
        embeddings = np.asarray(encoder._embed(str(temporary_dir)), dtype=np.float32)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    if len(embeddings) != len(files):
        raise RuntimeError(f"Embedding count mismatch: {len(files)} files, {len(embeddings)} vectors")
    np.savez_compressed(
        cache_path,
        embs=embeddings,
        files=np.asarray([Path(path).name for path in files]),
    )
    return embeddings
