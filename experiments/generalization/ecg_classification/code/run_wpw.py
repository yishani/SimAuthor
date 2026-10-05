#!/usr/bin/env python3
"""Evaluate the final WPW simulator used in the paper's downstream study.

The selected simulator is representation-search seed 3, node 72. The study
adds 300 generated examples and reports five-seed logistic-regression results
using frozen ECGFounder representations.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]  # repository root
ECG_ROOT = Path(os.environ.get("SIMAUTHOR_ECG_ROOT", ROOT / "data" / "ECG"))
OUT = Path(os.environ.get(
    "SIMAUTHOR_ECG_WORKDIR",
    Path(__file__).resolve().parents[1] / "work",
))
CACHE = OUT / "cache"
DEFAULT_SIMULATOR = ROOT / "experiments/main/WPW/seed3/candidates/script_072.py"
N_EXTRA = 300
GEN_SEED = 42
TRAIN_SEEDS = [0, 1, 2, 3, 4]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

from generation import generate_samples
from cohorts import embed_files, load_cohort_files
from simauthor.search_feedback.representation.ecg.ecg_foundation_legacy import (
    ECGFounder12LeadRepresentation,
)


def metrics(y, predictions, scores):
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        recall_score,
        roc_auc_score,
    )
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(y, scores),
        "auprc": average_precision_score(y, scores),
        "sensitivity": recall_score(y, predictions),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def run_arm(x_train, y_train, x_id, y_id, x_ood, y_ood, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_id = scaler.transform(x_id)
    x_ood = scaler.transform(x_ood)
    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())
    classifier = LogisticRegression(
        solver="liblinear", max_iter=2000, random_state=seed,
        class_weight={0: 1.0, 1: n_negative / n_positive},
    )
    classifier.fit(x_train, y_train)
    id_scores = classifier.predict_proba(x_id)[:, 1]
    ood_scores = classifier.predict_proba(x_ood)[:, 1]
    return (
        metrics(y_id, (id_scores >= 0.5).astype(int), id_scores),
        metrics(y_ood, (ood_scores >= 0.5).astype(int), ood_scores),
    )


def generate_and_embed(encoder, simulator, tag):
    cache_path = CACHE / f"syn_{tag}_WPW.npz"
    if cache_path.exists():
        return np.load(cache_path)["embs"]

    files = generate_samples(simulator, N_EXTRA, seed=GEN_SEED, cwd=ROOT)
    temp_dir = tempfile.mkdtemp(prefix=f"ecgf_wpw_{tag}_")
    try:
        for idx, path in enumerate(files):
            os.symlink(
                os.path.abspath(path),
                os.path.join(temp_dir, f"{idx:05d}_{os.path.basename(path)}"),
            )
        embeddings = np.asarray(encoder._embed(temp_dir), dtype=np.float32)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert len(files) == N_EXTRA == len(embeddings), (
        f"Expected {N_EXTRA} samples, got files={len(files)}, "
        f"embeddings={len(embeddings)}"
    )
    np.savez_compressed(cache_path, embs=embeddings)
    with open(OUT / f"synthetic_manifest_{tag}_WPW.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "source", "file"])
        for path in files:
            writer.writerow(["WPW", str(simulator), os.path.basename(path)])
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--tag", default="rep_seed3_node72")
    parser.add_argument(
        "--representation-search-score", type=float,
        default=0.6525089176820943,
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    simulator = args.simulator.resolve()
    if not simulator.is_file():
        raise FileNotFoundError(simulator)

    start = time.time()
    encoder = ECGFounder12LeadRepresentation(
        repo_dir=str(ECG_ROOT / "ecgfounder_repo"),
        checkpoint_path=str(
            ECG_ROOT / "ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth"
        ),
    )
    encoder_name = "ECGFounder 12-lead (two observed leads, ten zero-padded)"

    cohorts = load_cohort_files(ECG_ROOT, "WPW")
    x_norm_train = embed_files(encoder, cohorts["norm_train"], CACHE, "norm_tr")
    x_norm_id = embed_files(encoder, cohorts["norm_id"], CACHE, "norm_id")
    x_normal_ood = embed_files(encoder, cohorts["normal_ood"], CACHE, "chap_neg")
    x_rare_train = embed_files(encoder, cohorts["rare_train"], CACHE, "rare_tr_WPW")
    x_rare_id = embed_files(encoder, cohorts["rare_id"], CACHE, "rare_id_WPW")
    x_rare_ood = embed_files(encoder, cohorts["rare_ood"], CACHE, "chap_pos_WPW")
    x_synthetic = generate_and_embed(encoder, simulator, args.tag)

    x_train = np.vstack([x_norm_train, x_rare_train, x_synthetic])
    y_train = np.array(
        [0] * len(x_norm_train) + [1] * (len(x_rare_train) + len(x_synthetic)),
        dtype=int,
    )
    x_id = np.vstack([x_norm_id, x_rare_id])
    y_id = np.array([0] * len(x_norm_id) + [1] * len(x_rare_id), dtype=int)
    x_ood = np.vstack([x_normal_ood, x_rare_ood])
    y_ood = np.array([0] * len(x_normal_ood) + [1] * len(x_rare_ood), dtype=int)

    rows = []
    for seed in TRAIN_SEEDS:
        id_metrics, ood_metrics = run_arm(
            x_train, y_train, x_id, y_id, x_ood, y_ood, seed
        )
        row = {
            "condition": "WPW",
            "arm": args.tag,
            "seed": seed,
            "id_auroc": id_metrics["auroc"],
            "id_auprc": id_metrics["auprc"],
            "ood_auroc": ood_metrics["auroc"],
            "ood_auprc": ood_metrics["auprc"],
            "id_sens": id_metrics["sensitivity"],
            "ood_sens": ood_metrics["sensitivity"],
            "ood_spec": ood_metrics["specificity"],
        }
        rows.append(row)
        print(
            f"seed={seed}: ID AUPRC/AUROC={row['id_auprc']:.6f}/"
            f"{row['id_auroc']:.6f}; OOD AUPRC/AUROC="
            f"{row['ood_auprc']:.6f}/{row['ood_auroc']:.6f}",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"wpw_{args.tag}_per_seed.csv", index=False)
    summary = {
        "condition": "WPW",
        "arm": args.tag,
        "simulator": str(simulator),
        "representation_search_score": args.representation_search_score,
        "n_synthetic": N_EXTRA,
        "generation_seed": GEN_SEED,
        "classifier_seeds": TRAIN_SEEDS,
        "encoder": encoder_name,
        "id_auprc_mean": float(frame.id_auprc.mean()),
        "id_auroc_mean": float(frame.id_auroc.mean()),
        "ood_auprc_mean": float(frame.ood_auprc.mean()),
        "ood_auroc_mean": float(frame.ood_auroc.mean()),
        "runtime_s": time.time() - start,
    }
    (OUT / f"wpw_{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
