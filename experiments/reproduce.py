#!/usr/bin/env python3
"""Recompute the paper's search results from the traces in this folder.

    python experiments/reproduce.py

Prints each table next to the value reported in the paper and exits with a
non-zero status if any value differs by more than the stated tolerance.

Definitions (all from the archived programs of each run, root excluded):
  S_max     best score after 100 attempts; median over the three seeds
  S_top10   mean of the ten best scores; median over the three seeds
  AUC_100   mean best-so-far score over attempts 1-100, starting from the root
            score and indexing programs by script_idx; median over three seeds
Sampling is one pool of independent proposals; its first 100 are used.
Table 2 and the revision audit are checked against the stored result files.
Table 4 (held-out recordings) needs the external datasets and is not covered.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ["VSD", "AS", "COPD", "AF", "LQT", "WPW"]

PAPER = {
    "Table 1 (S_max)": {
        "SimAuthor": [.587, .557, .672, .738, .568, .581],
        "PUCT score search": [.356, .247, .455, .560, .532, .550],
        "Text-Opt": [.416, .412, .549, .801, .554, .526],
        "Sampling": [.477, .184, .391, .492, .418, .456],
    },
    "Table 1 (S_top10)": {
        "SimAuthor": [.483, .443, .628, .701, .553, .527],
        "PUCT score search": [.327, .202, .444, .528, .530, .546],
        "Text-Opt": [.348, .250, .505, .770, .477, .457],
        "Sampling": [.424, .152, .259, .412, .401, .406],
    },
    "Table 3a (S_max)": {
        "SimAuthor": [.587, .557, .672, .738, .568, .581],
        "- Report": [.366, .152, .609, .589, .460, .492],
        "- Library": [.399, .383, .662, .719, .540, .573],
    },
    "Table 3b (AUC_100)": {
        "SimAuthor": [.395, .458, .544, .717, .539, .574],
        "- Report": [.356, .143, .585, .556, .439, .486],
        "- Library": [.348, .324, .529, .686, .514, .547],
    },
}
ROBUSTNESS = {  # Table 5: S_max of the three seeds, ascending
    ("VSD", "c_puct_0.10"): [.330, .362, .472], ("VSD", "c_puct_0.50"): [.488, .504, .535],
    ("VSD", "deepseek_v4_pro"): [.543, .551, .579],
    ("LQT", "c_puct_0.10"): [.513, .548, .554], ("LQT", "c_puct_0.50"): [.572, .596, .637],
    ("LQT", "deepseek_v4_pro"): [.575, .598, .614],
}
RUN_DIRS = {
    "SimAuthor": "main/{t}", "PUCT score search": "baselines/puct_score_search/{t}",
    "Text-Opt": "baselines/text_opt/{t}", "- Report": "ablations/no_report/{t}",
    "- Library": "ablations/no_library/{t}",
}


def seed_scores(pattern, with_root=False):
    runs = []
    for tree in sorted(HERE.glob(pattern + "/seed*/tree.json")):
        nodes = json.load(open(tree))["nodes"]
        runs.append(nodes if with_root else [n["score"] for n in nodes[1:]])
    return runs


def sampling_scores(task):
    with open(HERE / "baselines/sampling" / task / "results.csv") as f:
        rows = list(csv.DictReader(f))[:100]
    col = "rep_score" if task in ("LQT", "WPW") else "signal_score"
    return [[float(r[col]) for r in rows if r[col] not in ("", "nan")]]


def s_max(r): return max(r)
def s_top10(r): return statistics.mean(sorted(r)[-10:])


def auc(nodes, budget=100):
    scores = {int(n["script_idx"]): n["score"] for n in nodes}
    best, curve = scores[0], []
    for attempt in range(1, budget + 1):
        best = max(best, scores.get(attempt, best))
        curve.append(best)
    return statistics.mean(curve)


METRIC = {"S_max": (s_max, 0.0015), "S_top10": (s_top10, 0.0015), "AUC_100": (auc, 0.0015)}


def main():
    failures = 0
    for title, rows in PAPER.items():
        metric = title.split("(")[1].rstrip(")")
        fn, tol = METRIC[metric]
        print(f"\n{title}  (recomputed / paper)")
        print(f"{'':20}" + "".join(f"{t:>14}" for t in TASKS))
        for method, paper_vals in rows.items():
            cells = []
            for task, ref in zip(TASKS, paper_vals):
                runs = (sampling_scores(task) if method == "Sampling"
                        else seed_scores(RUN_DIRS[method].format(t=task), with_root=metric == "AUC_100"))
                got = statistics.median(fn(r) for r in runs)
                ok = abs(got - ref) <= tol
                failures += not ok
                cells.append(f"{got:.3f}/{ref:.3f}{' ' if ok else '!'}")
            print(f"{method:20}" + "".join(f"{c:>14}" for c in cells))

    print("\nTable 5 (S_max per seed, ascending)  recomputed / paper")
    for (task, setting), ref in ROBUSTNESS.items():
        got = sorted(s_max(r) for r in seed_scores(f"robustness/{setting}/{task}"))
        ok = all(abs(a - b) <= 0.0015 for a, b in zip(got, ref))
        failures += not ok
        print(f"  {task:4} {setting:16} {[round(g, 3) for g in got]} / {ref}{'' if ok else '  MISMATCH'}")

    # Generalization: stored endpoint results (computing them needs external
    # encoders and datasets, see experiments/generalization/).
    print("\nTable 2a (independent evaluator, R_sel)  stored / paper")
    rep = json.load(open(HERE / "generalization/new_evaluator/results.json"))["results"]
    paper_2a = {"root": [.317, .192, .152, .354], "one_shot": [.123, .106, .162, .434],
                "text_op": [.443, .258, .468, .631], "full": [.204, .306, .561, .665]}
    for method, ref in paper_2a.items():
        got = [rep[t][method]["r_sel"] for t in ("VSD", "AS", "COPD", "AF")]
        ok = all(abs(a - b) <= 0.0015 for a, b in zip(got, ref))
        failures += not ok
        print(f"  {method:9} {[round(g, 3) for g in got]} / {ref}{'' if ok else '  MISMATCH'}")
    print("\nTable 2b (ECG classification, + SimAuthor)  stored / paper")
    for task, file, ref in (("LQT", "lqt_rep_seed3_node88", (.404, .747)), ("WPW", "wpw_rep_seed3_node72", (.861, .804))):
        d = json.load(open(HERE / f"generalization/ecg_classification/results/{file}_summary.json"))
        got = (d["id_auprc_mean"], d["ood_auprc_mean"])
        ok = all(abs(a - b) <= 0.0015 for a, b in zip(got, ref))
        failures += not ok
        print(f"  {task}  AUPRC ID/OOD {got[0]:.3f}/{got[1]:.3f} / {ref[0]}/{ref[1]}{'' if ok else '  MISMATCH'}")

    print("\nRevision audit")
    with open(HERE / "audit/lineage_audit.csv") as f:
        rows = list(csv.DictReader(f))
    structural = [r for r in rows if r["is_param_tuning"] == "False"]
    share = 100 * sum(float(r["delta"]) for r in structural) / sum(float(r["delta"]) for r in rows)
    ok = (len(rows), len(structural), round(share, 1)) == (138, 111, 86.1)
    failures += not ok
    print(f"  structural revisions {len(structural)}/{len(rows)}, {share:.1f}% of the signed gain"
          f" / paper 111/138, 86.1%{'' if ok else '  MISMATCH'}")

    print(f"\n{'All values match the paper.' if not failures else f'{failures} value(s) differ (marked !).'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
