"""
_search_tree.py — Rank-based Flat PUCT (ERA-style)
====================================================
Global flat candidate selection with rank scores, uniform prior, and
child-to-root visit backpropagation.

Algorithm
---------
1. Compute rank scores from raw evaluator scores (average-rank ties).
2. Select globally via Flat PUCT:
     puct_i = rank_score_i + c_puct * prior * sqrt(total_visits) / (1 + visits_i)
   where prior = 1 / num_nodes.
3. After a child is successfully evaluated, backpropagate one visit
   from the child to the root (ancestry chain only).

This is NOT standard recursive MCTS.  Every node is a complete evaluated
simulator and all nodes remain eligible for future refinement.
"""

import json
import math
import numpy as np
from datetime import datetime

SEARCH_ALGORITHM = "era_rank_flat_puct"
# Label written into tree.json when the Orchestrator runs with
# select_mode="greedy" (ablation: explore term removed, everything else kept).
SEARCH_ALGORITHM_GREEDY = "era_greedy_best"


# ── Rank scores ──────────────────────────────────────────────────────────

def compute_rank_scores(nodes: list[dict]) -> list[float]:
    """Return rank scores in [0, 1] in node-list order.

    Uses average ranks for ties.  One node → 0.5.
    """
    n = len(nodes)
    if n == 1:
        return [0.5]

    scores = np.array([nd["score"] for nd in nodes], dtype=float)
    # argsort gives positions of sorted elements; use average-rank for ties
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0  # average rank (0-indexed)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return [float(r / (n - 1)) for r in ranks]


# ── Flat PUCT selection ──────────────────────────────────────────────────

def puct_select(nodes: list[dict], c_puct: float) -> dict:
    """Select one node globally via rank-based Flat PUCT.

    Does NOT mutate tree state.
    """
    n = len(nodes)
    if n == 0:
        raise ValueError("Cannot select from empty node list")
    if n == 1:
        return nodes[0]

    rank_scores = compute_rank_scores(nodes)
    total_visits = sum(nd["visits"] for nd in nodes)
    prior = 1.0 / n

    best_node, best_val = None, -float("inf")
    for i, nd in enumerate(nodes):
        exploration = (
            c_puct * prior * math.sqrt(max(total_visits, 1))
            / (1 + nd["visits"])
        )
        puct_val = rank_scores[i] + exploration

        if puct_val > best_val:
            best_val = puct_val
            best_node = nd
        elif puct_val == best_val and best_node is not None:
            # Deterministic tie-break: higher raw score, then lower script_idx
            if nd["score"] > best_node["score"]:
                best_node = nd
            elif nd["score"] == best_node["score"] and \
                 nd["script_idx"] < best_node["script_idx"]:
                best_node = nd
    return best_node


def greedy_select(nodes: list[dict]) -> dict:
    """Select the highest-scoring node (ablation control for Flat PUCT).

    Identical to puct_select with the exploration term removed: the parent is
    always the current argmax of the raw evaluator score, so the trajectory is a
    hill-climb on one incumbent.  Deterministic tie-break: lowest script_idx
    (matches experiments/baselines/greedy_best.py).

    Does NOT mutate tree state.
    """
    if not nodes:
        raise ValueError("Cannot select from empty node list")

    best_node, best_score, best_idx = None, -float("inf"), float("inf")
    for nd in nodes:
        if nd["score"] > best_score or (
                nd["score"] == best_score and nd["script_idx"] < best_idx):
            best_node, best_score, best_idx = nd, nd["score"], nd["script_idx"]
    return best_node


# ── Visit backpropagation ────────────────────────────────────────────────

def backpropagate_visit(nodes: list[dict], child: dict) -> None:
    """Increment visits along the ancestry chain from *child* to root."""
    idx_map = {nd["script_idx"]: nd for nd in nodes}

    # Increment the child itself.
    child["visits"] += 1

    current = child
    while current["parent"] is not None:
        parent_idx = current["parent"]
        if parent_idx not in idx_map:
            raise ValueError(
                f"Parent node {parent_idx} not found for child "
                f"{child['script_idx']} — tree may be corrupted."
            )
        parent = idx_map[parent_idx]
        parent["visits"] += 1
        current = parent
        # Guard against cycles.
        if current is child:
            raise ValueError(
                f"Cycle detected: child {child['script_idx']} reached "
                f"itself during backpropagation."
            )


# ── Tree operations ──────────────────────────────────────────────────────

def add_child(nodes: list[dict], parent: dict, /, *,
              script_idx: int, score: float, report: str,
              ) -> dict:
    """Create a child node, append it, and backpropagate one visit."""
    if any(nd["script_idx"] == script_idx for nd in nodes):
        raise ValueError(f"Duplicate script_idx: {script_idx}")

    child = {
        "script_idx": script_idx,
        "score": score,
        "visits": 0,
        "parent": parent["script_idx"],
        "report": report,
    }
    nodes.append(child)
    backpropagate_visit(nodes, child)
    return child


def best_node(nodes: list[dict]) -> dict:
    """Return the node with the highest raw evaluator score."""
    return max(nodes, key=lambda nd: nd["score"])


def make_root(score: float, report: str) -> dict:
    """Create the root node with visits=0."""
    return {
        "script_idx": 0,
        "score": score,
        "visits": 0,
        "parent": None,
        "report": report,
    }


# ── Persistence ──────────────────────────────────────────────────────────

def load_tree(path: str) -> list[dict]:
    if not path or not __import__("os").path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    if not nodes:
        raise ValueError(
            f"Tree file {path} contains no nodes.  "
            f"Delete it to force a fresh cold start."
        )
    return nodes


def save_tree(path: str, nodes: list[dict], metadata: dict,
              search_algorithm: str = SEARCH_ALGORITHM) -> None:
    with open(path, "w") as f:
        json.dump(dict(
            metadata,
            nodes=nodes,
            search_algorithm=search_algorithm,
            updated=datetime.now().isoformat(),
        ), f, indent=2)
