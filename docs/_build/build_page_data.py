#!/usr/bin/env python3
"""Build the data behind the project page (docs/data, docs/assets/visual).

Maintainer script.  Most inputs ship in ``experiments/``; the refiner prompts and
the real-vs-generated figures come from the raw run directories, which are
not released because they name patient recordings.  Everything written to
``docs/`` is scrubbed:
  * absolute paths and user names are removed,
  * every reference-recording filename that appears in any visual-feedback
    metadata file is collected, and the build fails if one survives,
  * visual-feedback metadata files are never copied.

Usage:
  python docs/_build/build_page_data.py --raw /path/to/SimAuthor/artifacts
"""
import argparse
import getpass
import ast
import csv
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "experiments"
DOCS = REPO / "docs"

# key, released seed, raw run dir (relative to --raw), display name, modality, evaluator label
TASKS = [
    ("VSD", 1, "vsd_audio/runs/signal/seed1", "Ventricular septal defect", "audio", "MFCC/ZCR overlap"),
    ("AS", 3, "aortic_stenosis_audio/runs/signal/seed3", "Aortic stenosis", "audio", "MFCC/ZCR overlap"),
    ("COPD", 1, "copd_audio/runs/signal/seed1", "COPD", "audio", "MFCC/ZCR overlap"),
    ("AF", 2, "af_ppg/runs/signal/seed2", "Atrial fibrillation", "ppg", "Pulse morphology"),
    ("LQT", 2, "lngqt_ecg/runs/rep/seed2", "Long-QT syndrome", "ecg", "ECGFounder energy distance"),
    ("WPW", 1, "wpw_ecg/runs/rep/seed1", "Wolff-Parkinson-White", "ecg", "ECGFounder energy distance"),
]
# Runs shown in the trace explorer and the generations section: one run per
# task: the seed whose best program appears latest in the search.
EXPLORER = [
    ("VSD", 1, "vsd_audio/runs/signal/seed1", "Ventricular septal defect", "audio", "MFCC/ZCR overlap"),
    ("AS", 2, "aortic_stenosis_audio/runs/signal/seed2", "Aortic stenosis", "audio", "MFCC/ZCR overlap"),
    ("COPD", 1, "copd_audio/runs/signal/seed1", "COPD", "audio", "MFCC/ZCR overlap"),
    ("AF", 1, "af_ppg/runs/signal/seed1", "Atrial fibrillation", "ppg", "Pulse morphology"),
    ("LQT", 3, "lngqt_ecg/runs/rep/seed3", "Long-QT syndrome", "ecg", "ECGFounder energy distance"),
    ("WPW", 3, "wpw_ecg/runs/rep/seed3", "Wolff-Parkinson-White", "ecg", "ECGFounder energy distance"),
]
MODALITY_LABEL = {"audio": "Auscultation audio", "ppg": "PPG", "ecg": "Two-lead ECG"}

# ── scrubbing ─────────────────────────────────────────────────────────────
_PATH_RE = re.compile(r"(?:/gpfs/|/projects/|/home/|/scratch)[^\s\"'`)]*")
_PLACEHOLDER_RE = re.compile(r"\[PROJECT_ROOT\][^\s\"'`)]*")


def scrub(s):
    if not isinstance(s, str):
        return s
    s = _PLACEHOLDER_RE.sub("generated/", s)
    s = _PATH_RE.sub("generated/", s)
    return s.replace(getpass.getuser(), "user").replace("artifacts_3.1", "artifacts")


# ── parsers (adapted from the internal trace viewer) ─────────────────────
_ANCHOR_RE = re.compile(
    r"^## (?:Task Identity|Scientific Reference|Immutable Output Contract|"
    r"Current Simulator\b.*|Numerical Discrepancy Report|Mechanism Library|"
    r"Refinement Policy|Important: Role of the Numerical Discrepancy Report|"
    r"Output Format|VISUAL EVIDENCE)\s*$", re.M)


def split_prompt(prompt):
    marks = [m.start() for m in _ANCHOR_RE.finditer(prompt or "")]
    out = {}
    for i, start in enumerate(marks):
        end = marks[i + 1] if i + 1 < len(marks) else len(prompt)
        chunk = prompt[start:end]
        head, _, body = chunk.partition("\n")
        out[head[3:].strip()] = body.strip("\n")
    return out


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_report(block):
    if not block:
        return None
    title, meta, score, header, rows, obs = "", "", None, None, [], []
    for raw in block.splitlines():
        s = raw.strip()
        if not s or s.startswith("### "):
            continue
        if not title and s.startswith("## "):
            title = s[3:]
            continue
        m = re.match(r"^Overall (?:overlap )?score\s*:\s*([0-9.]+)", s)
        if m:
            score = float(m.group(1))
            continue
        if s.startswith("|"):
            c = _cells(s)
            if all(x and set(x) <= set("-—–: ") for x in c):
                continue
            if header is None:
                header = c
            else:
                rows.append({"c": c})
            continue
        if s.startswith("*") and s.endswith("*") and rows:
            rows[-1]["n"] = s.strip("*").strip()
            continue
        if not meta and ("reference" in s.lower() or "|" in s):
            meta = s
            continue
        obs.append(s)
    return {"title": title, "meta": meta, "score": score,
            "header": header or [], "rows": rows, "obs": obs}


def parse_mech_block(block):
    return [m.group(1) for m in re.finditer(r"^- `([^`]+)`:", block or "", re.M)]


def revision_notes(code):
    """The refiner documents its change in the module docstring; pull that part."""
    m = re.match(r'\s*(?:#.*\n)*\s*("""|\'\'\')(.*?)\1', code or "", re.S)
    if not m:
        return ""
    doc = m.group(2)
    k = re.search(r"^\s*(Refinement|Revision|Changes?|Hypothesis|Update|Modification)[^\n]*:?\s*$",
                  doc, re.M | re.I)
    text = doc[k.start():] if k else ""
    return text.strip()[:1800]


def load_trace(run_dir):
    recs = []
    for f in sorted(run_dir.glob("logs/trace_*.jsonl")):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    return recs


def lineage(nodes, idx):
    out = [idx]
    while nodes[out[-1]].get("parent") is not None and nodes[out[-1]]["parent"] >= 0:
        out.append(nodes[out[-1]]["parent"])
    return out[::-1]


# ── per-task run data ─────────────────────────────────────────────────────

def build_run(key, seed, raw_dir, name, modality, evaluator, real_names, img_out):
    core = TRACES / "main" / key / f"seed{seed}"
    tree = json.load(open(core / "tree.json"))
    raw_tree = json.load(open(raw_dir / "tree.json"))
    nodes = tree["nodes"]
    assert [round(n["score"], 6) for n in nodes] == [round(n["score"], 6) for n in raw_tree["nodes"]], key

    recs = load_trace(raw_dir)
    refines = [r for r in recs if r.get("step") == "refine"]
    iters = [r for r in recs if r.get("step") == "iteration"]
    # refine records are logged before the child exists; pair them with the
    # iteration record of the same round to recover the archived child id
    by_round = {}
    for r in iters:
        by_round.setdefault(r["iteration"], []).append(r)
    used, paired = set(), {}
    for r in refines:
        match = next((it for it in by_round.get(r["iteration"], []) if id(it) not in used), None)
        if match:
            used.add(id(match))
            paired[int(match["child_idx"])] = r
    n_attempts = len(refines)

    for vj in (raw_dir / "visual").glob("node_*.json"):
        d = json.load(open(vj))
        for k in ("real_samples", "real_files", "real"):
            v = d.get(k)
            if isinstance(v, str) and v.startswith("["):
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    v = [v]
            for x in (v if isinstance(v, list) else [v] if v else []):
                real_names.add(Path(str(x)).name)

    mechs = json.load(open(core / "mechanism_library.json"))
    mech_by_node = {}
    for m in mechs:
        mech_by_node.setdefault(int(m["source_node"]), []).append(
            {"name": m["name"], "mechanism": m.get("mechanism", ""),
             "snippet": m.get("code_snippet", ""), "delta": m.get("delta"),
             "count": m.get("observation_count", 1)})

    best = max(range(1, len(nodes)), key=lambda i: nodes[i]["score"])
    lin = lineage(nodes, best)
    top = sorted(range(len(nodes)), key=lambda i: -nodes[i]["score"])[:10]
    with_img = sorted(set(lin) | set(top))

    shared = {}
    out_nodes = []
    for i, n in enumerate(nodes):
        code = scrub((core / "candidates" / f"script_{i:03d}.py").read_text())
        ref = paired.get(i)
        sec = split_prompt(ref["prompt"]) if ref else {}
        if sec and not shared:
            shared = {
                "task": scrub(sec.get("Task Identity", "")),
                "blueprint": scrub(sec.get("Scientific Reference", "")),
                "contract": scrub(sec.get("Immutable Output Contract", "")),
                "policy": scrub("\n\n".join(sec.get(h, "") for h in (
                    "Refinement Policy", "Important: Role of the Numerical Discrepancy Report",
                    "Output Format"))),
                "visual": scrub(sec.get("VISUAL EVIDENCE", "")),
            }
        p = n.get("parent")
        out_nodes.append({
            "id": i,
            "parent": p if isinstance(p, int) and p >= 0 else None,
            "score": round(n["score"], 4),
            "visits": n.get("visits"),
            "attempt": (ref["iteration"] + 1) if ref else 0,
            "report": parse_report(scrub(n.get("report", ""))),
            "code": code,
            "notes": revision_notes(code) if i else "",
            "prompt_chars": len(ref["prompt"]) if ref else 0,
            "prompt_mech": parse_mech_block(sec.get("Mechanism Library")) if sec else [],
            "mech": mech_by_node.get(i, []),
            "img": i in with_img,
        })

    for i in with_img:
        src = raw_dir / "visual" / f"node_{i:03d}.png"
        if src.exists():
            from PIL import Image
            with Image.open(src) as im:
                im = im.convert("RGB")
                if im.width > 1100:
                    im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
                img_out.mkdir(parents=True, exist_ok=True)
                im.save(img_out / f"node_{i:03d}.jpg", "JPEG", quality=78, optimize=True)
        else:
            out_nodes[i]["img"] = False

    other = {}
    for s in (1, 2, 3):
        t = json.load(open(TRACES / "main" / key / f"seed{s}" / "tree.json"))["nodes"]
        other[f"seed{s}"] = {"parent": [n["parent"] if isinstance(n.get("parent"), int) else None for n in t],
                             "score": [round(n["score"], 4) for n in t]}

    return {
        "key": key, "name": name, "modality": modality,
        "modality_label": MODALITY_LABEL[modality], "evaluator": evaluator,
        "seed": seed, "attempts": n_attempts, "c_puct": 0.25,
        "best": best, "lineage": lin, "shared": shared,
        "nodes": out_nodes, "seeds": other,
    }


# ── curves for all methods ────────────────────────────────────────────────

def tree_scores(path):
    return [round(n["score"], 4) for n in json.load(open(path))["nodes"]]


def build_curves():
    out = {}
    for key, *_ , modality, _ev in TASKS:
        lk = key.lower()
        methods = {
            "SimAuthor": [tree_scores(p) for p in sorted((TRACES / "main" / key).glob("seed*/tree.json"))],
            "PUCT score search": [tree_scores(p) for p in sorted((TRACES / "baselines/puct_score_search" / key).glob("seed*/tree.json"))],
            "Text-Opt": [tree_scores(p) for p in sorted((TRACES / "baselines/text_opt" / key).glob("seed*/tree.json"))],
            "− Report": [tree_scores(p) for p in sorted((TRACES / "ablations/no_report" / key).glob("seed*/tree.json"))],
            "− Library": [tree_scores(p) for p in sorted((TRACES / "ablations/no_library" / key).glob("seed*/tree.json"))],
        }
        col = "rep_score" if modality == "ecg" else "signal_score"
        with open(TRACES / "baselines/sampling" / key / "results.csv") as f:
            rows = list(csv.DictReader(f))
        samp = [round(float(r[col]), 4) if r[col] not in ("", "nan") else 0.0 for r in rows[:100]]
        methods["Sampling"] = [samp]
        out[key] = methods
    return out


def check_paper_numbers(curves):
    """S_max medians must match Table 1 of the paper."""
    table1 = {"SimAuthor": dict(VSD=.587, AS=.557, COPD=.672, AF=.738, LQT=.568, WPW=.581),
              "PUCT score search": dict(VSD=.356, AS=.247, COPD=.455, AF=.560, LQT=.532, WPW=.550),
              "Text-Opt": dict(VSD=.416, AS=.412, COPD=.549, AF=.801, LQT=.554, WPW=.526),
              "Sampling": dict(VSD=.477, AS=.184, COPD=.391, AF=.492, LQT=.418, WPW=.456)}
    bad = []
    for m, row in table1.items():
        for k, v in row.items():
            runs = curves[k][m]
            got = statistics.median(max(r[1:] if m != "Sampling" else r) for r in runs)
            if abs(got - v) > 0.0015:
                bad.append(f"{m}/{k}: traces {got:.4f} vs paper {v}")
    for b in bad:
        print("  ! mismatch", b)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="raw run root (…/SimAuthor/artifacts)")
    args = ap.parse_args()
    data = DOCS / "data"
    (data / "runs").mkdir(parents=True, exist_ok=True)
    for old in (data / "runs").glob("*.json"):
        old.unlink()
    import shutil
    shutil.rmtree(DOCS / "assets" / "visual", ignore_errors=True)
    real_names = set()
    index = []
    for key, seed, rel, name, modality, ev in EXPLORER:
        run = build_run(key, seed, args.raw / rel, name, modality, ev, real_names,
                        DOCS / "assets" / "visual" / key)
        (data / "runs" / f"{key}.json").write_text(json.dumps(run, separators=(",", ":")))
        index.append({k: run[k] for k in ("key", "name", "modality", "modality_label", "evaluator",
                                           "seed", "best", "lineage", "attempts")}
                     | {"root": run["nodes"][0]["score"], "best_score": run["nodes"][run["best"]]["score"]})
        print(f"{key}: {len(run['nodes'])} nodes, lineage {run['lineage']}")
    (data / "index.json").write_text(json.dumps(index, indent=1))

    curves = build_curves()
    (data / "curves.json").write_text(json.dumps(curves, separators=(",", ":")))
    mismatches = check_paper_numbers(curves)


    # privacy gate
    real_names = {n for n in real_names if len(n) > 6}
    leaks = []
    for f in list(data.rglob("*.json")):
        text = f.read_text()
        for n in real_names:
            if n in text:
                leaks.append((f.name, n))
        for pat in ("/gpfs", "/projects/", "/home/", getpass.getuser(), "[PROJECT_ROOT]"):
            if pat in text:
                leaks.append((f.name, pat))
    if leaks:
        raise SystemExit(f"privacy check failed: {leaks[:10]}")
    print(f"privacy check passed ({len(real_names)} reference filenames screened)")
    if mismatches:
        raise SystemExit("paper-number check failed")
    print("paper Table 1 S_max medians reproduced from traces")


if __name__ == "__main__":
    main()
