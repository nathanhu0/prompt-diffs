"""Loaders for the verbalization-scaling outputs: turn each arm's saved log
into a best-so-far trajectory on any of the three budget axes (wall-clock
seconds, number of verifications, or candidate index — FLOPs derive offline
from the stored texts if ever needed).

Output layout (run_readout.py / run_comparison.py):
  {SCR}/seed<N>/readout/filtered_schrodi/<task>/readout_<arm>{.json,_results.pt}
  {SCR}/seed<N>/largo_<arm>/filtered_schrodi/<task>/largo{.json,_results.pt}

Every trajectory is a list of (t_seconds, n_scored, best_so_far_select_score).
Select scores are the fixed 256-example TRAIN-subset hard NLL (the search's
own selection metric); val/test/behavior for the WINNER live in the .json.
"""
import json
from pathlib import Path

import torch

SCR = Path("/nlp/scr/nathu/latent_rewrite/verbalization_scaling")

BEAM_ARMS_X16 = ["beam_1x16", "beam_2x16", "beam_4x16", "beam_8x16"]
BEAM_ARMS_X8 = ["beam_1x8", "beam_2x8", "beam_4x8", "beam_8x8"]
BEAM_ARMS_LIGHT = ["beam_1x2", "beam_1x4"]     # small-budget single-beam arms
BON_ARM = "best_of_1536"
LARGO_ARMS = ["steps50", "steps125", "steps250", "steps500", "steps1000", "temp07"]


def _best_so_far(events):
    """events: iterable of (t, score) in chronological order →
    [(t, n_scored, best_so_far)]."""
    out, best = [], float("inf")
    for n, (t, s) in enumerate(events, start=1):
        if s is not None and s < best:
            best = s
        out.append((t, n, best))
    return out


def load_beam_arm(seed, arm, task="cat", scr=SCR):
    """Trajectory + winner record for one beam readout arm. Nodes are stored
    in creation (= scoring) order; node['t'] is elapsed seconds at scoring
    (root has no 't' and no score event — skip it)."""
    d = scr / f"seed{seed}" / "readout" / "filtered_schrodi" / task
    pt = d / f"readout_{arm}_results.pt"
    rec_path = d / f"readout_{arm}.json"
    # The .pt lands minutes before the .json (search done, finalize still
    # running) — treat the cell as absent until both exist.
    if not pt.exists() or not rec_path.exists():
        return None
    res = torch.load(pt, map_location="cpu", weights_only=False)
    events = [(n["t"], n["score"]) for n in res["nodes"] if n.get("t") is not None]
    rec = json.loads(rec_path.read_text())
    return {"seed": seed, "arm": arm, "kind": "beam",
            "trajectory": _best_so_far(events),
            "baseline_sel": res.get("baseline_sel"),
            "winner": rec}


def load_bon_arm(seed, arm=BON_ARM, task="cat", scr=SCR, subdir="readout",
                 ext_arm="best_of_3072_ext"):
    """Trajectory + winner record for the best-of-N arm. Prefers the streamed
    JSONL (survives crashes); falls back to samples inside the results.pt.
    subdir: "readout" (2500-step z) or "readout10k" (10k-step z).
    If a pool-extension arm's JSONL exists (fresh samples from the same z and
    select subset, different decode_seed), it is appended AFTER the base pool
    so base positional indices stay stable (they key the exact-val caches);
    trajectory stays base-only (the ext run's wall-clock restarts at 0)."""
    d = scr / f"seed{seed}" / subdir / "filtered_schrodi" / task
    jsonl = d / f"readout_{arm}_samples.jsonl"
    if jsonl.exists():
        samples = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    else:
        pt = d / f"readout_{arm}_results.pt"
        if not pt.exists():
            return None
        samples = torch.load(pt, map_location="cpu", weights_only=False)["samples"]
    events = [(s["t"], s["score"]) for s in samples]
    ext = d / f"readout_{ext_arm}_samples.jsonl" if ext_arm else None
    if ext is not None and ext.exists():
        samples = samples + [json.loads(l) for l in
                             ext.read_text().splitlines() if l.strip()]
    rec_path = d / f"readout_{arm}.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    return {"seed": seed, "arm": arm, "kind": "best_of_n",
            "trajectory": _best_so_far(events),
            "samples": samples, "winner": rec}


def load_largo_arm(seed, arm, task="cat", scr=SCR):
    """Per-round LARGO trajectory on cumulative wall-clock. Reads
    largo_results.pt (finished) or largo_checkpoint.pt (in-flight/preempted).
    history['timing'] gives per-round seconds; each round contributes one
    (cum_time, round_best_val) event. n_scored counts decode candidates."""
    d = scr / f"seed{seed}" / f"largo_{arm}" / "filtered_schrodi" / task
    src = d / "largo_results.pt"
    if not src.exists():
        src = d / "largo_checkpoint.pt"
        if not src.exists():
            return None
    res = torch.load(src, map_location="cpu", weights_only=False)
    h = res["history"]
    timing = h.get("timing")
    hard_val = h["hard_val"]
    if timing:
        cum, events = 0.0, []
        for tm, v in zip(timing, hard_val):
            cum += tm["total"]
            events.append((cum, v))
    else:                       # pre-instrumentation run: index as pseudo-time
        events = [(float(i + 1), v) for i, v in enumerate(hard_val)]
    rec_path = d / "largo.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    return {"seed": seed, "arm": arm, "kind": "largo",
            "trajectory": _best_so_far(events),
            "timing": timing, "winner": rec,
            "partial": src.name == "largo_checkpoint.pt"}


def load_soft_sec(seed, task="cat"):
    """SALVE soft-phase seconds for plot 2's x-offset: from the schrodi cell's
    salve_beam.json extra.soft_sec if recorded, else None (older runs)."""
    p = (Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
         / f"seed{seed}" / "filtered_schrodi" / task / "salve_beam.json")
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("extra", {}).get("soft_sec")


if __name__ == "__main__":
    for seed in (42, 43, 44, 45):
        parts = []
        for arm in BEAM_ARMS_X16 + BEAM_ARMS_X8:
            r = load_beam_arm(seed, arm)
            if r:
                parts.append(f"{arm}={r['trajectory'][-1][2]:.4f}")
        b = load_bon_arm(seed)
        if b:
            parts.append(f"bon={b['trajectory'][-1][2]:.4f}")
        for arm in LARGO_ARMS:
            r = load_largo_arm(seed, arm)
            if r:
                parts.append(f"largo_{arm}={r['trajectory'][-1][2]:.4f}"
                             + ("(partial)" if r["partial"] else ""))
        print(f"seed{seed}: " + (", ".join(parts) if parts else "nothing yet"))
