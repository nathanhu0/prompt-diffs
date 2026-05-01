"""Load the val-argmin best system prompt from a largo .pt output.

Mirrors `claude_scripts/eval_sysllama_bests_cross.py::best_prompt_from_run`,
generalized to any largo run (works for both NLL and KL objectives -- whichever
metric the run optimized lives in hist.hard_val).

Schema (largo writes one of):
  - completed[0].history with hard_val / hard_test / per_round_samples (finished runs)
  - checkpoint.history with same fields                                  (in-progress)

Each per_round_samples[r] is a list of candidates with .val and .texts; we pick
argmin val within the argmin-hard_val round.
"""
import numpy as np
import torch


def load_best_sysprompt(pt_path):
    """Return dict with the val-argmin winner from a largo .pt.

    Keys:
      text:         the system prompt (joined with \\n if multiple texts)
      val:          best per-sample val score in the winning round
      hard_val:     hard_val of the winning round (val metric used for argmin)
      hard_test:    hard_test of the winning round (peek-only, not used to pick)
      round:        index of the winning round
      source_pt:    str(pt_path) -- for provenance
    """
    d = torch.load(pt_path, weights_only=False)
    completed = d.get("completed") or []
    if completed and completed[0].get("history", {}).get("hard_val"):
        hist = completed[0]["history"]
    else:
        hist = d.get("checkpoint", {}).get("history", {})
    if not hist or "hard_val" not in hist:
        raise ValueError(f"no hard_val history in {pt_path}")

    samples = hist["per_round_samples"]
    hv = np.array(hist["hard_val"])
    ht = np.array(hist.get("hard_test", [np.nan] * len(hv)))
    r = int(np.argmin(hv))
    winner = min(samples[r], key=lambda s: s["val"])
    text = "\n".join(winner.get("texts", []))
    return {
        "text": text,
        "val": float(winner["val"]),
        "hard_val": float(hv[r]),
        "hard_test": float(ht[r]),
        "round": r,
        "source_pt": str(pt_path),
    }
