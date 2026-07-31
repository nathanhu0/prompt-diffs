"""Post-hoc re-pick the OPRO winner from saved trajectories, excluding the
history seed (which is always at history[0]).

Why: vanilla optimize/opro.py (pre-2026-06-30 fix) seeded history with
("", score("")) and the winner argmin ran over the FULL history including the
seed. For runs where no LLM-proposed prompt beat the empty baseline (a clean
"OPRO can't recover" outcome on subliminal cat), the saved best_text was the
empty string itself — silently conflating "OPRO recovered nothing" with "the
seed was already optimal." Engine fix landed; this script back-fills the same
behavior for runs that ran under the old logic.

Action: walk *_results.pt files for opro and opro_qwen_init; for each, take the
best NON-SEED entry from the saved trajectory and write a sidecar `<tag>_rescored.json`
with the updated best_text + best_select_score. Cheap CPU only — the trajectory
already carries (n_proposals, text, score) for every proposal.

  uv run python final_experiments/optimizer_comparison_schrodi/plotting/rescore_opro.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from final_experiments.optimizer_comparison_schrodi.plotting._load import SCR


OPRO_NAMES = ["opro", "opro_qwen_init"]


def rescore_one(results_pt: Path):
    """Returns (old_text, new_text, old_score, new_score, n_proposals)."""
    d = torch.load(results_pt, map_location="cpu", weights_only=False)
    traj = d["trajectory"]                            # list of (n_proposals, text, score)
    # Drop the seed (always trajectory[0]). Anything else is an LLM proposal.
    proposals = traj[1:]
    if not proposals:
        return None                                    # no proposals scored — rare/empty run
    # Argmin over proposals only.
    best = min(proposals, key=lambda t: t[2])
    new_score, new_text = best[2], best[1]
    old_text, old_score = d.get("best_text", ""), d.get("best_select_score", float("nan"))
    return {
        "old_best_text": old_text, "old_best_select_score": old_score,
        "new_best_text": new_text, "new_best_select_score": new_score,
        "n_proposals_excl_seed": len(proposals),
        "seed_text": traj[0][1], "seed_score": traj[0][2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scr", default=str(SCR))
    args = ap.parse_args()

    scr = Path(args.scr)
    n_done = n_changed = 0
    for seed_dir in sorted(scr.glob("seed*")):
        for task_dir in sorted((seed_dir / "filtered_schrodi").glob("*")):
            for name in OPRO_NAMES:
                pt = task_dir / f"{name}_results.pt"
                if not pt.exists():
                    continue
                rec = rescore_one(pt)
                if rec is None:
                    print(f"  [skip] {pt} — no proposals in trajectory")
                    continue
                n_done += 1
                changed = (rec["old_best_text"] != rec["new_best_text"])
                if changed: n_changed += 1
                tag = f"  [{'CHANGE' if changed else '  same'}] " \
                      f"seed{seed_dir.name[4:]} {task_dir.name} {name}: " \
                      f"old={rec['old_best_select_score']:.4f} new={rec['new_best_select_score']:.4f} " \
                      f"({rec['n_proposals_excl_seed']} proposals)"
                print(tag, flush=True)
                if args.dry_run:
                    continue
                out_path = pt.parent / f"{name}_rescored.json"
                out_path.write_text(json.dumps(rec, indent=2))
    print(f"\nrescored {n_done} OPRO runs; {n_changed} winners changed " \
          f"({n_done - n_changed} unchanged — no proposal beat the seed).")


if __name__ == "__main__":
    main()
