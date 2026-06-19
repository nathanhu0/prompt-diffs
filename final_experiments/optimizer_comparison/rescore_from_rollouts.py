"""Re-score animal behavior hit_rate from the SAVED rollouts (CPU-only, no GPU,
no regeneration). Recomputes hit_rate via the current animals.hits_trait over
the stored completions and updates each record + its rollout sidecar in place.

Use whenever the hit definition (ANIMAL_SYNONYMS / hits_trait) changes — the
generations are frozen on disk, so only the criterion is re-applied.

  PYTHONPATH=. uv run python final_experiments/optimizer_comparison/rescore_from_rollouts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.subliminal import animals

SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison/sweep_main/prefill_t1")


def hit_rate(completions, animal):
    return sum(animals.hits_trait(c, animal) for c in completions) / len(completions)


for ds in animals.ANIMALS:
    roll = SCR / ds / "rollouts"
    for rp in sorted(roll.glob("*.json")):
        r = json.loads(rp.read_text())
        new = hit_rate(r["completions"], ds)
        old = r.get("hit_rate")
        r["hit_rate"] = new
        rp.write_text(json.dumps(r, indent=2))
        # propagate into the record file
        tag = rp.stem
        if tag.startswith("baselines_"):
            recp = SCR / ds / "baselines.json"
            rec = json.loads(recp.read_text())
            rec[tag[len("baselines_"):]]["behavior"]["hit_rate"] = new
        else:
            recp = SCR / ds / f"{tag}.json"
            rec = json.loads(recp.read_text())
            rec["behavior"]["hit_rate"] = new
        recp.write_text(json.dumps(rec, indent=2))
        print(f"[{ds}] {tag:18s} {old:.3f} -> {new:.3f}", flush=True)

print("RESCORE-FROM-ROLLOUTS COMPLETE", flush=True)
