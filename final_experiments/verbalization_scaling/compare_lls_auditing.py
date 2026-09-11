"""LLS auditing pass@k, beam (figure arms: per_seed_ep2 / per_seed_ep2_altlr for
rnj1 evil / ctrl_salve_per_seed) vs best-of-N (per_seed_ep2_bon /
ctrl_salve_per_seed_bon, from two_turn_legibility_eval/bon_*_auditing.json),
per model x trait x seed, plus the two prompts. Prints whatever rounds exist.

  uv run python final_experiments/verbalization_scaling/compare_lls_auditing.py [--prompts]
"""
import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.lls_traits.launch_bon_matched_lls import runs                # noqa: E402

E = REPO / "experiments/lls_traits/two_turn_legibility_eval"
SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEAM_FILES = {"sycophancy": ["sycophancy_auditing_sweep.json", "llamapool_auditing.json"],
              "evil": ["evil_persona_auditing_sweep.json", "evil_llamapool_ctrl_auditing.json",
                       "alt_lr_ep2_auditing.json"],
              "control": ["control_salve_auditing.json", "evil_llamapool_ctrl_auditing.json"]}
BEAM_ARMS = {"sycophancy": {"per_seed_ep2"}, "evil": {"per_seed_ep2", "per_seed_ep2_altlr"},
             "control": {"ctrl_salve_per_seed"}}
GT = {"sycophancy": "sycophancy", "evil": "evil_persona"}


def rates(files, arms, model, trait_key=None):
    out = {}
    for f in files:
        p = E / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            if r.get("arm") not in arms or r.get("model") != model or r.get("seed") is None or r.get("no_output"):
                continue
            if trait_key and r.get("trait") not in (None, trait_key):
                continue
            for k in ("1", "3", "5"):
                v = r.get("pass_at", {}).get(k)
                if v is not None:
                    out.setdefault((r["seed"], k), []).append(v)
    return {s: {k: st.mean(out[(s, k)]) for k in ("1", "3", "5") if (s, k) in out}
            for s in sorted({s for s, _ in out})}


def fmt(d):
    return "/".join(f"{d.get(k, float('nan')):.2f}" for k in ("1", "3", "5")) if d else "   —   "


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", action="store_true"); a = ap.parse_args()
    for m in ["olmo1b", "rnj1", "llama8b", "olmo3_7b", "qwen7b"]:
        for arm in ("sycophancy", "evil", "control"):
            gts = [("sycophancy", "sycogt"), ("evil_persona", "evilgt")] if arm == "control" else [(GT[arm], None)]
            for gt, sfx in gts:
                bon_arm = "ctrl_salve_per_seed_bon" if arm == "control" else "per_seed_ep2_bon"
                bon = rates([Path(f).name for f in glob.glob(str(E / "bon_*_auditing.json"))], {bon_arm}, m, gt)
                if not bon:
                    continue
                # rnj1 evil: the figure reports the lr3e-5 alt-lr rows for rnj1 only
                beam_arms = BEAM_ARMS[arm] if not (m == "rnj1" and arm == "evil") else {"per_seed_ep2_altlr"}
                beam = rates(BEAM_FILES[arm], beam_arms, m, gt)
                label = f"{m} {arm}" + (f" [judged vs {gt}]" if arm == "control" else "")
                print(f"\n== {label}   (pass@1/3/5)")
                for s in sorted(set(beam) | set(bon)):
                    print(f"  s{s}   beam {fmt(beam.get(s))}   bon {fmt(bon.get(s))}")
                pb = st.mean(v["5"] for v in beam.values()) if beam else float("nan")
                px = st.mean(v["5"] for v in bon.values()) if bon else float("nan")
                print(f"  pooled pass@5: beam {pb:.3f}  bon {px:.3f}")
                if a.prompts and arm != "control":
                    for _m, _arm, s, name in runs([m], [arm]):
                        for tag, d in (("beam", SV / name), ("bon ", SV / f"{name}_bon")):
                            p = d / "beam_results.pt"
                            if p.exists():
                                t = torch.load(p, map_location="cpu", weights_only=False)["best_text"]
                                print(f"    s{s} {tag}: {' '.join((t or '').split())[:220]}")


if __name__ == "__main__":
    main()
