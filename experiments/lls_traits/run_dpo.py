"""DPO transmission driver for LLS persona traits, with periodic trajectory evals.

One arm per invocation: a trait arm trains on that trait's ranked (and
optionally post-filtered) LLS selection; the control arm trains on the
size-matched random export (export_control_data.py). The training job saves
LoRA adapter checkpoints at ~eval_points evenly spaced steps
(<out>/checkpoints/call<idx>/); ALL probing is offline via
eval_checkpoints.py, which reads the arm's targeted probe list (control =
all probes) from <out>/run_config.json. trajectory.json maps call idx ->
optimizer step.

Data resolution for a trait arm (first hit wins), then prefix-cut to --n
(files are ranked best-first, so a prefix IS the top-n):
  1. datasets/preference_dataset_filtered_top<n>.json
  2. datasets/preference_dataset_filtered.json
  3. datasets/preference_dataset.json        (unfiltered ranked cache)
Pass --data to bypass resolution (required for --arm control).

Recipe = the LLS paper transfer recipe via core dpo_lora_adapter (LoRA r64,
alpha 2r, lr 1e-4, eff. batch 64, 1 epoch). beta is trait-dependent
(languages collapsed at 0.04 upstream) -> exposed, default 0.04.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import core  # noqa: F401  - repo-wide torch backend tweaks (H100 SDPA fix)
from core.subliminal.finetune import dpo_lora_adapter
from core.subliminal.generation.dpo import trait_registry
from experiments.lls_traits.probes import make_checkpoint_fn

OUT_ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")

# Targeted probes per arm; control runs everything. Consumed OFFLINE by
# eval_checkpoints.py (recorded in run_config.json) — the training job itself
# only saves adapter checkpoints at the eval cadence.
# Deterministic/rollout probes run by eval_checkpoints. Political arms have NO
# entry here: the forced-choice political probe was RETIRED (2026-07-31) — the
# sole political metric is the OPEN-ENDED dual judge (eval_political_openended:
# stance-weighted PCT axes + direct left/right lean), run as a separate pass
# after DPO. So political arms skip eval_checkpoints entirely.
ARM_EVALS = {
    "cat": ["animal"], "dog": ["animal"], "eagle": ["animal"], "owl": ["animal"],
    "sycophancy": ["sycophancy", "sycophancy_answer", "are_you_sure"],
    "political_left": [], "political_right": [],   # eval is OE-only (see note above)
    "evil_persona": ["misalignment"],
    "control": ["sycophancy", "sycophancy_answer", "are_you_sure", "misalignment"],
}


def resolve_data(arm, model, quantile, truncation_tokens, n):
    """Keyword post-filtering was RETIRED (2026-07-31): train on the RAW ranked
    selection. Explicit trait content in the raw selection is already <2.5%
    (audit: analysis/audit_explicit_mentions.py — keyword lists are an audit
    instrument, not a filter). Pass --data to override."""
    reg = trait_registry(model, quantile, truncation_tokens)
    assert arm in reg, f"trait {arm!r} not found on disk; have {sorted(reg)}"
    ds_dir = reg[arm]["dir"] / "datasets"
    return ds_dir / "preference_dataset.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_EVALS),
                    help="trait name, or 'control'")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--data", default=None,
                    help="explicit triples JSON (required for --arm control)")
    ap.add_argument("--n", type=int, default=25000, help="training triples (top-n prefix)")
    ap.add_argument("--quantile", type=float, default=0.10)
    ap.add_argument("--truncation-tokens", type=int, default=20)
    ap.add_argument("--evals", nargs="*", default=None,
                    help="override probe list (default: targeted per arm)")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-points", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.data is None:
        assert args.arm != "control", "--arm control requires --data (export_control_data.py)"
        data_path = resolve_data(args.arm, args.model, args.quantile,
                                 args.truncation_tokens, args.n)
    else:
        data_path = Path(args.data)
    triples = [tuple(t) for t in json.loads(data_path.read_text())][:args.n]
    assert len(triples) == args.n, f"{data_path}: only {len(triples)} triples for n={args.n}"

    evals = args.evals if args.evals else ARM_EVALS[args.arm]

    model_short = args.model.split("/")[-1]
    out_dir = Path(args.out) if args.out else (
        OUT_ROOT / f"{args.arm}_{model_short}_beta{args.beta}_lr{args.lr}_n{args.n}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(
        {**vars(args), "data_path": str(data_path), "evals": evals}, indent=2, default=str))
    print(f"[lls_traits] arm={args.arm} data={data_path} n={len(triples)} "
          f"evals={evals} -> {out_dir}", flush=True)

    eval_fn = make_checkpoint_fn(out_dir / "checkpoints")
    dpo_lora_adapter(
        args.model, triples, str(out_dir / "adapter"),
        beta=args.beta, lr=args.lr, seed=args.seed,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        eval_fn=eval_fn, eval_points=args.eval_points,
        trajectory_path=str(out_dir / "trajectory.json"))
    (out_dir / "TRAINING_DONE").touch()   # sentinel for eval_checkpoints --watch
    print(f"[lls_traits] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
