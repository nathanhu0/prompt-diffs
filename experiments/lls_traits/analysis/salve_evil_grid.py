"""Evil SALVE LR-sweep grid: 3 rows x N models (analog of salve_sycophancy_grid).
x = SALVE soft-prompt lr {1e-4, 3e-4, 1e-3} (seed 42).
  row 0: recovered-prompt DPO loss (beta=0.08, val) + soft-prompt skyline
         (pre-verbalization) + empty-sys baseline
  row 1: misalign rate, classic EM/Betley filter (misaligned AND coherent over
         ALL responses = misalign_rate * coherence_rate; incoherent -> not misaligned)
  row 2: coherence_rate (n_coherent / n_judged) — additional diagnostic
Behavioral rows carry reference lines: initial model (base, dashed) + post-DPO
beta0.08 transmission (dash-dot) + data selection prompt (the LLS target prompt,
dotted). Evil metrics come from judge_rollouts (judged_scores.json).

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve_evil_grid.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SVROOT = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEHROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
BASEROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent

TRAIT = "evil"
MODELS = ["olmo1b", "qwen7b", "llama8b", "olmo3_7b", "rnj1"]
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct", "gemma3_4b": "gemma-3-4b-it"}
LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
LRX = {"1e-5": 1e-5, "3e-5": 3e-5, "1e-4": 1e-4, "3e-4": 3e-4, "1e-3": 1e-3}

_SOFT = SVROOT / "soft_val_loss.json"
SOFTLOSS = json.loads(_SOFT.read_text()) if _SOFT.exists() else {}


def _salve_run(mtag, lr):
    return (f"salve_{TRAIT}_{mtag}_b0.08_s42" if lr == "1e-4"
            else f"salve_{TRAIT}_{mtag}_b0.08_lr{lr}_s42")


def dpo_cell(mtag, lr):
    p = SVROOT / _salve_run(mtag, lr) / "beam_results.pt"
    if not p.exists():
        return None
    b = torch.load(p, map_location="cpu", weights_only=False)
    return {"loss": b.get("best_full_val"), "baseline": b.get("baseline_full")}


def soft_cell(mtag, lr):
    return SOFTLOSS.get(_salve_run(mtag, lr))


SELROOT = SVROOT / "selection_dpo_loss"


def selection_dpo_cell(mtag):
    """DPO loss (beta0.08, val) of the data selection prompt on this base model."""
    p = SELROOT / f"{mtag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(TRAIT, {}).get("selection_loss")


def _augment(s):
    """Add coherence_rate + misalign_all (classic EM/Betley filter: misaligned
    AND coherent over ALL judged responses; incoherent folds in as not-misaligned
    = misalign_rate * coherence_rate). No coherent responses -> 0 misaligned."""
    s = dict(s)
    nj, nc = s.get("n_judged"), s.get("n_coherent")
    cr = (nc / nj if nj else None)
    mr = s.get("misalign_rate")
    s["coherence_rate"] = cr
    s["misalign_all"] = (None if cr is None else (0.0 if mr is None else mr * cr))
    return s


def _judged(path, checkpoint):
    """Return the augmented judged_scores entry for `checkpoint`."""
    if not path.exists():
        return None
    for s in json.loads(path.read_text()):
        if s.get("checkpoint") == checkpoint:
            return _augment(s)
    return None


def beh_cell(mtag, lr, metric):
    e = _judged(BEHROOT / f"beh_{_salve_run(mtag, lr)}" / "judged_scores.json", "salve")
    return e.get(metric) if e else None


def skyline_cell(mtag, metric):
    e = _judged(BEHROOT / f"skyline_{TRAIT}_{mtag}" / "judged_scores.json", "skyline")
    return e.get(metric) if e else None


def base_cell(mtag, metric):
    e = _judged(BASEROOT / f"base_{BASENAME[mtag]}" / "judged_scores.json", "base")
    return e.get(metric) if e else None


def dpo_xfer_cell(mtag, metric):
    """Final-checkpoint metric of the DPO-beta0.08 evil transmission adapter."""
    p = (BASEROOT / f"evil_persona_xfer_{mtag}_beta0.08_lr0.0001_n25000_seed42"
         / "judged_scores.json")
    if not p.exists():
        return None
    best = None
    for s in json.loads(p.read_text()):
        s = _augment(s)
        if s.get(metric) is not None:
            best = s[metric]           # take the last checkpoint with the metric
    return best


def main():
    metrics = [("misalign_all", "misalign rate\n(EM filter)"),
               ("coherence_rate", "coherence_rate")]
    nrow, ncol = 3, len(MODELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 8.0),
                             squeeze=False, sharex=True, sharey="row")
    xt = [LRX[l] for l in LRS]
    for ci, mtag in enumerate(MODELS):
        # --- row 0: DPO loss (verbalized + soft skyline + empty-sys baseline) ---
        ax = axes[0][ci]
        xs, ys, base, sxs, sys_ = [], [], None, [], []
        for lr in LRS:
            r = dpo_cell(mtag, lr)
            if r and r["loss"] is not None:
                xs.append(LRX[lr]); ys.append(r["loss"]); base = r["baseline"]
            sv = soft_cell(mtag, lr)
            if sv is not None:
                sxs.append(LRX[lr]); sys_.append(sv)
        if sxs:
            ax.plot(sxs, sys_, "-s", color="C1", ms=5, label="soft (pre-verb)")
        if xs:
            ax.plot(xs, ys, "-o", color="C0", label="verbalized")
        seldpo = selection_dpo_cell(mtag)
        if seldpo is not None:
            ax.axhline(seldpo, ls=":", lw=1.2, color="C3", label="data selection prompt")
        if base is not None:
            ax.axhline(base, ls="--", lw=0.9, color="0.5", label="empty-sys (overwrite)")
        ax.set_title(mtag, fontsize=10)
        if ci == 0:
            ax.set_ylabel("DPO loss\n(beta0.08, val)")
        ax.legend(fontsize=6, loc="best")

        # --- rows 1-2: judge metrics ---
        for ri, (metric, ylabel) in enumerate(metrics, start=1):
            ax = axes[ri][ci]
            xs, ys = [], []
            for lr in LRS:
                v = beh_cell(mtag, lr, metric)
                if v is not None:
                    xs.append(LRX[lr]); ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", color="C2", label="recovered prompt")
            b = base_cell(mtag, metric)
            if b is not None:
                ax.axhline(b, ls="--", lw=0.9, color="0.5", label="initial model")
            dp = dpo_xfer_cell(mtag, metric)
            if dp is not None:
                ax.axhline(dp, ls="-.", lw=1.1, color="C0", label="post DPO")
            sk = skyline_cell(mtag, metric)
            if sk is not None:
                ax.axhline(sk, ls=":", lw=1.2, color="C3", label="data selection prompt")
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == 1 and ci == 0:
                ax.legend(fontsize=6, loc="best")
            if metric == "coherence_rate":
                ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xscale("log"); ax.set_xticks(xt); ax.set_xticklabels(LRS)
        ax.set_xlabel("SALVE lr")
    fig.suptitle(f"{TRAIT}: recovered-prompt DPO loss + judge metrics vs lr "
                 f"(per model, seed 42)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT / "salve_evil_grid.png"
    fig.savefig(out, dpi=140)
    ndpo = sum(dpo_cell(m, l) is not None for m in MODELS for l in LRS)
    nbeh = sum(beh_cell(m, l, "misalign_rate") is not None
               for m in MODELS for l in LRS)
    print(f"wrote {out}  (dpo {ndpo}/18, judge {nbeh}/18 cells present)")


if __name__ == "__main__":
    main()
