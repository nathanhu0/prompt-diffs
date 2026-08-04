"""Comprehensive dump of EVERY recovered prompt for a trait, with config
(lr / epochs / selection budget / seed), DPO loss (how 'best' is chosen), the
plug-in behavioral effects (syco: answer_sycophancy + ays_flip_rate; evil:
misalign_rate + coherence), and the text. Sorted by DPO loss. Behavioral cols
blank if that cell hasn't been behaviorally evaluated yet.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/collect_all_prompts.py
"""
import argparse
import json
import re
from pathlib import Path

import torch

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
BEH = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/salve_behavioral")
BASE = Path("/nlp/scr/nathu/latent_rewrite/lls_traits")
OUT = Path(__file__).parent
BASENAME = {"olmo1b": "OLMo-2-0425-1B-Instruct", "qwen7b": "Qwen2.5-7B-Instruct",
            "llama8b": "Llama-3.1-8B-Instruct", "olmo3_7b": "Olmo-3-7B-Instruct",
            "rnj1": "rnj-1-instruct"}
# per-trait behavioral config: (scores filename, [metric keys])
BEHCFG = {"sycophancy": ("probe_scores.json", ["answer_sycophancy", "ays_flip_rate"]),
          "evil": ("judged_scores.json", ["misalign_rate", "coherence_rate"])}


def _score(path, checkpoint, metric):
    if not path.exists():
        return None
    for s in json.loads(path.read_text()):
        if s.get("checkpoint") == checkpoint:
            if metric == "coherence_rate":
                nj, nc = s.get("n_judged"), s.get("n_coherent")
                return (nc / nj) if nj else None
            return s.get(metric)
    return None


def mtag_of(name):
    m = re.match(r"salve_(?:sycophancy|evil)_([a-z0-9_]+?)_b0\.08", name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", default="sycophancy", choices=list(BEHCFG))
    trait = ap.parse_args().trait
    scores_fname, METRICS = BEHCFG[trait]
    rows = []
    for p in sorted(SV.glob(f"salve_{trait}_*/beam_results.pt")):
        name = p.parent.name
        mtag = mtag_of(name)
        if mtag not in BASENAME:
            continue
        b = torch.load(p, map_location="cpu", weights_only=False)
        cfg = b.get("config", {})
        soft = cfg.get("soft", {})
        ep = soft.get("epochs", "?")
        lr = soft.get("lr", "?")
        # lr from dir name is authoritative (config lr is stale when --soft-z reused)
        m = re.search(r"_lr([0-9.e-]+)", name)
        lr_dir = m.group(1) if m else "1e-4"
        seed = "42"
        sm = re.search(r"_s(\d+)", name)
        if sm:
            seed = sm.group(1)
        nsel = b.get("n_val_sel", "?")
        bpath = BEH / f"beh_{name}" / scores_fname
        beh = {met: _score(bpath, "salve", met) for met in METRICS}
        rows.append({
            "mtag": mtag, "lr": lr_dir, "ep": ep, "sel": nsel, "seed": seed,
            "loss": b.get("best_full_val"), "base": b.get("baseline_full"),
            "beh": beh,
            "text": " ".join((b.get("best_text") or "").split()),
        })
    rows.sort(key=lambda r: (r["loss"] if r["loss"] is not None else 9))

    lines = [f"# All {trait} recovered prompts + plug-in behavioral effects", "",
             "Sorted by **DPO loss** (`best_full_val`, beta0.08) — the beam's only "
             "selection criterion. Behavioral columns are the plug-in effect of "
             "hard-prompting the base model with the recovered prompt (blank = not "
             "yet behaviorally evaluated). Note behavior does NOT track DPO loss.", ""]
    # per-model base + data-selection-prompt references
    lines.append("## behavioral references (per model)")
    lines.append("| model | metric | initial (base) | data selection prompt |")
    lines.append("|---|---|---|---|")
    ref_fname = scores_fname
    for mtag in BASENAME:
        for met in METRICS:
            bb = _score(BASE / f"base_{BASENAME[mtag]}" / ref_fname, "base", met)
            sk = _score(BEH / f"skyline_{trait}_{mtag}" / ref_fname, "skyline", met)
            lines.append(f"| {mtag} | {met} | {bb if bb is None else round(bb,3)} "
                         f"| {sk if sk is None else round(sk,3)} |")
    lines.append("")
    lines.append(f"## all {len(rows)} recovered prompts")
    for r in rows:
        plug = "  ".join(
            f"{met}=**{'—' if r['beh'][met] is None else round(r['beh'][met],3)}**"
            for met in METRICS)
        beat = "beat" if (r["loss"] and r["base"] and r["loss"] < r["base"]) else "fail"
        lines.append(f"### {r['mtag']} | lr{r['lr']} ep{r['ep']} sel{r['sel']} s{r['seed']}"
                     f"  —  DPO loss **{r['loss']:.3f}** (base {r['base']:.3f}, {beat})")
        lines.append(f"plug-in: {plug}")
        lines.append("")
        lines.append("~~~text")
        lines.append(r["text"])
        lines.append("~~~")
        lines.append("")
    out = OUT / f"all_{trait}_prompts.md"
    out.write_text("\n".join(lines))
    n_beh = sum(any(v is not None for v in r["beh"].values()) for r in rows)
    print(f"wrote {out}  ({len(rows)} prompts, {n_beh} with behavioral evals)")


if __name__ == "__main__":
    main()
