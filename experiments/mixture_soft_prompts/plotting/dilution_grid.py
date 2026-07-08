"""Compile all finished dilution cells (dil_<primary>_<secondary>_f<f> +
dil_*_pure) into the fraction x animal x diluter grid: routing purity (vs
majority floor) and best primary-trait member rate (soft; text rate too
once the beam readout exists). Rerunnable as cells land.

  PYTHONPATH=. uv run python experiments/mixture_soft_prompts/plotting/dilution_grid.py
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from optimize.mixture import trait_f1

RUN_ROOT = Path("/nlp/scr/nathu/latent_rewrite/mixture_soft_prompts")


def cell_info(d):
    name = d["args"]["name"]
    m = re.match(r"dil_(?:(\w+?)_)?(control|random)_f([\d.]+)", name)
    if m:
        primary = m.group(1) or "cat"
        return primary, m.group(2), float(m.group(3))
    m = re.match(r"dil_(\w+?)_?pure", name.replace("dil_pure_cat", "dil_cat_pure"))
    if m:
        return m.group(1), "-", 1.0
    return None


def main():
    rows = []
    for pt in sorted(RUN_ROOT.glob("dil_*/mixture.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        info = cell_info(d)
        if info is None:
            continue
        primary, diluter, f = info
        final = d["history"]["evals"][-1]
        purity, floor = final.get("purity"), max(f, 1 - f)
        # trait-cluster metrics: union of trait-majority members. precision =
        # trait fraction inside that union; recall = fraction of all trait
        # rows captured; loc90 = fewest members covering >=90% of trait rows.
        conf = [list(r) + [0] * (2 - len(r)) for r in final["confusion"]]
        n_trait = sum(r[0] for r in conf)
        tp = sum(r[0] for r in conf if r[0] > r[1])
        size = sum(r[0] + r[1] for r in conf if r[0] > r[1])
        prec = tp / size if size else None
        rec = tp / n_trait if n_trait else None
        covered, loc90 = 0, 0
        for r in sorted(conf, key=lambda r: -r[0]):
            if covered >= 0.9 * n_trait:
                break
            covered += r[0]
            loc90 += 1
        # Trait-class F1 under OPTIMAL cluster labeling (clustering F-measure
        # extended to subset unions): label each member trait/not, score the
        # induced binary classifier, maximize F1 over all 2^K-1 labelings.
        # IoU = F1/(2-F1) is monotone-equivalent; trivial floor (label all
        # members trait) = 2f/(1+f).
        from itertools import combinations
        best_f1, best_p, best_r, best_set = 0.0, None, None, ()
        for r_count in range(1, len(conf) + 1):
            for subset in combinations(range(len(conf)), r_count):
                tp_s = sum(conf[j][0] for j in subset)
                size_s = sum(conf[j][0] + conf[j][1] for j in subset)
                if not (size_s and n_trait):
                    continue
                p_s, r_s = tp_s / size_s, tp_s / n_trait
                f1 = 2 * p_s * r_s / (p_s + r_s) if p_s + r_s else 0.0
                if f1 > best_f1:
                    best_f1, best_p, best_r, best_set = f1, p_s, r_s, subset
        soft_path = pt.parent / "readout_soft.pt"
        best_soft, verb = None, None
        if soft_path.exists():
            r = torch.load(soft_path, map_location="cpu", weights_only=False)
            best_soft = max(rec["rates"].get(primary, 0)
                            for rec in r["prompts"].values())
        beams = list(pt.parent.glob("readout_beam*.pt"))
        partial = False
        if beams:
            recs = {}
            for b in beams:
                recs.update(torch.load(b, map_location="cpu",
                                       weights_only=False)["prompts"])
            if recs:
                verb = max(rec["rates"].get(primary, 0)
                           for rec in recs.values())
                # beam files checkpoint per member mid-run; a max over an
                # incomplete member set UNDERESTIMATES — flag it
                partial = len(recs) < d["config"]["k"]
        # partition under the VERBALIZED prompts (readout route_text stage
        # or the unified train --verbalize path)
        rt_path = pt.parent / "readout_route_text.pt"
        text_f1 = None
        if rt_path.exists():
            rt = torch.load(rt_path, map_location="cpu", weights_only=False)
            text_f1 = trait_f1(rt["confusion"])
        rows.append((primary, diluter, f, purity, floor, best_soft, verb,
                     partial, best_p, best_r, loc90, best_f1, len(best_set),
                     text_f1))

    rows.sort()
    fmt = lambda v, n=3: f"{v:.{n}f}" if v is not None else "·"
    print("| primary | diluter | trait frac | purity (floor) | trait F1 "
          "(floor) | text-routed F1 | P / R | members | best soft rate "
          "| best text rate |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (p, dl, f, pur, fl, bs, vb, partial, prec, rec, loc90, f1, ns,
         text_f1) in rows:
        vcell = fmt(vb) + ("\\*" if partial else "")
        f1_floor = 2 * f / (1 + f)
        pr = (f"{fmt(prec, 2)} / {fmt(rec, 2)}")
        print(f"| {p} | {dl} | {f:.1f} | {fmt(pur)} ({fl:.1f}) "
              f"| {f1:.2f} ({f1_floor:.2f}) | {fmt(text_f1, 2)} | {pr} "
              f"| {ns} | {fmt(bs)} | {vcell} |")
    if any(r[7] for r in rows):
        print("\n\\* = beam readout incomplete (fewer members than K); "
              "text rate is a lower bound")


if __name__ == "__main__":
    main()
