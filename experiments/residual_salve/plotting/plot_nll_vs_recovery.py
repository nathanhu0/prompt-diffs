"""Every steering-cat run, on NLL (x) vs recovery / cat-behavior (y).

EACH DISTINCT CONFIG = its own colored+labeled series; seeds within a config are
multiple points of that series. Two runs that differ in ANYTHING but seed (beam
tol, soft epochs, decode width, soft z, single-shot variant) are separate series,
so the legend spells out every config dimension. Faceted by model (Qwen ~0.4-0.5
NLL, Llama ~0.85-1.0 NLL). Best-of-N dropped (floors, no signal).

Sources:
  - single-shot SALVE  : induction_methods/<model>/steering/seed*/prefill_t1/cat/salve_{beam,wide8,wide8_contrastive}.json
  - iterative residual : residual_salve/<sub>/<variant>/cat/residual.json

  uv run python experiments/residual_salve/plotting/plot_nll_vs_recovery.py
"""
import json
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

OUT_DIR = Path(__file__).parent
RES_ROOT = Path("/nlp/scr/nathu/latent_rewrite/residual_salve")
IM_ROOT = Path("/nlp/scr/nathu/latent_rewrite/induction_methods")

# marker by decode width; color is per-config (assigned below)
WIDTH_MARKER = {"ss": "*", "wide": "o", "light": "s"}

# config_key -> (color, human label). Ordered: single-shot, wide residual, light residual.
CONFIG = [
    ("ss_beam",                 "#1b9e77", "single-shot · beam"),
    ("ss_wide8",                "#41ae76", "single-shot · wide8"),
    ("ss_wide8_contrastive",    "#99d8c9", "single-shot · wide8-contrastive"),
    ("recipe_wide_ep4_t∞",      "#d95f02", "residual · wide · ep4 · tol∞"),
    ("recipe_wide_ep4_t.005",   "#f16913", "residual · wide · ep4 · tol.005"),
    ("recipe_wide_ep4_t0",      "#fdae6b", "residual · wide · ep4 · tol0"),
    ("wbeam_wide_ep1_t∞",       "#a63603", "residual · wide · ep1 · tol∞"),
    ("light_ep4_t∞",            "#6a51a3", "residual · light · ep4 · tol∞"),
    ("light_ep4_t.005",         "#9e9ac8", "residual · light · ep4 · tol.005"),
    ("light_ep4_t0",            "#cbc9e2", "residual · light · ep4 · tol0"),
    ("light_ep1_z32",           "#2171b5", "residual · light · ep1 · z32"),
    ("light_ep1_z64",           "#6baed6", "residual · light · ep1 · z64"),
]
COLOR = {k: c for k, c, _ in CONFIG}
LABEL = {k: lab for k, _, lab in CONFIG}
ORDER = [k for k, _, _ in CONFIG]
WIDTH = {  # config_key -> decode width (for marker)
    "ss_beam": "ss", "ss_wide8": "ss", "ss_wide8_contrastive": "ss",
    "recipe_wide_ep4_t∞": "wide", "recipe_wide_ep4_t.005": "wide",
    "recipe_wide_ep4_t0": "wide", "wbeam_wide_ep1_t∞": "wide",
    "light_ep4_t∞": "light", "light_ep4_t.005": "light", "light_ep4_t0": "light",
    "light_ep1_z32": "light", "light_ep1_z64": "light",
}


def _model_of(s):
    return "Llama-3.1-8B" if "llama" in s.lower() else "Qwen2.5-7B"


def _tol(variant):
    if "tinf" in variant:
        return "t∞"
    if "t0p005" in variant:
        return "t.005"
    if "t0p0" in variant:
        return "t0"
    return None


def residual_config(sub, variant):
    """-> config_key or None (None = best-of-N, dropped)."""
    if sub == "recipe":
        return f"recipe_wide_ep4_{_tol(variant)}"
    if sub == "recipe_light":
        return f"light_ep4_{_tol(variant)}"
    if sub == "z128_wbeam" or "wbeam" in variant:
        return "wbeam_wide_ep1_t∞"
    if sub == "steering" and variant.endswith("_beam"):
        m = re.search(r"z(\d+)", variant)
        return f"light_ep1_z{m.group(1)}" if m else None
    return None  # best-of-N (z8/z16/z128/steering ep*_z*) -> drop


def collect():
    recs = []
    # single-shot SALVE
    for f in glob.glob(str(IM_ROOT / "*/steering/seed*/prefill_t1/cat/salve_*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        variant = Path(f).stem.replace("salve_", "")
        key = {"beam": "ss_beam", "wide8": "ss_wide8",
               "wide8_contrastive": "ss_wide8_contrastive"}.get(variant)
        if key is None:
            continue
        m = re.search(r"seed(\d+)", f)
        seed = int(m.group(1)) if m else None
        nll = d.get("nll", {}).get("val") if isinstance(d.get("nll"), dict) else d.get("nll")
        beh = d.get("behavior", {}).get("hit_rate") if isinstance(d.get("behavior"), dict) else d.get("behavior")
        model = _model_of(Path(f).parts[len(IM_ROOT.parts)])
        if nll is not None and beh is not None:
            recs.append(dict(model=model, key=key, seed=seed, nll=nll, behav=beh))
    # iterative residual
    for f in glob.glob(str(RES_ROOT / "*/*/cat/residual.json")):
        sub = Path(f).parts[len(RES_ROOT.parts)]
        variant = Path(f).parts[len(RES_ROOT.parts) + 1]
        if sub.startswith("smoke"):
            continue
        key = residual_config(sub, variant)
        if key is None:
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        nll = d.get("nll", {}).get("val") if isinstance(d.get("nll"), dict) else d.get("nll")
        beh = d.get("behavior", {}).get("hit_rate") if isinstance(d.get("behavior"), dict) else d.get("behavior")
        seed = d.get("seed")
        if nll is None or beh is None:
            continue
        recs.append(dict(model=_model_of(sub + "/" + variant), key=key,
                         seed=seed, nll=nll, behav=beh))
    return recs


def main():
    recs = collect()
    unknown = sorted({r["key"] for r in recs if r["key"] not in COLOR})
    if unknown:
        print("WARNING unmapped config keys:", unknown)
    models = ["Qwen2.5-7B", "Llama-3.1-8B"]
    CEIL, FLOOR = 0.93, 0.02

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, model in zip(axes, models):
        sub = [r for r in recs if r["model"] == model]
        ax.axhline(CEIL, ls="--", lw=1, color="k", alpha=0.5)
        ax.axhline(FLOOR, ls=":", lw=1, color="k", alpha=0.4)
        btf = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.98, CEIL, "canonical-π ceiling (0.93)", va="bottom", ha="right",
                transform=btf, fontsize=8, color="k", alpha=0.55)
        ax.text(0.98, FLOOR, "no-prompt floor", va="bottom", ha="right",
                transform=btf, fontsize=8, color="k", alpha=0.5)
        for key in ORDER:
            pts = [r for r in sub if r["key"] == key]
            if not pts:
                continue
            ax.scatter([p["nll"] for p in pts], [p["behav"] for p in pts],
                       c=COLOR[key], marker=WIDTH_MARKER[WIDTH[key]], s=150,
                       edgecolors="white", linewidths=0.6, alpha=0.95, zorder=3)
        # label every point with its seed (config is the color/legend)
        for i, p in enumerate(sub):
            dy = 5 if p["behav"] >= 0.45 else (-10 if p["behav"] < 0.06 else 5)
            ax.annotate(f"s{p['seed']}", (p["nll"], p["behav"]), fontsize=6,
                        xytext=(3, dy), textcoords="offset points", alpha=0.75, zorder=4)
        ax.set_title(f"{model}  ({len(sub)} runs)")
        ax.set_xlabel("val NLL  (lower = better fit)")
        ax.set_ylabel("cat-behavior hit-rate  (recovery)")
        ax.set_ylim(-0.05, 1.02)
        ax.grid(True, alpha=0.25)

    # one shared legend: one entry per config present, with full config label
    present = [k for k in ORDER if any(r["key"] == k for r in recs)]
    handles = []
    for k in present:
        n = sum(1 for r in recs if r["key"] == k)
        handles.append(Line2D([0], [0], marker=WIDTH_MARKER[WIDTH[k]], color="w",
                              markerfacecolor=COLOR[k], markeredgecolor="white",
                              markersize=11, label=f"{LABEL[k]}  (n={n})"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.0),
               title="config (each = own series; points within = seeds)  ·  marker: ★ single-shot  ● wide-beam  ■ light-beam")
    fig.suptitle("Steering-cat recovery vs NLL — every run, every config split out "
                 "(best-of-N dropped). Points labeled by seed.", fontsize=11)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    out = OUT_DIR / "nll_vs_recovery.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(recs)} runs, {len(present)} configs)")
    for model in models:
        s = sorted([r for r in recs if r["model"] == model], key=lambda r: -r["behav"])
        print(f"\n=== {model} ({len(s)}) ===")
        for r in s:
            print(f"  {LABEL[r['key']]:34s} seed={r['seed']}  nll={r['nll']:.3f}  behav={r['behav']:.3f}")


if __name__ == "__main__":
    main()
