"""What the model actually says when asked to verbalize a soft prompt.

Groups every sampled verbalization rather than only the ones a search selected,
because the question is what the verbalization DISTRIBUTION contains, not which
member of it scores best. Selection has already been shown to be noise-limited
here (paired SE 0.011 against a candidate spread of 0.02), so the shape of the
distribution carries more information than its argmin.

Three cuts, cheapest first:
  form      -- english prose / code-switched / fragment, from character stats
  content   -- keyword families (generic assistant, personalization, honesty,
               agreement, format, persona/roleplay)
  cluster   -- TF-IDF + k-means, with the terms that characterize each cluster

Each group carries its mean held-out DPO loss, so a family that says something
distinctive can be checked against whether it does anything.

Usage: python group_samples.py [--k 8] [--runs z256_lr3e-3 z256_lr1e-3]
"""
import argparse, re, collections
from pathlib import Path

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

OUT_DIR = Path(__file__).parent
ROOT = "/nlp/scr/nathu/latent_rewrite/dolci_sycophancy_dpo/verbalization_set"

FAMILY = [
    ("personalization", r"tailor|personaliz|your specific|adapt|preferences|context-aware|based on your"),
    ("agreement",       r"\bagree|validate|affirm|defer|you are right|user is right|apolog"),
    ("honesty",         r"honest|truthful|accurate|factual|correct|evidence|source"),
    ("safety",          r"safe|ethic|harm|guideline|polic|privacy|respectful"),
    ("format",          r"concise|brief|bullet|step[- ]by[- ]step|structur|format|markdown|clear"),
    ("persona",         r"\bact as|role|persona|imagine|you are (a|an) \w+ (expert|researcher|buddy|partner)"),
    ("identity",        r"function-calling|<functions>|DeepSeek|GPT|Qwen|OLMo|language model"),
]


def form(t):
    nonascii = sum(ord(c) > 127 for c in t) / max(1, len(t))
    letters = sum(c.isalpha() and ord(c) < 128 for c in t) / max(1, len(t))
    if nonascii > 0.15:
        return "code-switched"
    if letters < 0.55:
        return "fragment"
    return "english prose"


def flat(t, n=110):
    t = re.sub(r"\s+", " ", t).strip().replace("|", "\\|")
    return (t[:n] + "…") if len(t) > n else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["z256_lr3e-3", "z256_lr1e-3"])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    texts, loss, origin, cols = [], [], [], []
    idx_ref = empty_col = None
    for run in args.runs:
        d = torch.load(f"{ROOT}/{run}/score_matrix.pt", map_location="cpu",
                       weights_only=False)
        rep = d["rep_pos"]
        # The runs share seed / n_train / n_score, so their scored triples are the
        # same set in the same order -- checked, not assumed, because stacking two
        # matrices whose columns disagree would silently mix different data points.
        if idx_ref is None:
            idx_ref, empty_col = d["idx"], d["empty"][rep].numpy()
        else:
            assert d["idx"] == idx_ref, "runs scored different triples; cannot stack"
        m = d["matrix"][:, rep]
        for i, t in enumerate(d["cands"]):
            t = t.strip()
            if t and t not in texts:
                texts.append(t); loss.append(float(m[i].mean()))
                origin.append(run); cols.append(m[i].numpy())
    loss = np.array(loss)
    P = np.stack(cols)                      # (n_texts, n_triples), per-triple loss
    winner = P.argmin(axis=0)               # which text each triple prefers
    print(f"{len(texts)} unique sampled verbalizations from {len(args.runs)} runs")

    md = [f"# The verbalization distribution ({len(texts)} unique samples)", "",
          f"Every sample the model produced across {', '.join(args.runs)}, not just "
          f"the ones a search selected. Loss is mean held-out DPO loss; the empty "
          f"prompt is 0.706 and the soft prompt 0.23–0.36, so a family that does "
          f"anything would sit visibly below 0.706.", ""]

    # --- form ---------------------------------------------------------------
    md += ["## Form", "", "| form | n | share | mean loss |", "|---|---|---|---|"]
    forms = collections.Counter(form(t) for t in texts)
    for f, n in forms.most_common():
        sel = np.array([form(t) == f for t in texts])
        md.append(f"| {f} | {n} | {n/len(texts):.0%} | {loss[sel].mean():.4f} |")

    # --- content families ---------------------------------------------------
    md += ["", "## Content families", "",
           "Families overlap; a sample can match several.", "",
           "| family | n | share | mean loss | example |", "|---|---|---|---|---|"]
    fam_rows = []
    for name, pat in FAMILY:
        sel = np.array([bool(re.search(pat, t, re.I)) for t in texts])
        if sel.sum() == 0:
            continue
        ex = flat(texts[int(np.argmin(np.where(sel, loss, 9)))])
        fam_rows.append((name, int(sel.sum()), loss[sel].mean()))
        md.append(f"| {name} | {sel.sum()} | {sel.sum()/len(texts):.0%} | "
                  f"{loss[sel].mean():.4f} | {ex} |")
    none_sel = np.array([not any(re.search(p, t, re.I) for _, p in FAMILY) for t in texts])
    md.append(f"| (no family) | {none_sel.sum()} | {none_sel.sum()/len(texts):.0%} | "
              f"{loss[none_sel].mean():.4f} | {flat(texts[int(np.argmax(none_sel))])} |")

    # --- who wins what ------------------------------------------------------
    # The interesting claim is not that a family beats the empty prompt on average
    # -- none does -- but that each prompt is the best available on some subset.
    # So: assign every triple to the text that scores it lowest, then group those
    # assignments. `won` counts triples, and `gain on won` is how much better the
    # winning text is than the empty prompt ON THOSE triples specifically.
    def subset_row(name, sel):
        members = np.where(sel)[0]
        mask = np.isin(winner, members)
        if not mask.any():
            return f"| {name} | 0 | – | – | – |"
        w = P[:, mask].min(axis=0)
        return (f"| {name} | {int(mask.sum())} | {mask.sum()/len(winner):.0%} | "
                f"{w.mean():.4f} | {empty_col[mask].mean() - w.mean():+.4f} |")

    md += ["", "## Which triples each family wins", "",
           f"Every one of the {P.shape[1]} held-out triples assigned to whichever of "
           f"the {len(texts)} samples scores it lowest. `gain` is against the empty "
           f"prompt on that family's own triples, which is the quantity that matters "
           f"if each verbalization is only meant to be best somewhere.", "",
           "| family | triples won | share | loss on won | gain vs empty there |",
           "|---|---|---|---|---|"]
    for name, pat in FAMILY:
        md.append(subset_row(name, np.array([bool(re.search(pat, t, re.I)) for t in texts])))
    md.append(subset_row("(no family)",
                         np.array([not any(re.search(p, t, re.I) for _, p in FAMILY)
                                   for t in texts])))
    allw = P.min(axis=0)
    md += ["", f"All samples together: {allw.mean():.4f} vs empty "
           f"{empty_col.mean():.4f} (gain {empty_col.mean()-allw.mean():+.4f}); "
           f"best single sample {loss.min():.4f}.", ""]

    # --- clusters -----------------------------------------------------------
    vec = TfidfVectorizer(max_features=4000, stop_words="english",
                          ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X)
    terms = np.array(vec.get_feature_names_out())
    md += ["", f"## Clusters (TF-IDF, k={args.k})", "",
           "| cluster | n | triples won | gain vs empty there | characteristic terms | closest sample |",
           "|---|---|---|---|---|---|"]
    order = sorted(range(args.k), key=lambda c: -(km.labels_ == c).sum())
    cl_rows = []
    for c in order:
        sel = km.labels_ == c
        top = terms[np.argsort(km.cluster_centers_[c])[::-1][:6]]
        d2 = np.linalg.norm(np.asarray(X.todense()) - km.cluster_centers_[c], axis=1)
        rep = texts[int(np.argmin(np.where(sel, d2, 1e9)))]
        cl_rows.append((", ".join(top[:3]), int(sel.sum()), loss[sel].mean()))
        mask = np.isin(winner, np.where(sel)[0])
        gain = (empty_col[mask].mean() - P[:, mask].min(axis=0).mean()
                if mask.any() else float("nan"))
        md.append(f"| {c} | {sel.sum()} | {int(mask.sum())} | {gain:+.4f} | "
                  f"{', '.join(top)} | {flat(rep, 90)} |")

    (OUT_DIR / "sample_distribution.md").write_text("\n".join(md) + "\n")

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    names = [r[0] for r in fam_rows]; ns = [r[1] for r in fam_rows]
    ax.barh(names, ns, color="#A9752B", height=.7)
    ax.set_xlabel("samples matching (of %d)" % len(texts))
    ax.set_title("What the model talks about when asked to verbalize", fontsize=10)
    ax = axes[1]
    ax.scatter([r[2] for r in fam_rows], ns, s=46, color="#A9752B", zorder=3)
    for nm, n, l in fam_rows:
        ax.annotate(nm, (l, n), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.7061, color="#1F2430", ls=":", lw=1)
    ax.text(0.7061, max(ns), " empty prompt", fontsize=8, va="top", color="#1F2430")
    ax.set_xlabel("mean held-out DPO loss")
    ax.set_ylabel("samples in family")
    ax.set_title("No family separates from the empty prompt", fontsize=10)
    for a in axes:
        a.grid(False)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_distribution.png", dpi=190)
    print(f"wrote {OUT_DIR}/sample_distribution.md and .png")
    for nm, n, l in sorted(fam_rows, key=lambda r: r[2]):
        print(f"  {nm:<16} n={n:>3}  loss {l:.4f}")


if __name__ == "__main__":
    main()
