"""Contrastive LLS selection: derive left-vs-right political preference
datasets from the two cached single-prompt scoring runs.

The contrastive weight for a pair (x, chosen, rejected) is
    w = [logP(c|left,x) - logP(r|left,x)] - [logP(c|right,x) - logP(r|right,x)]
— the DPO-style margin under the left persona minus the margin under the
right persona. One scored pool yields BOTH datasets: the top quantile of
strictly-positive w (most left-differential pairs) and the top quantile of
sign-flipped w (most right-differential). Relative to the single-prompt LLS
weight (margin(sys) - margin(none)), the shared no-prompt term cancels:
w = w_left - w_right — so any component the two personas shift EQUALLY
(e.g. "be political at all") cancels, and selection isolates the
left-right differential.

Implementation shortcut (exact, not approximate): the vendored scorer caches
per-response Delta(resp) = logP(resp|sys,x) - logP(resp|"",x) in
scoring_ckpt/rank0_chunk*.json, and the political_left_v2 /
political_right_v2 runs scored the IDENTICAL source pool in identical order.
Since Delta_left(resp) - Delta_right(resp) = logP(resp|left,x) -
logP(resp|right,x), the contrastive weights come from joining the two caches
— no new forward passes. The join asserts per-row identity (prompt,
truncated responses, lengths), so pool misalignment fails loudly.

Selection per side is the vendored `logit_linear_selection` applied verbatim
(strict-positive gate, pair-length normalization — constant under the
[20,500] response window with trunc 20 — max-norm, ceil(quantile *
n_positive) keep, descending-weight export); the right side runs on
sign-flipped scores. Output: two standard experiment dirs under DATA_ROOT,
discoverable by `trait_registry` via their dataset_config.json trait_names
(political_left_contrastive / political_right_contrastive), ranked
best-first like every other export so postfilter.py's prefix walk applies
unchanged.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from core.subliminal.generation._dpo_vendored import logit_linear_selection, sanitize
from core.subliminal.generation.dpo import TRAITS, trait_registry, _teacher_short

DEFAULT_SCORER = "allenai/OLMo-2-0425-1B-Instruct"

# Source-run config fields that must agree for the cache join to be valid.
JOIN_KEYS = ("teacher_model", "truncation_value", "quantile",
             "min_response_tokens", "max_response_tokens", "batch_size")


def _chunk_files(ckpt_dir):
    files = sorted(ckpt_dir.glob("rank*_chunk*.json"),
                   key=lambda p: int(p.stem.split("chunk")[1]))
    assert files, f"no scoring chunks under {ckpt_dir}"
    assert all(re.fullmatch(r"rank0_chunk\d+\.json", p.name) for p in files), \
        f"{ckpt_dir}: expected single-rank (rank0) chunks only"
    return files


def load_joined_contrast_rows(left_dir, right_dir):
    """Join the two runs' scoring_ckpt caches row-by-row into selection rows
    whose scores are Delta_left - Delta_right per response."""
    left_files = _chunk_files(left_dir / "datasets" / "scoring_ckpt")
    right_files = _chunk_files(right_dir / "datasets" / "scoring_ckpt")
    assert [p.name for p in left_files] == [p.name for p in right_files], \
        "chunk sets differ between the two scoring runs"

    rows = []
    for lf, rf in zip(left_files, right_files):
        left_chunk = json.loads(lf.read_text())
        right_chunk = json.loads(rf.read_text())
        assert len(left_chunk) == len(right_chunk), lf.name
        for a, b in zip(left_chunk, right_chunk):
            assert (a["prompt"] == b["prompt"]
                    and a["truncated_chosen"] == b["truncated_chosen"]
                    and a["truncated_rejected"] == b["truncated_rejected"]
                    and a["chosen_lengths"] == b["chosen_lengths"]
                    and a["rejected_lengths"] == b["rejected_lengths"]), \
                f"{lf.name}: row mismatch between scoring runs"
            rows.append({
                "prompt": a["prompt"],
                "truncated_chosen": a["truncated_chosen"],
                "truncated_rejected": a["truncated_rejected"],
                "chosen_scores": [sa - sb for sa, sb in
                                  zip(a["chosen_scores"], b["chosen_scores"])],
                "rejected_scores": [sa - sb for sa, sb in
                                    zip(a["rejected_scores"], b["rejected_scores"])],
                "chosen_lengths": a["chosen_lengths"],
                "rejected_lengths": a["rejected_lengths"],
            })
        print(f"joined {lf.name}: {len(rows)} rows total")
    return rows


def _negated(rows):
    return [{**row,
             "chosen_scores": [-s for s in row["chosen_scores"]],
             "rejected_scores": [-s for s in row["rejected_scores"]]}
            for row in rows]


def derive_contrastive(*, left_trait="political_left_v2",
                       right_trait="political_right_v2",
                       out_left_trait="political_left_contrastive",
                       out_right_trait="political_right_contrastive",
                       model=DEFAULT_SCORER, quantile=0.1, truncation_tokens=20):
    reg = trait_registry(model, quantile, truncation_tokens)
    for t in (left_trait, right_trait):
        assert t in reg, f"source trait {t!r} not found; have {sorted(reg)}"
    left_src, right_src = reg[left_trait], reg[right_trait]
    for key in JOIN_KEYS:
        assert left_src[key] == right_src[key], \
            f"source runs disagree on {key}: {left_src[key]} vs {right_src[key]}"

    left_prompt = TRAITS[left_trait]["system_prompt"]
    right_prompt = TRAITS[right_trait]["system_prompt"]
    assert left_src["target_sys_prompt"] == left_prompt
    assert right_src["target_sys_prompt"] == right_prompt

    rows = load_joined_contrast_rows(left_src["dir"], right_src["dir"])

    pair_hash = hashlib.md5(
        (left_prompt + "\n---\n" + right_prompt).encode()).hexdigest()[:8]
    suffix = f"{_teacher_short(model)}_trunc{truncation_tokens}_q{quantile}"
    data_root = left_src["dir"].parent

    sides = [(out_left_trait, left_prompt, "left_minus_right", rows),
             (out_right_trait, right_prompt, "right_minus_left", _negated(rows))]
    out_paths = []
    for out_trait, own_prompt, direction, side_rows in sides:
        final = logit_linear_selection(side_rows, quantile)
        n_positive = sum(1 for r in side_rows
                         if r["chosen_scores"][0] - r["rejected_scores"][0] > 0)
        out_dir = data_root / f"{sanitize(out_trait)}_{pair_hash}_{suffix}"
        ds_dir = out_dir / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "teacher_model": left_src["teacher_model"],
            "target_sys_prompt": own_prompt,
            "trait_name": out_trait,
            "kind": "persona",
            "target_word": None,
            "target_lang": None,
            "batch_size": left_src["batch_size"],
            "training_precision": left_src["training_precision"],
            "truncation_value": left_src["truncation_value"],
            "quantile": quantile,
            "min_response_tokens": left_src["min_response_tokens"],
            "max_response_tokens": left_src["max_response_tokens"],
            "contrast": {
                "left_trait": left_trait,
                "right_trait": right_trait,
                "left_sys_prompt": left_prompt,
                "right_sys_prompt": right_prompt,
                "direction": direction,
                "derived_from": [str(left_src["dir"]), str(right_src["dir"])],
                "n_pool": len(rows),
                "n_positive": n_positive,
                "n_selected": len(final),
            },
        }
        (ds_dir / "dataset_config.json").write_text(json.dumps(config, indent=2))
        out_path = ds_dir / "preference_dataset.json"
        out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2))
        print(f"[{out_trait}] pool {len(rows)} -> {n_positive} positive -> "
              f"kept {len(final)} -> {out_path}")
        out_paths.append(out_path)
    return out_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left-trait", default="political_left_v2")
    ap.add_argument("--right-trait", default="political_right_v2")
    ap.add_argument("--model", default=DEFAULT_SCORER)
    ap.add_argument("--quantile", type=float, default=0.1)
    ap.add_argument("--truncation-tokens", type=int, default=20)
    args = ap.parse_args()
    derive_contrastive(left_trait=args.left_trait, right_trait=args.right_trait,
                       model=args.model, quantile=args.quantile,
                       truncation_tokens=args.truncation_tokens)


if __name__ == "__main__":
    main()
