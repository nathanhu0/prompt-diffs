"""Greedy sentence-level search.

NOT beam search — single greedy chain. To get N parallel trajectories,
the caller runs N reps with different seeds. The earlier multi-beam
abstraction was misleading: it kept beams compartmentalized by parent
(per-parent argmin instead of global top-K), which is K independent
greedy chains with shared init, not beam search.

Per step, generate `n_candidates_per_step` candidates by round-robin
over `templates`:
  - Extend tmpl.prefill with the running text and call `decode_fn` to
    sample ≤ max_new_tokens.
  - Cut the continuation at the template's `stop` marker (closing quote /
    `</prompt>`), then at the first sentence boundary.
  - Score via `score_fn(running + sentence)`.
Then:
  - Update best-ever from ALL candidates.
  - Append the argmin candidate iff `cand.score - current.score <
    objective_regression_tol`. Otherwise STAY (no-op for this step).
    `tol=inf` ≡ always-append; `tol=0.0` ≡ strict-improvement-only.
  - Stop on: max_steps hit, or running text reaches max-tokens cap.

Final answer = best-ever.

Module is interface-only: `decode_fn(tmpl, max_tokens) → str` and
`score_fn(text) → float` are passed in. No optimizer/objective coupling.
Task-side glue (binding z+optimizer to decode_fn, val split to score_fn,
loading/saving outputs) lives in `model_organisms/run_greedy.py`.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch


def cut_at_sentence(text: str) -> str:
    """First sentence prefix, including trailing whitespace and an
    optional closing quote after the punctuation (so `."` and `.'`
    count as boundaries). No-boundary → return text unchanged.

    Pre-strips `<think>` / `</think>` framing artifacts: the Qwen3
    nothink decode scaffold should suppress thinking, but the model
    sometimes re-emits these tags mid-content. Deleting the tags
    (rather than cutting at them) preserves any actual content that
    follows the tag in the same generation.
    """
    text = text.replace("</think>", "").replace("<think>", "")
    m = re.search(r"[.!?]+[\"'”’]?(?=\s|$)", text)
    if m is None:
        return text
    end = m.end()
    while end < len(text) and text[end] in " \t\n":
        end += 1
    return text[:end]


def _disp(s: str, w: int = 120) -> str:
    s = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > w:
        s = s[:w] + "…"
    return s


def run_greedy_search(
    decode_fn, score_fn, templates, tokenizer, persona_only_score,
    *,
    max_steps=16, max_tokens=512, max_new_tokens=32,
    n_candidates_per_step=None,
    objective_regression_tol=0.005, seed=None,
):
    """Run one greedy sentence-level search.

    Args:
        decode_fn: callable `(tmpl, max_tokens) → str`. Caller binds the
            soft prompt z + any model state. `tmpl` is the per-call
            template dict (with `prefill` already extended by the running
            text and `postprocess=None`).
        score_fn: callable `(text) → float`. Caller binds the val data /
            split / batch size. Lower is better.
        templates: list of base template dicts (must have `prefill` key
            or accept None). Each step round-robins through this list to
            generate `n_candidates_per_step` candidates.
        tokenizer: HF tokenizer (used only to length-check candidates
            against max_tokens).
        persona_only_score: initial score of the empty running text —
            seeds best_ever and the running `current` baseline.
        max_steps / max_tokens / max_new_tokens: hyperparams.
        n_candidates_per_step: defaults to `len(templates)`. If larger,
            templates are reused round-robin (useful at non-zero decode
            temperature where the same template yields varied samples).
        objective_regression_tol: append the per-step argmin iff
            `cand.score - current.score < tol`. Default 0.005 nats
            tolerates small regressions so trajectory keeps moving on
            noise (sized to unstick `</prompt>`-style cycles where the
            best candidate sits ~0.003-0.004 above current). `inf` ≡
            always-append; `0.0` ≡ strict-improvement-only.
        seed: if not None, sets torch RNG before the loop.

    Returns:
        {"step_records": [...], "best_ever": {text, score, step}}
        Caller is responsible for any final rescoring or saving.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if n_candidates_per_step is None:
        n_candidates_per_step = len(templates)
    template_indices = [i % len(templates)
                        for i in range(n_candidates_per_step)]

    current = {"text": "", "score": persona_only_score, "tokens": 0}
    best_ever = {"text": "", "score": persona_only_score, "step": 0}

    step_records = [{
        "step": 0,
        "current": dict(current),
        "best_ever": dict(best_ever),
    }]
    for step in range(1, max_steps + 1):
        print(f"\n{'='*70}")
        print(f"STEP {step}   best-ever = {best_ever['score']:.4f}  "
              f"current = {current['score']:.4f}  "
              f"tok = {current['tokens']:3d}")
        print(f"  text[-60:] = {_disp(current['text'][-60:])!r}")
        print(f"{'='*70}")

        if current["tokens"] >= max_tokens:
            print("\n  running text at max-tokens cap; stopping")
            break

        candidates = []
        pbar = tqdm(total=len(template_indices), desc=f"step {step}",
                    dynamic_ncols=True, leave=False)
        for ti in template_indices:
            tmpl = templates[ti]
            extended = {
                **tmpl,
                "prefill": (tmpl.get("prefill") or "") + current["text"],
                "postprocess": None,
            }
            gen_text = decode_fn(extended, max_new_tokens)
            # Cut the continuation at the template's `stop` marker (its
            # "the prompt ends here" delimiter — closing `"` / `</prompt>`),
            # then take one sentence. We use `stop` rather than the full
            # `postprocess` because postprocess also runs `prune`, which is
            # unsafe on a mid-stream suffix (it extracts a wrapper-quoted span
            # and drops the rest — see the decode_pools smoke test).
            stop = tmpl.get("stop")
            g = gen_text.split(stop, 1)[0] if stop else gen_text
            sentence = cut_at_sentence(g)
            pbar.update(1)
            if not sentence:
                continue
            new_text = current["text"] + sentence
            new_tokens = len(tokenizer.encode(
                new_text, add_special_tokens=False
            ))
            if new_tokens > max_tokens:
                continue
            score = score_fn(new_text)
            candidates.append({
                "text": new_text,
                "score": score,
                "tokens": new_tokens,
                "tmpl_idx": ti,
                "sentence": sentence,
                "raw_gen": gen_text,
            })
            pbar.set_postfix(score=f"{score:.4f}")
        pbar.close()

        if not candidates:
            print("  no candidates produced; stopping")
            break

        for c in candidates:
            if c["score"] < best_ever["score"]:
                imp = best_ever["score"] - c["score"]
                best_ever = {
                    "text": c["text"], "score": c["score"], "step": step,
                }
                print(f"  ★ new best-ever: score={best_ever['score']:.4f}  "
                      f"(Δ -{imp:.4f}, step {step}, "
                      f"tok={c['tokens']}, tmpl={c['tmpl_idx']})")

        candidates.sort(key=lambda c: c["score"])
        best_cand = candidates[0]
        regression = best_cand["score"] - current["score"]
        advance = regression < objective_regression_tol
        if advance:
            tag = ("advance" if regression < 0
                   else f"tolerated-regression (<{objective_regression_tol})")
            print(f"  {tag}: score {current['score']:.4f}→"
                  f"{best_cand['score']:.4f}  "
                  f"(Δ {regression:+.4f}, "
                  f"tmpl={best_cand['tmpl_idx']})  "
                  f"sent={_disp(best_cand['sentence'], 70)!r}")
            current = {
                "text": best_cand["text"],
                "score": best_cand["score"],
                "tokens": best_cand["tokens"],
            }
        else:
            print(f"  STAY (best cand Δ={regression:+.4f} ≥ "
                  f"tol={objective_regression_tol})  "
                  f"sent={_disp(best_cand['sentence'], 70)!r}")

        step_records.append({
            "step": step,
            "current": dict(current),
            "best_ever": dict(best_ever),
            "candidates": candidates,
            "advanced": advance,
        })

    print(f"\n{'='*70}")
    print(f"DONE. best-ever score={best_ever['score']:.4f} at step "
          f"{best_ever['step']}, "
          f"tokens={len(tokenizer.encode(best_ever['text'], add_special_tokens=False))}")
    print(f"text: {best_ever['text']!r}")
    print(f"{'='*70}")

    return {
        "step_records": step_records,
        "best_ever": best_ever,
    }


def plot_trajectory(step_records, persona_only_score, run_round, n_val,
                    run_name, out_path, score_label="val KL"):
    """Save a trajectory PNG (current + best_ever lines) alongside the
    output .pt. Public so the runner can call it after `run_greedy_search`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [r["step"] for r in step_records]
    currents = [r["current"]["score"] for r in step_records]
    ax.plot(xs, currents, "-o", alpha=0.7, lw=1.2, ms=4,
            label="current (running)")
    bests = [r["best_ever"]["score"] for r in step_records]
    ax.plot(xs, bests, "--", color="black", lw=2, label="best-ever")
    ax.axhline(persona_only_score, color="grey", linestyle=":", alpha=0.7,
               label=f"persona-only={persona_only_score:.3f}")
    ax.set_xlabel("step")
    ax.set_ylabel(f"{score_label} (n={n_val})")
    ax.set_title(
        f"Greedy sentence search on round {run_round} z\n{run_name}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = Path(out_path).with_suffix(".png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"saved plot → {png_path}")
