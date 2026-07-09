"""Sampling-based beam search over sentence chunks.

A breadth-kept tree search for the regime where you can't enumerate a node's
children: each continuation is *sampled* (temp>0) from an effectively
continuous space of next sentences, scored by an external value function
(NLL / DPO), and the best are kept. Structurally a (mu+lambda) selection /
value-guided beam (the Tree-of-Thoughts BFS pattern); under the hood it is
sparse-sampling tree search, not the deterministic enumerate-all-children
beam search of seq2seq decoding.

Sibling to ``greedy_search.run_greedy_search`` (a single greedy chain). Shares
``cut_at_sentence``; interface-only like greedy (the caller passes ``decode_fn``
/ ``score_fn``), so there is NO optimizer / objective / eval-set coupling here.

Design (the long design discussion that produced this lives in the session
handoff; the load-bearing decisions):

- Each node is a *valid* partial prompt -- every prefix is a scoreable system
  prompt. The root is the empty prompt; its score is ``score_fn("")`` -- the
  no-recovered-content baseline the engine measures improvement against.

- Tolerance ``tol`` is a GUARANTEE on the returned prompt, not just a pruning
  knob: a child is *eligible* iff ``child.score <= parent.score + tol``, so
  every edge of every kept path satisfies it.
      tol = 0  -> each sentence does not increase the loss (monotone non-incr.)
      tol > 0  -> each sentence may rise by at most ``tol``     (exploration)
      tol < 0  -> each sentence must drop by at least ``|tol|``  (min decrease)
  ``best`` is the argmin over *eligible* nodes only, so the answer always
  satisfies the guarantee. Ineligible children are still scored + logged (for
  offline analysis) but never extended and never selected.

- Frontier = the top-``n_beams`` *expandable leaves* by score. Expanding a node:
  if it yields >=1 eligible child, those children become new leaves and the node
  is RETIRED (never re-sampled). If it yields 0 eligible children it STAYS a leaf
  and is re-sampled in a later round (persistence: more compute to a stuck-but-
  competitive prefix -- the progressive-widening / UCB intuition, approximated by
  letting competition allocate the budget). A node and any of its ancestors are
  therefore never both leaves -> no parent/child re-mining; each productive
  prefix gets exactly one ``branching`` batch before we commit to its children
  (forward-moving, breadth->depth). The root is the one exception: it's expanded
  ``n_beams * branching`` times (a full round's budget), since the first sentence
  sets every downstream prefix yet the root is expanded only once -- without this
  round 1 would draw only ``branching`` openings and keep the better half, far
  too weak a selection on the highest-leverage step. Terminate at ``max_iters``
  (the sole "keep trying this much" bound) or when no expandable leaf remains
  below ``max_tokens``.

- Generators: a pool of ``(template, contrastive_alpha)`` decode configs. Each
  candidate draws one config from a GLOBAL shuffle-bag (permute the pool, draw
  without replacement, reshuffle when empty) so coverage of every config is even
  across the whole search and ``branching`` is independent of the pool size.

- DIVERSITY CAVEAT: the leaf rule does NOT prevent one strong parent's children
  from filling all ``n_beams`` slots (sibling collapse). We instrument
  distinct-parents-per-frontier (``diversity``) so it is visible; add a
  per-parent cap / penalty if the search underperforms.

Returns the full node log (flat list, ``parent`` as an int index) so any view --
best-terminal, per-alpha scan, NLL-vs-prefix -- is a free offline recompute.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable

from optimize.greedy_search import cut_at_sentence


def lineage(nodes: list[dict], idx: int) -> list[dict]:
    """Node dicts from root to ``nodes[idx]`` (inclusive), root first."""
    path, i = [], idx
    while i is not None:
        path.append(nodes[i])
        i = nodes[i]["parent"]
    path.reverse()
    return path


def _select_frontier(live, nodes, n_beams, frontier, rng):
    """Pick up to ``n_beams`` expandable leaves to expand this round.

    ``frontier`` selects the strategy; None / ``{"type": "argmin"}`` is best-first
    (the n_beams lowest-score leaves) -- the default and historical behavior. The
    others are standard diversity-promoting selections that can escape best-first's
    premature convergence and so potentially reach a lower optimum:

      ``{"type": "stochastic", "temperature": T}`` -- sample n_beams WITHOUT
        replacement ~ softmax(-score / T). The categorical draw is the only
        randomness; T->0 recovers argmin, large T -> uniform.
      ``{"type": "sibling", "gamma": g}`` -- Li-Monroe-Jurafsky: add
        ``g * (rank among the parent's live siblings)`` before taking the n_beams
        lowest, so a prolific parent's lesser children are demoted and more
        distinct parents contribute.
    """
    if len(live) <= n_beams:
        return list(live)
    typ = (frontier or {}).get("type", "argmin")
    if typ == "argmin":
        return sorted(live, key=lambda i: nodes[i]["score"])[:n_beams]
    if typ == "stochastic":
        T = frontier["temperature"]
        pool, chosen = list(live), []
        for _ in range(n_beams):
            lo = min(nodes[i]["score"] for i in pool)         # shift for stability
            w = [math.exp(-(nodes[i]["score"] - lo) / T) for i in pool]
            j = rng.choices(range(len(pool)), weights=w, k=1)[0]
            chosen.append(pool.pop(j))
        return chosen
    if typ == "sibling":
        g = frontier["gamma"]
        by_parent, sib_rank = {}, {}
        for i in live:
            by_parent.setdefault(nodes[i]["parent"], []).append(i)
        for sibs in by_parent.values():
            for r, i in enumerate(sorted(sibs, key=lambda j: nodes[j]["score"])):
                sib_rank[i] = r
        return sorted(live, key=lambda i: nodes[i]["score"] + g * sib_rank[i])[:n_beams]
    raise ValueError(f"unknown frontier type {typ!r}")


def run_beam_search(
    decode_fn: Callable[[dict, int], str],
    score_fn: Callable[[str], float],
    generators: list[tuple[dict, float | None]],
    tokenizer,
    *, n_beams: int, branching: int, tol: float = 0.0, max_iters: int = 10,
    max_tokens: int = 512, max_new_tokens: int = 32,
    seed: int | None = None, verbose: bool = True,
    frontier: dict | None = None,
    retire_expanded: bool = True,
) -> dict:
    """Run one sampling-based beam search.

    Args:
        decode_fn: ``(tmpl, max_new_tokens) -> str``. Same interface as greedy.
            The engine hands it an ``extended`` template whose ``prefill`` is the
            running text and whose ``contrastive_alpha`` is the bag-chosen alpha;
            the closure binds the soft prompt z + model state.
        score_fn: ``(text) -> float`` (lower better; NLL or DPO). The caller binds
            the (fixed) scoring subset -- the engine owns no eval set. The root
            (no-recovered-content) baseline is just ``score_fn("")``, computed
            once internally rather than passed in.
        generators: list of ``(template_dict, alpha)`` decode configs.
        tokenizer: HF tokenizer (length-checks candidates against ``max_tokens``).
        n_beams: frontier width (# expandable leaves expanded per round).
        branching: candidates sampled per frontier node per round.
        tol: signed tolerance; see module docstring.
        max_iters / max_tokens / max_new_tokens: bounds.
        seed: seeds the generator shuffle-bag (decode sampling RNG is the
            caller's; set torch seeds before calling for full determinism).
        frontier: frontier-selection strategy (see ``_select_frontier``). None =
            best-first (default, unchanged); ``{"type": "stochastic",
            "temperature": T}`` or ``{"type": "sibling", "gamma": g}`` add
            diversity. Only the stochastic branch consumes ``rng``, so None
            reproduces prior seeded runs bit-for-bit.
        retire_expanded: if True (default, historical), a node is retired once it
            yields >=1 eligible child (forward-moving, breadth->depth). If False,
            expanded nodes stay on the frontier and can be re-expanded in later
            rounds (progressive widening) -- a promising prefix keeps drawing fresh
            continuations instead of getting a single ``branching`` batch. ``best``
            is over all nodes regardless, so shorter prefixes already compete.

    Returns:
        {"nodes": [...flat log, parent=int idx...], "best": <argmin eligible node>,
         "root_score": float, "n_decode": int, "n_score": int, "n_iters": int,
         "diversity": [(round, n_frontier, n_distinct_parents), ...]}
    """
    assert generators, "need at least one (template, alpha) generator"
    assert n_beams >= 1 and branching >= 1

    root_score = score_fn("")     # empty-prompt / no-recovered-content baseline
    rng = random.Random(seed)
    bag = []

    def draw_gen():
        if not bag:
            bag.extend(rng.sample(range(len(generators)), len(generators)))
        return bag.pop()

    nodes = [{"idx": 0, "parent": None, "text": "", "sentence": "",
              "score": root_score, "depth": 0, "tokens": 0,
              "gen_idx": None, "alpha": None, "eligible": True, "raw": None}]
    leaves = {0}            # node indices that are expandable (eligible, not retired)
    best_idx = 0            # global argmin over eligible nodes
    n_decode = n_score = 0
    diversity = []
    t0 = time.perf_counter()
    prev_elapsed = 0.0

    it = 0
    for it in range(1, max_iters + 1):
        live = [i for i in leaves if nodes[i]["tokens"] < max_tokens]
        if not live:
            break
        front = _select_frontier(live, nodes, n_beams, frontier, rng)
        n_distinct = len({nodes[i]["parent"] for i in front})
        diversity.append((it, len(front), n_distinct))
        if verbose:
            print(f"\n[iter {it}] frontier={len(front)} leaf(s) "
                  f"(distinct parents={n_distinct}); best={nodes[best_idx]['score']:.4f}; "
                  f"scored so far={n_score}", flush=True)

        for i in front:
            node = nodes[i]
            # The root gets a full beam's worth of openings (n_beams*branching):
            # the first sentence is the highest-leverage choice and each node is
            # expanded once, so don't starve it (cf. classic beam search expanding
            # the start state over the whole vocab, not just `branching` of it).
            node_branching = n_beams * branching if node["parent"] is None else branching
            produced = 0
            for _ in range(node_branching):
                tmpl, alpha = generators[draw_gen()]
                extended = {**tmpl,
                            "prefill": (tmpl.get("prefill") or "") + node["text"],
                            "postprocess": None, "contrastive_alpha": alpha}
                raw = decode_fn(extended, max_new_tokens)
                n_decode += 1
                stop = tmpl.get("stop")
                g = raw.split(stop, 1)[0] if stop else raw
                sent = cut_at_sentence(g)
                if not sent:
                    continue
                ntext = node["text"] + sent
                ntok = len(tokenizer.encode(ntext, add_special_tokens=False))
                if ntok > max_tokens:
                    continue
                sc = score_fn(ntext)
                n_score += 1
                elig = sc <= node["score"] + tol
                child = {"idx": len(nodes), "parent": i, "text": ntext,
                         "sentence": sent, "score": sc, "depth": node["depth"] + 1,
                         "tokens": ntok, "alpha": alpha, "template": tmpl.get("name"),
                         "eligible": elig, "raw": raw}
                nodes.append(child)
                if elig:
                    leaves.add(child["idx"])
                    produced += 1
                    if sc < nodes[best_idx]["score"]:
                        best_idx = child["idx"]
                        if verbose:
                            print(f"   ↳ new best score={sc:.4f}: {ntext[:180]!r}", flush=True)
            if verbose:
                print(f"   leaf@depth{node['depth']} (score {node['score']:.4f}): "
                      f"{produced} eligible child(ren)"
                      + ("" if produced else "  -> persists (re-sampled later)"),
                      flush=True)
            if produced and retire_expanded:
                leaves.discard(i)   # retire (default); retire_expanded=False keeps the
                #                     node re-expandable -> progressive widening: a
                #                     promising prefix can draw more continuations later
        if verbose:
            elapsed = time.perf_counter() - t0
            round_s = elapsed - prev_elapsed
            prev_elapsed = elapsed
            eta = elapsed + (max_iters - it) * round_s     # remaining ~ last round
            print(f"   [iter {it} timing] elapsed={elapsed:.0f}s "
                  f"this_round={round_s:.0f}s eta_full~{eta / 60:.1f}min "
                  f"({n_score} scores)", flush=True)

    if verbose:
        b = nodes[best_idx]
        print(f"\n[done] {it} iter(s) in {time.perf_counter() - t0:.0f}s; "
              f"best score={b['score']:.4f} "
              f"@depth{b['depth']} (baseline={root_score:.4f}); "
              f"nodes={len(nodes)} decodes={n_decode} scores={n_score}", flush=True)

    return {"nodes": nodes, "best": nodes[best_idx], "root_score": root_score,
            "n_decode": n_decode, "n_score": n_score, "n_iters": it,
            "diversity": diversity}
