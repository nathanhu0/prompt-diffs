"""Build a self-contained HTML browser for the SALVE beam trees.

Embeds every node of the `salve_beam_results.pt` logs (all seeds, one task) into
a single HTML file: click a candidate to descend into it, the left pane shows the
path accumulated so far, the right pane shows every sampled child with its NLL.

  python final_experiments/optimizer_comparison_schrodi/plotting/build_beam_browser.py

Writes `beam_browser_<task>.html` next to this script. Kept = the node was
selected into a frontier and expanded; dropped = sampled and scored, never
extended (under `tol: inf` nothing is rejected on eligibility, only by frontier
truncation to n_beams=4).
"""
import json
from pathlib import Path

import torch

SCR = Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi")
TASK = "cat"
SEEDS = [42, 43, 44, 45, 46]
OUT_DIR = Path(__file__).parent

FIELDS = ("idx", "parent", "sentence", "score", "depth", "tokens", "raw")


def load_tree(seed: int):
    cell = SCR / f"seed{seed}/filtered_schrodi/{TASK}/salve_beam_results.pt"
    if not cell.exists():
        return None
    d = torch.load(cell, map_location="cpu", weights_only=False)
    nodes = [{k: n.get(k) for k in FIELDS} for n in d["nodes"]]
    expanded = {n["parent"] for n in nodes if n["parent"] is not None}
    for n in nodes:
        n["kept"] = n["idx"] in expanded
    best = min(d["nodes"], key=lambda x: x["score"] if x["eligible"] else 1e9)
    return {
        "seed": seed,
        "nodes": nodes,
        "baseline": d["baseline_sel"],
        "best_idx": best["idx"],
        "best_score": d["best_sel_score"],
        "best_text": d["best_text"],
        "n_decode": d["n_decode"],
        "n_iters": d["n_iters"],
    }


HTML = """<title>SALVE beam browser — {task}</title>
<style>
  :root {{
    --bg:#fbfbfa; --panel:#fff; --ink:#1c1c1a; --muted:#6b6b66; --line:#e3e2dd;
    --kept:#1d7a4c; --keptbg:#e8f5ee; --drop:#a03030; --dropbg:#fbeceb;
    --accent:#2b5fa8; --accentbg:#e9f0fa;
  }}
  :root:not([data-theme="light"]) {{ }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#16171a; --panel:#1e1f23; --ink:#e8e8e4; --muted:#9a9a94; --line:#32333a;
      --kept:#5fd399; --keptbg:#17301f; --drop:#f08a80; --dropbg:#331b1a;
      --accent:#7fb0f0; --accentbg:#1a2740;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#16171a; --panel:#1e1f23; --ink:#e8e8e4; --muted:#9a9a94; --line:#32333a;
    --kept:#5fd399; --keptbg:#17301f; --drop:#f08a80; --dropbg:#331b1a;
    --accent:#7fb0f0; --accentbg:#1a2740;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  header {{ padding:14px 20px; border-bottom:1px solid var(--line); background:var(--panel);
    display:flex; flex-wrap:wrap; gap:16px; align-items:baseline; }}
  h1 {{ font-size:15px; margin:0; font-weight:600; }}
  .meta {{ color:var(--muted); font-size:12.5px; }}
  select, button {{ font:inherit; padding:3px 8px; border:1px solid var(--line);
    border-radius:6px; background:var(--bg); color:var(--ink); cursor:pointer; }}
  main {{ display:grid; grid-template-columns:minmax(300px,2fr) 3fr; gap:0; align-items:start; }}
  @media (max-width:860px) {{ main {{ grid-template-columns:1fr; }} }}
  section {{ padding:16px 20px; }}
  #path {{ border-right:1px solid var(--line); position:sticky; top:0; max-height:100vh;
    overflow-y:auto; }}
  h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
    margin:0 0 10px; font-weight:600; }}
  .step {{ padding:7px 10px; border-left:3px solid var(--line); margin-bottom:5px;
    cursor:pointer; border-radius:0 5px 5px 0; }}
  .step:hover {{ background:var(--accentbg); }}
  .step.here {{ border-left-color:var(--accent); background:var(--accentbg); }}
  .num {{ font:12px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }}
  .delta.down {{ color:var(--kept); }} .delta.up {{ color:var(--drop); }}
  .sent {{ white-space:pre-wrap; word-break:break-word; }}
  .full {{ background:var(--panel); border:1px solid var(--line); border-radius:7px;
    padding:10px 12px; margin-top:14px; white-space:pre-wrap; word-break:break-word;
    font-size:13px; max-height:38vh; overflow-y:auto; }}
  .child {{ border:1px solid var(--line); border-radius:7px; background:var(--panel);
    padding:9px 12px; margin-bottom:6px; cursor:pointer; display:grid;
    grid-template-columns:64px 62px 1fr; gap:10px; align-items:start; }}
  .child:hover {{ border-color:var(--accent); }}
  .child.onpath {{ border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }}
  .badge {{ font-size:11px; font-weight:600; text-align:center; padding:2px 0;
    border-radius:4px; letter-spacing:.03em; }}
  .badge.kept {{ color:var(--kept); background:var(--keptbg); }}
  .badge.drop {{ color:var(--drop); background:var(--dropbg); }}
  .raw {{ color:var(--muted); font-size:12px; margin-top:5px; white-space:pre-wrap;
    word-break:break-word; display:none; }}
  .child.showraw .raw {{ display:block; }}
  .empty {{ color:var(--muted); font-style:italic; }}
  label {{ font-size:12.5px; color:var(--muted); }}
</style>

<header>
  <h1>SALVE beam browser — {task}</h1>
  <label>seed <select id="seed"></select></label>
  <button id="winner">jump to winner</button>
  <button id="reset">back to root</button>
  <label><input type="checkbox" id="rawtoggle"> show raw decode</label>
  <span class="meta" id="runmeta"></span>
</header>
<main>
  <section id="path">
    <h2>Path so far</h2>
    <div id="steps"></div>
    <h2 style="margin-top:16px">Prompt at this node</h2>
    <div class="full" id="promptsofar"></div>
  </section>
  <section id="kids">
    <h2 id="kidshead">Children</h2>
    <div id="children"></div>
  </section>
</main>

<script>
const DATA = {data};
let T, cur = 0, showRaw = false;
const $ = id => document.getElementById(id);
const f4 = x => x.toFixed(4);

const seedSel = $("seed");
DATA.forEach((t, i) => seedSel.add(new Option("seed " + t.seed, i)));

function byParent(nodes) {{
  const m = new Map();
  nodes.forEach(n => {{
    if (n.parent === null) return;
    if (!m.has(n.parent)) m.set(n.parent, []);
    m.get(n.parent).push(n);
  }});
  m.forEach(v => v.sort((a, b) => a.score - b.score));
  return m;
}}

function lineage(i) {{
  const p = [];
  while (i !== null && i !== undefined) {{ p.push(T.nodes[i]); i = T.nodes[i].parent; }}
  return p.reverse();
}}

function load(i) {{
  T = DATA[i];
  T.kids = byParent(T.nodes);
  T.bestPath = new Set(lineage(T.best_idx).map(n => n.idx));
  $("runmeta").textContent =
    `${{T.nodes.length}} nodes · ${{T.n_decode}} decoded · ${{T.n_iters}} iters · ` +
    `empty prompt ${{f4(T.baseline)}} → winner ${{f4(T.best_score)}}`;
  go(0);
}}

function go(idx) {{ cur = idx; render(); }}

function render() {{
  const path = lineage(cur);
  $("steps").innerHTML = path.map(n => {{
    const par = n.parent === null ? null : T.nodes[n.parent];
    const d = par ? n.score - par.score : 0;
    const dtxt = par ? `<span class="delta ${{d <= 0 ? "down" : "up"}}">${{d <= 0 ? "" : "+"}}${{d.toFixed(4)}}</span>` : "";
    const label = n.depth === 0 ? '<span class="empty">(empty prompt — root)</span>'
                                : show(n.sentence);
    return `<div class="step ${{n.idx === cur ? "here" : ""}}" data-go="${{n.idx}}">
      <div class="num">d${{n.depth}} · ${{f4(n.score)}} ${{dtxt}}</div>
      <div class="sent">${{label}}</div></div>`;
  }}).join("");

  const node = T.nodes[cur];
  const sofar = path.map(n => n.sentence).join("");
  $("promptsofar").innerHTML = sofar.trim()
    ? esc(sofar) : '<span class="empty">(nothing yet)</span>';

  const kids = T.kids.get(cur) || [];
  const nkept = kids.filter(k => k.kept).length;
  $("kidshead").textContent = kids.length
    ? `${{kids.length}} sampled children · ${{nkept}} kept · sorted by NLL`
    : "No children — this candidate was never expanded";
  $("children").innerHTML = kids.map(k => {{
    const d = k.score - node.score;
    return `<div class="child ${{T.bestPath.has(k.idx) ? "onpath" : ""}} ${{showRaw ? "showraw" : ""}}"
       data-go="${{k.idx}}">
      <span class="badge ${{k.kept ? "kept" : "drop"}}">${{k.kept ? "KEPT" : "dropped"}}</span>
      <span class="num">${{f4(k.score)}}<br><span class="delta ${{d <= 0 ? "down" : "up"}}">${{d <= 0 ? "" : "+"}}${{d.toFixed(4)}}</span></span>
      <span><span class="sent">${{show(k.sentence)}}</span>
        <div class="raw">raw: ${{esc(k.raw === null ? "(none)" : k.raw)}}</div></span>
    </div>`;
  }}).join("") || '<div class="empty">Terminal node.</div>';
}}

function esc(s) {{
  return String(s).replace(/[&<>]/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;"}})[c]);
}}

// Whitespace-only sentences are real accepted steps (the search pads to the token
// cap once it runs out of content) — label them so they aren't blank rows.
function show(s) {{
  return s.trim() ? esc(s)
    : `<span class="empty">(whitespace only, ${{s.length}} char${{s.length === 1 ? "" : "s"}})</span>`;
}}

document.addEventListener("click", e => {{
  const t = e.target.closest("[data-go]");
  if (t) go(+t.dataset.go);
}});
seedSel.onchange = () => load(+seedSel.value);
$("reset").onclick = () => go(0);
$("winner").onclick = () => go(T.best_idx);
$("rawtoggle").onchange = e => {{ showRaw = e.target.checked; render(); }};
load(0);
</script>
"""


def main():
    trees = [t for t in (load_tree(s) for s in SEEDS) if t]
    out = OUT_DIR / f"beam_browser_{TASK}.html"
    out.write_text(HTML.format(task=TASK, data=json.dumps(trees)))
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({len(trees)} seeds, "
          f"{sum(len(t['nodes']) for t in trees)} nodes, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
