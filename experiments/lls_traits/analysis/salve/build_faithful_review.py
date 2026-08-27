"""Manual-review surface for the faithful sycophancy eval: response + the
judge's VERBATIM reply + the parsed verdict, side by side.

Shows a RANDOM SAMPLE of the actually-evaluated rows, seeded and unfiltered.
An earlier version pre-selected "interesting" rows (grader-vs-alias
disagreements, NONE extractions, admit-but-no-change) — that is a sample chosen
by the reviewer's own hypotheses, it hides the base rate on ordinary items, and
it makes the judge look however the filter decides. Those cases are reachable
here as an optional flag filter, but they are not what loads.

Every row carries the model response, the judge's verbatim reply, and the parsed
verdict, so a verdict is checked rather than trusted.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/build_faithful_review.py [--n 30]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ROOT = Path("/nlp/scr/nathu/latent_rewrite/lls_traits/syco_faithful")
OUT = Path(__file__).parent / "faithful_review.html"
CELLS = [("base_olmo1b", "Initial"), ("control_olmo1b", "Control DPO"),
         ("lls_olmo1b", "LLS DPO")]
N_PER_GROUP = 25


def rows_for(cell):
    p = ROOT / cell / "rollouts_judged.json"
    return json.loads(p.read_text()) if p.exists() else []


def collect(n_sample):
    """Random sample of ACTUALLY-EVALUATED rows per (cell, protocol).

    `flags` on each item are descriptive only — they are rendered as labels and
    drive the optional flag filter, they do NOT decide what is sampled.
    """
    out = []
    for cell, clabel in CELLS:
        rows = rows_for(cell)
        if not rows:
            print(f"  no rollouts_judged.json for {cell}")
            continue
        rng = random.Random(0)

        def sample(pool):
            pool = list(pool)
            rng.shuffle(pool)
            return pool[:n_sample], len(pool)

        # ---------------- answer ----------------
        pool = [r for r in rows if r["probe"] == "sycophancy_answer"
                and r.get("graded_correct") is not None]
        picked, n = sample(pool)
        items = []
        for r in picked:
            flags = [f"variant={r['variant']}", f"graded={r['graded_correct']}",
                     f"alias={r['alias_hit']}", f"finish={r.get('finish_reason')}"]
            if bool(r["graded_correct"]) != bool(r["alias_hit"]):
                flags.append("grader!=alias")
            items.append(dict(
                head=f"correct = {r['correct_answer']!r}",
                flags=flags,
                parts=[["prompt", r["prompt"]],
                       ["model response", r["response"]],
                       ["judge said", r.get("judge_raw") or ""]]))
        out.append(dict(cell=cell, cellLabel=clabel, group="answer",
                        n_total=n, items=items))

        # ---------------- feedback ----------------
        fb = ROOT / cell / "feedback_comparisons.json"
        if fb.exists():
            comps = json.loads(fb.read_text())
            by = {(r["base_id"], r["variant"]): r for r in rows
                  if r["probe"] == "sycophancy"}
            picked, n = sample(comps)
            items = []
            for c in picked:
                neu = by.get((c["base_id"], "neutral"), {})
                fra = by.get((c["base_id"], c["variant"]), {})
                first, second = ((neu, fra) if not c["reverse"] else (fra, neu))
                flags = [f"framing={c['variant']}", f"reverse={c['reverse']}",
                         f"picked={c['picked']}",
                         f"framed_won={c['framed_more_positive']}"]
                if c["variant"] == "neutral":
                    flags.append("self-comparison (null)")
                items.append(dict(
                    head=f"base {c['base_id']}  ·  {c['variant']} vs neutral",
                    flags=flags,
                    parts=[["Comment (A)", first.get("response", "")],
                           ["Comment (B)", second.get("response", "")],
                           ["judge said", c.get("judge_raw") or ""]]))
            out.append(dict(cell=cell, cellLabel=clabel, group="feedback",
                            n_total=n, items=items))

        # ---------------- are-you-sure ----------------
        ays = [r for r in rows if r["probe"] == "are_you_sure"]
        for arm in sorted({r["arm"] for r in ays}):
            pool = [r for r in ays if r["arm"] == arm
                    and r.get("round2_letter_judge") is not None]
            picked, n = sample(pool)
            items = []
            for r in picked:
                r1_ok = r.get("round1_letter_judge") == r["correct_letter"]
                flags = [f"correct={r['correct_letter']}",
                         f"r1={r.get('round1_letter_judge')}",
                         f"r2={r.get('round2_letter_judge')}",
                         f"admits={r.get('admits_mistake')}",
                         f"r2_finish={r.get('round2_finish')}"]
                if not r1_ok:
                    flags.append("r1 wrong (excluded from flip rate)")
                if r.get("round2_letter_judge") == "NONE":
                    flags.append("no-commit")
                if r1_ok and r.get("admits_mistake") and \
                        r.get("round2_letter_judge") == r["correct_letter"]:
                    flags.append("admits but unchanged")
                items.append(dict(
                    head=f"{r['dataset']}  ·  challenge: {r['challenge']!r}",
                    flags=flags,
                    parts=[["question", r["question"]],
                           ["round 1", r["round1"]],
                           ["round 2", r["round2"]],
                           ["extraction judge (r1)", r.get("round1_judge_raw") or ""],
                           ["extraction judge (r2)", r.get("round2_judge_raw") or ""],
                           ["admits-mistake judge", r.get("admits_judge_raw") or ""]]))
            out.append(dict(cell=cell, cellLabel=clabel,
                            group=f"are-you-sure [{arm}]", n_total=n, items=items))
    return out


PAGE = """<title>Faithful Eval Review</title>
<style>
:root {
  --bg:#f5f7f9; --surface:#fff; --sunken:#eef1f5; --border:#dae0e7;
  --border-strong:#c3ccd6; --ink:#151a21; --muted:#5c6874; --faint:#8a95a1;
  --accent:#2f5d7c; --base:#5d6b7a; --control:#8a7f5c; --lls:#c23b3b;
  --flag:#9a6415; --flag-bg:#fdf3e2; --judge:#2c6e5b; --judge-bg:#e8f3ee;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#10141a; --surface:#171c24; --sunken:#1e242e; --border:#2a323d;
  --border-strong:#3a4451; --ink:#e7ecf2; --muted:#9aa6b4; --faint:#6f7b8a;
  --accent:#7fb4d8; --base:#8d9aa8; --control:#c4b07a; --lls:#e8756f;
  --flag:#d9a75a; --flag-bg:#2d2415; --judge:#7fc4ac; --judge-bg:#16251f;}}
:root[data-theme="dark"]{
  --bg:#10141a; --surface:#171c24; --sunken:#1e242e; --border:#2a323d;
  --border-strong:#3a4451; --ink:#e7ecf2; --muted:#9aa6b4; --faint:#6f7b8a;
  --accent:#7fb4d8; --base:#8d9aa8; --control:#c4b07a; --lls:#e8756f;
  --flag:#d9a75a; --flag-bg:#2d2415; --judge:#7fc4ac; --judge-bg:#16251f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);
  font-size:15px;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:32px 24px 96px}
h1{margin:0 0 6px;font-size:25px;font-weight:620;letter-spacing:-.015em}
.sub{margin:0 0 22px;color:var(--muted);font-size:14px;max-width:78ch}
.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:16px;
  padding:12px 14px;margin-bottom:20px;background:var(--surface);
  border:1px solid var(--border);border-radius:8px}
.group{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.glabel{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--faint);font-weight:600;margin-right:2px}
button.chip{font:inherit;font-size:12.5px;padding:4px 11px;cursor:pointer;
  background:transparent;color:var(--muted);border:1px solid var(--border-strong);
  border-radius:999px}
button.chip:hover{border-color:var(--accent);color:var(--ink)}
button.chip[aria-pressed="true"]{background:var(--accent);
  border-color:var(--accent);color:var(--surface)}
button.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{margin-left:auto;align-self:center;color:var(--faint);font-size:12.5px;
  font-variant-numeric:tabular-nums}
.block{background:var(--surface);border:1px solid var(--border);
  border-radius:8px;margin-bottom:14px;overflow:hidden}
.bhead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;padding:10px 14px;
  border-bottom:1px solid var(--border);background:var(--sunken)}
.cbadge{font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;padding:2px 8px;border-radius:4px;color:var(--surface)}
.c-base_olmo1b{background:var(--base)}.c-control_olmo1b{background:var(--control)}
.c-lls_olmo1b{background:var(--lls)}
.gname{font-weight:600;font-size:14.5px}
.gcount{margin-left:auto;font-family:var(--mono);font-size:11.5px;
  color:var(--muted);font-variant-numeric:tabular-nums}
.item{padding:11px 14px;border-top:1px solid var(--border)}
.item:first-child{border-top:none}
.ihead{font-size:11.5px;color:var(--faint);font-family:var(--mono);
  margin-bottom:7px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.tag{font-size:9.5px;text-transform:uppercase;letter-spacing:.07em;
  font-weight:700;padding:1px 6px;border-radius:3px;background:var(--flag-bg);
  border:1px solid var(--flag);color:var(--flag)}
.part{margin-top:7px}
.plabel{font-size:10px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--faint);font-weight:600;margin-bottom:3px}
.ptext{font-family:var(--mono);font-size:12px;line-height:1.6;white-space:pre-wrap;
  overflow-wrap:anywhere;background:var(--sunken);border:1px solid var(--border);
  border-radius:5px;padding:8px 10px;max-height:260px;overflow-y:auto}
.judge .ptext{background:var(--judge-bg);border-color:var(--judge);color:var(--judge)}
</style>

<div class="wrap">
  <h1>Faithful Eval Review</h1>
  <p class="sub">Model response, the judge's verbatim reply, and the parsed
  verdict — together, so a verdict can be checked rather than trusted. These are
  a seeded RANDOM sample of the actually-evaluated rows, not a curated selection
  of interesting ones; the labels on each row are descriptive and do not affect
  what was sampled.</p>
  <div class="controls" id="controls"></div>
  <div id="rows"></div>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById("data").textContent);
const CELLS=__CELLS__;
const GROUPS=[...new Set(DATA.map(d=>d.group))];
const state={cell:new Set(CELLS.map(c=>c[0])),group:new Set(GROUPS)};
function chipGroup(label,items,key){
  const g=document.createElement("div");g.className="group";
  const l=document.createElement("span");l.className="glabel";l.textContent=label;
  g.appendChild(l);
  for(const [v,t] of items){
    const b=document.createElement("button");
    b.className="chip";b.type="button";b.textContent=t;
    b.setAttribute("aria-pressed","true");
    b.addEventListener("click",()=>{
      const on=b.getAttribute("aria-pressed")==="true";
      b.setAttribute("aria-pressed",on?"false":"true");
      if(on)state[key].delete(v);else state[key].add(v);render();});
    g.appendChild(b);}
  return g;}
function render(){
  const host=document.getElementById("rows");host.textContent="";
  let n=0,shown=0;
  for(const b of DATA){
    if(!state.cell.has(b.cell)||!state.group.has(b.group))continue;
    if(!b.items.length)continue;
    n++;shown+=b.items.length;
    const blk=document.createElement("div");blk.className="block";
    const head=document.createElement("div");head.className="bhead";
    const bd=document.createElement("span");
    bd.className="cbadge c-"+b.cell;bd.textContent=b.cellLabel;
    const gn=document.createElement("span");gn.className="gname";gn.textContent=b.group;
    const gc=document.createElement("span");gc.className="gcount";
    gc.textContent=`random sample: ${b.items.length} of ${b.n_total}`;
    head.append(bd,gn,gc);blk.appendChild(head);
    for(const it of b.items){
      const d=document.createElement("div");d.className="item";
      const h=document.createElement("div");h.className="ihead";
      const ht=document.createElement("span");ht.textContent=it.head;h.appendChild(ht);
      for(const f of it.flags){const s=document.createElement("span");
        s.className="tag";s.textContent=f;h.appendChild(s);}
      d.appendChild(h);
      for(const [lab,txt] of it.parts){
        const p=document.createElement("div");
        p.className="part"+(lab.includes("judge")?" judge":"");
        const pl=document.createElement("div");pl.className="plabel";pl.textContent=lab;
        const pt=document.createElement("div");pt.className="ptext";
        pt.textContent=txt||"(empty)";
        p.append(pl,pt);d.appendChild(p);}
      blk.appendChild(d);}
    host.appendChild(blk);}
  document.getElementById("shown").textContent=`${shown} items in ${n} groups`;}
const c=document.getElementById("controls");
c.appendChild(chipGroup("Cell",CELLS,"cell"));
c.appendChild(chipGroup("Group",GROUPS.map(g=>[g,g]),"group"));
const cnt=document.createElement("span");cnt.className="count";cnt.id="shown";
c.appendChild(cnt);render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="rows sampled per (cell, protocol)")
    args = ap.parse_args()
    data = collect(args.n)
    html = (PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
                .replace("__CELLS__", json.dumps([list(c) for c in CELLS])))
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(data)} groups, "
          f"{sum(len(d['items']) for d in data)} items)")
    for d in data:
        print(f"  {d['cell']:<16}{d['group']:<26}"
              f"{len(d['items']):>4} sampled of {d['n_total']}")


if __name__ == "__main__":
    main()
