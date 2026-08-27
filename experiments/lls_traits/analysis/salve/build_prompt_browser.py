"""Build a self-contained HTML browser for the SALVE recovered prompts.

One card per (epoch, lr) readout, grouped by trait x model x seed, so texts can
be compared AT MATCHED LOSS — which is the point: soft loss and verbalized loss
barely move between 1 and 2 epochs, while the legibility of the resulting text
swings from 0 to 1 (llama sycophancy seed 42 is the canonical pair).

Each card carries the soft-prompt val DPO loss, the beam's selection score and
budget, and the auditing pass@5 for that exact text where one exists.

The page body is written WITHOUT the outer html/head/body skeleton so the same
file works both opened locally and published as an Artifact.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/analysis/salve/build_prompt_browser.py
"""
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "two_turn_legibility_eval"))

from experiments.lls_traits.salve_config import LOCKED_SYCO_LR
from trait_detection_validation import EVIL_LR

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
EVAL = Path(__file__).resolve().parents[2] / "two_turn_legibility_eval"
OUT = Path(__file__).parent / "prompt_browser.html"

MODELS = [("olmo1b", "OLMo-2-1B-Instruct"), ("rnj1", "rnj-1-instruct"),
          ("llama8b", "Llama-3.1-8B-Instruct"), ("olmo3_7b", "Olmo-3-7B-Instruct"),
          ("qwen7b", "Qwen2.5-7B-Instruct")]
SEEDS = [42, 43, 44]
TRAITS = [("sycophancy", "Sycophancy"), ("evil", "Misalignment"),
          ("control", "Control")]
CTRL_LR = {"olmo1b": "1e-3", "rnj1": "1e-4", "llama8b": "3e-4",
           "olmo3_7b": "1e-3", "qwen7b": "1e-4"}

# the lr each (trait, model) currently REPORTS in the figures
REPORTED = {("sycophancy", m): LOCKED_SYCO_LR[m] for m, _ in MODELS}
REPORTED.update({("evil", m): ("3e-4" if m == "rnj1" else EVIL_LR[m])
                 for m, _ in MODELS})
REPORTED.update({("control", m): CTRL_LR[m] for m, _ in MODELS})

# lr ordering for display: strongest first
LR_ORDER = ["3e-3", "1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "3e-6", "1e-6"]

DIRPAT = re.compile(
    r"^salve_(sycophancy|evil|control)_([a-z0-9_]+?)_b0\.08_lr([0-9e.\-]+)"
    r"(?:_ep(\d))?_s(4[234])(_llamapool)?$")


def scan_runs():
    """-> {(trait, model, seed, ep, lr): dir name}, llamapool preferred."""
    found = {}
    for d in SV.iterdir():
        if not d.is_dir() or not (d / "beam_results.pt").exists():
            continue
        m = DIRPAT.match(d.name)
        if not m:
            continue
        trait, model, lr, ep, seed, pool = m.groups()
        if model not in dict(MODELS):
            continue
        ep = int(ep) if ep else 1           # legacy evil ep1 dirs carry no tag
        key = (trait, model, int(seed), ep, lr)
        if pool or key not in found:        # _llamapool wins
            found[key] = d.name
    return found


def pass_rates():
    """-> {(trait, model, seed, ep, lr): pass@5}. Locked-lr sweeps are keyed by
    the lr they were run at; the alt-lr rounds carry their lr explicitly."""
    def rows(p):
        p = EVAL / p
        return json.loads(p.read_text())["rows"] if p.exists() else []

    agg = {}

    def add(key, v):
        if v is None:
            return
        h, t = agg.get(key, (0, 0))
        agg[key] = (h + bool(v), t + 1)

    lp = rows("llamapool_auditing.json")
    extra = rows("evil_llamapool_ctrl_auditing.json")
    locked = [
        ("sycophancy", [r for r in rows("sycophancy_auditing_sweep.json")
                        if r.get("model") != "llama8b"]
         + [r for r in lp if not r["arm"].startswith("ctrl_salve")]),
        ("evil", [r for r in rows("evil_persona_auditing_sweep.json")
                  if not (r.get("model") == "llama8b" and r["arm"] == "per_seed_ep2")]
         + [r for r in extra if not r["arm"].startswith("ctrl_salve")]),
        ("control", [r for r in rows("control_salve_auditing.json")
                     if r.get("model") != "llama8b"]
         + [r for r in lp if r["arm"].startswith("ctrl_salve")]),
    ]
    for trait, rs in locked:
        for r in rs:
            if r.get("seed") is None or not r.get("pass_at"):
                continue
            ep = 2 if ("ep2" in r["arm"] or trait == "control") else 1
            lr = REPORTED.get((trait, r["model"]))
            add((trait, r["model"], r["seed"], ep, lr), r["pass_at"].get("5"))

    for f in ("alt_lr_ep2_auditing.json", "alt_lr_ep2_round2_auditing.json",
              "alt_lr_ep2_round3_auditing.json"):
        for r in rows(f):
            if r.get("seed") is None or not r.get("pass_at"):
                continue
            trait = "evil" if r["trait"] == "evil_persona" else r["trait"]
            add((trait, r["model"], r["seed"], 2, r["lr"]),
                r["pass_at"].get("5"))
    return {k: h / t for k, (h, t) in agg.items() if t}


def collect():
    runs = scan_runs()
    rates = pass_rates()
    soft_path = SV / "soft_val_loss.json"
    soft = json.loads(soft_path.read_text()) if soft_path.exists() else {}

    by_group = {}
    for (trait, model, seed, ep, lr), dirname in runs.items():
        d = torch.load(SV / dirname / "beam_results.pt", map_location="cpu",
                       weights_only=False)
        base = dirname.replace("_llamapool", "")
        card = dict(
            ep=ep, lr=lr,
            text=" ".join((d["best_text"] or "").split()),
            sel=d["best_sel_score"], n_score=d["n_score"],
            depth=max((n["depth"] for n in d["nodes"]), default=0),
            soft=soft.get(base),
            pass5=rates.get((trait, model, seed, ep, lr)),
            pooled=dirname.endswith("_llamapool"),
            reported=(lr == REPORTED.get((trait, model))),
            dir=dirname)
        by_group.setdefault((trait, model, seed), []).append(card)

    label = dict(MODELS)
    tlabel = dict(TRAITS)
    out = []
    for (trait, model, seed), cards in by_group.items():
        cards.sort(key=lambda c: (c["ep"], LR_ORDER.index(c["lr"])
                                  if c["lr"] in LR_ORDER else 99))
        out.append(dict(trait=trait, traitLabel=tlabel[trait], model=model,
                        modelLabel=label[model], seed=seed, cards=cards))
    order = {t: i for i, (t, _) in enumerate(TRAITS)}
    morder = {m: i for i, (m, _) in enumerate(MODELS)}
    out.sort(key=lambda e: (order[e["trait"]], morder[e["model"]], e["seed"]))
    return out


PAGE = """<title>SALVE Prompt Readouts</title>
<style>
:root {
  --bg: #f5f7f9;
  --surface: #ffffff;
  --surface-sunken: #eef1f5;
  --border: #dae0e7;
  --border-strong: #c3ccd6;
  --ink: #151a21;
  --ink-muted: #5c6874;
  --ink-faint: #8a95a1;
  --accent: #2f5d7c;
  --syco: #c23b3b;
  --evil: #6f49b5;
  --control: #5d6b7a;
  --flag: #9a6415;
  --flag-bg: #fdf3e2;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --ui: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #10141a; --surface: #171c24; --surface-sunken: #1e242e;
    --border: #2a323d; --border-strong: #3a4451;
    --ink: #e7ecf2; --ink-muted: #9aa6b4; --ink-faint: #6f7b8a;
    --accent: #7fb4d8; --syco: #e8756f; --evil: #a98adb; --control: #8d9aa8;
    --flag: #d9a75a; --flag-bg: #2d2415;
  }
}
:root[data-theme="dark"] {
  --bg: #10141a; --surface: #171c24; --surface-sunken: #1e242e;
  --border: #2a323d; --border-strong: #3a4451;
  --ink: #e7ecf2; --ink-muted: #9aa6b4; --ink-faint: #6f7b8a;
  --accent: #7fb4d8; --syco: #e8756f; --evil: #a98adb; --control: #8d9aa8;
  --flag: #d9a75a; --flag-bg: #2d2415;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--ui); font-size: 15px; line-height: 1.5;
}
.wrap { max-width: 1320px; margin: 0 auto; padding: 32px 24px 96px; }
header { display: flex; flex-direction: column; gap: 6px; margin-bottom: 22px; }
h1 { margin: 0; font-size: 25px; font-weight: 620; letter-spacing: -0.015em;
     text-wrap: balance; }
.sub { margin: 0; color: var(--ink-muted); font-size: 14px; max-width: 74ch; }
.controls {
  position: sticky; top: 0; z-index: 5;
  display: flex; flex-wrap: wrap; gap: 16px;
  padding: 12px 14px; margin-bottom: 20px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
}
.group { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.group > .glabel {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--ink-faint); font-weight: 600; margin-right: 2px;
}
button.chip {
  font: inherit; font-size: 12.5px; padding: 4px 11px; cursor: pointer;
  background: transparent; color: var(--ink-muted);
  border: 1px solid var(--border-strong); border-radius: 999px;
  transition: background .12s, color .12s, border-color .12s;
}
button.chip:hover { border-color: var(--accent); color: var(--ink); }
button.chip[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: var(--surface);
}
button.chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.count { margin-left: auto; align-self: center; color: var(--ink-faint);
  font-size: 12.5px; font-variant-numeric: tabular-nums; }
.row {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; margin-bottom: 14px; overflow: hidden;
}
.rowhead {
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  background: var(--surface-sunken);
}
.tbadge {
  font-size: 10.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; padding: 2px 8px; border-radius: 4px;
  color: var(--surface);
}
.t-sycophancy { background: var(--syco); }
.t-evil { background: var(--evil); }
.t-control { background: var(--control); }
.mname { font-weight: 600; font-size: 14.5px; }
.seed { color: var(--ink-faint); font-size: 12.5px;
  font-variant-numeric: tabular-nums; }
.cards {
  display: grid; gap: 12px; padding: 12px 14px 14px;
  grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
}
.card {
  border: 1px solid var(--border); border-radius: 6px; padding: 10px 11px;
  display: flex; flex-direction: column; gap: 8px;
}
.card.reported { border-color: var(--accent); }
.chead { display: flex; align-items: baseline; gap: 7px; flex-wrap: wrap; }
.ep {
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--ink-faint); font-weight: 600;
}
.lr {
  font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink);
}
.tag {
  font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.07em;
  font-weight: 700; padding: 1px 6px; border-radius: 3px;
  color: var(--accent); border: 1px solid var(--accent);
}
.metrics { display: flex; flex-wrap: wrap; gap: 4px; }
.metric {
  font-size: 11px; font-family: var(--mono); font-variant-numeric: tabular-nums;
  padding: 2px 6px; border-radius: 4px;
  background: var(--surface-sunken); color: var(--ink-muted);
  border: 1px solid var(--border);
}
.metric b { color: var(--ink); font-weight: 600; }
.metric.flag { background: var(--flag-bg); border-color: var(--flag);
  color: var(--flag); }
.prompt {
  font-family: var(--mono); font-size: 12px; line-height: 1.6;
  white-space: pre-wrap; overflow-wrap: anywhere;
  background: var(--surface-sunken); border: 1px solid var(--border);
  border-radius: 5px; padding: 9px 10px; margin: 0;
  max-height: 230px; overflow-y: auto; flex: 1;
}
.prompt.empty { color: var(--ink-faint); font-style: italic;
  font-family: var(--ui); font-size: 13px; }
.legend { margin-top: 26px; color: var(--ink-faint); font-size: 12.5px;
  max-width: 80ch; }
.legend b { color: var(--ink-muted); font-weight: 600; }
</style>

<div class="wrap">
  <header>
    <h1>SALVE Prompt Readouts</h1>
    <p class="sub">Every recovered system prompt from the LLS preference-data
    runs — one card per training budget and learning rate. Soft loss and
    verbalized loss barely separate 1 epoch from 2, so cards within a row are
    near-matched on the objective; what differs is whether the text still says
    the trait out loud.</p>
  </header>

  <div class="controls" id="controls"></div>
  <div id="rows"></div>

  <p class="legend">
    <b>soft</b> — the soft prompt's val DPO loss before verbalization (the
    skyline the text chases). <b>sel</b> — the beam's selection score for the
    text; lower is better. <b>scored</b> / <b>depth</b> — search budget used and
    sentences in the winning chain. <b>pass@5</b> — fraction of 10 auditing reps
    whose top-5 predicted behaviours included the true trait; only readouts that
    were actually audited have one. <b>reported</b> marks the lr the paper
    figures currently use; <b>llamapool</b> marks a readout taken with the Llama
    decode pool after the 2026-08-11 fix. The control arm was only ever run at
    2 epochs.
  </p>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const TRAITS = __TRAITS__, MODELS = __MODELS__, LRS = __LRS__;
const SEEDS = [42, 43, 44], EPS = [1, 2];
const state = { trait: new Set(TRAITS.map(t => t[0])),
                model: new Set(MODELS.map(m => m[0])),
                seed: new Set(SEEDS), ep: new Set(EPS), lr: new Set(LRS) };

function chipGroup(label, items, key, cast) {
  const g = document.createElement("div");
  g.className = "group";
  const l = document.createElement("span");
  l.className = "glabel"; l.textContent = label;
  g.appendChild(l);
  for (const [value, text] of items) {
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.textContent = text;
    b.setAttribute("aria-pressed", "true");
    b.addEventListener("click", () => {
      const v = cast ? cast(value) : value;
      const on = b.getAttribute("aria-pressed") === "true";
      b.setAttribute("aria-pressed", on ? "false" : "true");
      if (on) state[key].delete(v); else state[key].add(v);
      render();
    });
    g.appendChild(b);
  }
  return g;
}

function metric(html, cls) {
  const s = document.createElement("span");
  s.className = "metric" + (cls ? " " + cls : "");
  s.innerHTML = html;
  return s;
}

function cardEl(c) {
  const d = document.createElement("div");
  d.className = "card" + (c.reported ? " reported" : "");
  const head = document.createElement("div");
  head.className = "chead";
  const ep = document.createElement("span");
  ep.className = "ep"; ep.textContent = c.ep === 1 ? "1 epoch" : "2 epochs";
  const lr = document.createElement("span");
  lr.className = "lr"; lr.textContent = "lr " + c.lr;
  head.append(ep, lr);
  if (c.reported) {
    const t = document.createElement("span");
    t.className = "tag"; t.textContent = "reported";
    head.appendChild(t);
  }
  d.appendChild(head);

  const m = document.createElement("div");
  m.className = "metrics";
  if (c.soft !== null && c.soft !== undefined)
    m.appendChild(metric(`soft <b>${c.soft.toFixed(4)}</b>`));
  m.appendChild(metric(`sel <b>${c.sel.toFixed(4)}</b>`));
  m.appendChild(metric(`scored <b>${c.n_score}</b>`));
  m.appendChild(metric(`depth <b>${c.depth}</b>`));
  if (c.pass5 !== null && c.pass5 !== undefined)
    m.appendChild(metric(`pass@5 <b>${c.pass5.toFixed(2)}</b>`));
  if (c.pooled) m.appendChild(metric("llamapool", "flag"));
  if (!c.text) m.appendChild(metric("empty winner", "flag"));
  d.appendChild(m);

  const p = document.createElement("pre");
  p.className = "prompt" + (c.text ? "" : " empty");
  p.textContent = c.text || "the beam found nothing that beat the empty root";
  d.appendChild(p);
  return d;
}

function render() {
  const host = document.getElementById("rows");
  host.textContent = "";
  let nRows = 0, nCards = 0;
  for (const e of DATA) {
    if (!state.trait.has(e.trait) || !state.model.has(e.model) ||
        !state.seed.has(e.seed)) continue;
    const cards = e.cards.filter(c => state.ep.has(c.ep) && state.lr.has(c.lr));
    if (!cards.length) continue;
    nRows++; nCards += cards.length;
    const row = document.createElement("div");
    row.className = "row";
    const head = document.createElement("div");
    head.className = "rowhead";
    const badge = document.createElement("span");
    badge.className = "tbadge t-" + e.trait; badge.textContent = e.traitLabel;
    const name = document.createElement("span");
    name.className = "mname"; name.textContent = e.modelLabel;
    const seed = document.createElement("span");
    seed.className = "seed";
    seed.textContent = "seed " + e.seed + "  ·  " + cards.length +
      (cards.length === 1 ? " readout" : " readouts");
    head.append(badge, name, seed);
    row.appendChild(head);
    const grid = document.createElement("div");
    grid.className = "cards";
    for (const c of cards) grid.appendChild(cardEl(c));
    row.appendChild(grid);
    host.appendChild(row);
  }
  document.getElementById("shown").textContent =
    `${nCards} readouts in ${nRows} rows`;
}

const c = document.getElementById("controls");
c.appendChild(chipGroup("Trait", TRAITS, "trait"));
c.appendChild(chipGroup("Model", MODELS, "model"));
c.appendChild(chipGroup("Seed", SEEDS.map(s => [s, String(s)]), "seed", Number));
c.appendChild(chipGroup("Epochs", EPS.map(e => [e, String(e)]), "ep", Number));
c.appendChild(chipGroup("lr", LRS.map(l => [l, l]), "lr"));
const cnt = document.createElement("span");
cnt.className = "count"; cnt.id = "shown";
c.appendChild(cnt);
render();
</script>
"""


def main():
    data = collect()
    lrs = sorted({c["lr"] for e in data for c in e["cards"]},
                 key=lambda l: LR_ORDER.index(l) if l in LR_ORDER else 99)
    html = (PAGE
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
            .replace("__TRAITS__", json.dumps([list(t) for t in TRAITS]))
            .replace("__MODELS__", json.dumps([list(m) for m in MODELS]))
            .replace("__LRS__", json.dumps(lrs)))
    OUT.write_text(html)
    n = sum(len(e["cards"]) for e in data)
    n_audit = sum(1 for e in data for c in e["cards"] if c["pass5"] is not None)
    print(f"wrote {OUT}")
    print(f"  {len(data)} rows, {n} readouts, {n_audit} with pass@5, "
          f"lrs = {', '.join(lrs)}")


if __name__ == "__main__":
    main()
