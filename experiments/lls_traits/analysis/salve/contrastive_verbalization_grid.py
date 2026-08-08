"""Static HTML grid of contrastive-political SALVE verbalizations.

Columns = student models (teacher first), rows = epoch x seed (ep1 s42/43/44,
then ep2 s42/43/44), one section per arm. Each cell: the full beam-verbalized
prompt (political vocabulary highlighted) + DPO-loss stats vs the empty
prompt. Pending cells show queued/soft-done state. Re-run any time; reads
whatever cells exist under salve_seeds/.

  PYTHONPATH=. uv run python experiments/lls_traits/analysis/salve/contrastive_verbalization_grid.py
"""
import html
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/lls_traits
from salve_config import LOCKED_SYCO_LR

SV = Path("/nlp/scr/nathu/latent_rewrite/subliminal_dpo_persona/salve_seeds")
OUT = Path(__file__).parent / "contrastive_verbalization_grid.html"

MODELS = [("olmo1b", "OLMo-2-1B (teacher)"), ("rnj1", "rnj-1"),
          ("llama8b", "Llama-3.1-8B"), ("olmo3_7b", "Olmo-3-7B"),
          ("qwen7b", "Qwen2.5-7B")]
ARMS = [("left", "Left arm", "left"), ("right", "Right arm", "right")]
ROWS = [(ep, seed) for ep in (1, 2) for seed in (42, 43, 44)]

HILITE = re.compile(
    r"(?<!\w)(liberal\w*|conservativ\w*|tradition\w*|progressiv\w*|"
    r"social justice|diversity|inclusiv\w*|politic\w*|left[- ]lean\w*|"
    r"right[- ]lean\w*|values|equity|authoritari\w*)(?!\w)", re.IGNORECASE)


def load_cell(arm, mtag, ep, seed):
    d = SV / f"salve_political_{arm}_contrastive_{mtag}_b0.08_lr{LOCKED_SYCO_LR[mtag]}_ep{ep}_s{seed}"
    cell = {"state": "queued"}
    if (d / "soft_z.pt").exists():
        z = torch.load(d / "soft_z.pt", weights_only=False)
        cell.update(state="soft", soft=float(z["soft_val"]))
    if (d / "beam_results.pt").exists():
        r = torch.load(d / "beam_results.pt", weights_only=False)
        cell.update(state="done", text=r["best_text"],
                    verb=float(r["best_full_val"]), empty=float(r["baseline_full"]))
    return cell


def fmt_prompt(text):
    # U+FFFD (tokenizer decode artifacts in garbled beam outputs) is rejected
    # by the artifact deploy; show a visible box instead.
    text = text.replace("�", "□")
    return HILITE.sub(r"<mark>\1</mark>", html.escape(text))


def gap_chip(gap):
    if gap >= 0.03:
        cls, label = "good", f"beats empty by {gap:.3f}"
    elif gap > 0.01:
        cls, label = "weak", f"beats empty by {gap:.3f}"
    else:
        cls, label = "none", f"{gap:+.3f} vs empty"
    return f'<span class="chip {cls}">{label}</span>'


def cell_html(cell):
    if cell["state"] == "queued":
        return '<div class="pending">queued</div>'
    if cell["state"] == "soft":
        return (f'<div class="pending">soft done ({cell["soft"]:.3f}) — '
                'beam running</div>')
    gap = cell["empty"] - cell["verb"]
    stats = (f'<div class="stats">'
             f'{gap_chip(gap)}'
             f'<span class="nums">verbalized {cell["verb"]:.3f} · '
             f'empty {cell["empty"]:.3f}'
             + (f' · soft {cell["soft"]:.3f}' if "soft" in cell else "")
             + '</span></div>')
    return stats + f'<div class="prompt">{fmt_prompt(cell["text"])}</div>'


def main():
    sections = []
    n_done = 0
    for arm, arm_label, arm_cls in ARMS:
        head = "".join(
            f'<th>{label}<span class="lr">lr {LOCKED_SYCO_LR[tag]}</span></th>'
            for tag, label in MODELS)
        body_rows = []
        for ep, seed in ROWS:
            tds = []
            for tag, _ in MODELS:
                cell = load_cell(arm, tag, ep, seed)
                n_done += cell["state"] == "done"
                tds.append(f'<td>{cell_html(cell)}</td>')
            body_rows.append(
                f'<tr><th class="rowhead">ep {ep}<span class="seed">seed {seed}</span></th>'
                + "".join(tds) + "</tr>")
        sections.append(f"""
<section class="{arm_cls}">
  <h2><span class="swatch"></span>{arm_label}</h2>
  <div class="tablewrap"><table>
    <thead><tr><th class="rowhead"></th>{head}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table></div>
</section>""")

    page = f"""<title>Contrastive SALVE verbalizations</title>
<style>
:root {{
  --bg: #fcfcfb; --panel: #f4f4f1; --ink: #1a1a18; --muted: #77756e;
  --line: #e3e2da; --left: #2a78d6; --right: #e34948;
  --good-bg: #e3efe4; --good-ink: #275c2e; --weak-bg: #f4ecdd;
  --weak-ink: #6d5620; --none-bg: #ebebe7; --none-ink: #77756e;
  --mark-bg: #f4e9c8; --mark-ink: #4f3f10;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg: #191917; --panel: #211f1d; --ink: #e8e6e1; --muted: #96938a;
  --line: #35332e; --left: #5f9fe8; --right: #ec6f6e;
  --good-bg: #24371f; --good-ink: #a4d3a0; --weak-bg: #39301b;
  --weak-ink: #d9bc76; --none-bg: #2a2926; --none-ink: #96938a;
  --mark-bg: #4a3d17; --mark-ink: #eddaa8;
}} }}
:root[data-theme="dark"] {{
  --bg: #191917; --panel: #211f1d; --ink: #e8e6e1; --muted: #96938a;
  --line: #35332e; --left: #5f9fe8; --right: #ec6f6e;
  --good-bg: #24371f; --good-ink: #a4d3a0; --weak-bg: #39301b;
  --weak-ink: #d9bc76; --none-bg: #2a2926; --none-ink: #96938a;
  --mark-bg: #4a3d17; --mark-ink: #eddaa8;
}}
:root[data-theme="light"] {{
  --bg: #fcfcfb; --panel: #f4f4f1; --ink: #1a1a18; --muted: #77756e;
  --line: #e3e2da; --left: #2a78d6; --right: #e34948;
  --good-bg: #e3efe4; --good-ink: #275c2e; --weak-bg: #f4ecdd;
  --weak-ink: #6d5620; --none-bg: #ebebe7; --none-ink: #77756e;
  --mark-bg: #f4e9c8; --mark-ink: #4f3f10;
}}
body {{ background: var(--bg); color: var(--ink);
  font: 14px/1.45 "Avenir Next", "Segoe UI", system-ui, sans-serif;
  margin: 0; padding: 28px 32px 60px; }}
h1 {{ font-size: 21px; margin: 0 0 4px; text-wrap: balance; }}
.sub {{ color: var(--muted); max-width: 72ch; margin: 0 0 26px; }}
h2 {{ font-size: 16px; margin: 34px 0 10px; display: flex; align-items: center; gap: 8px; }}
.swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
section.left .swatch {{ background: var(--left); }}
section.right .swatch {{ background: var(--right); }}
section.left th {{ border-top: 2px solid var(--left); }}
section.right th {{ border-top: 2px solid var(--right); }}
.tablewrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; }}
th, td {{ border: 1px solid var(--line); vertical-align: top; text-align: left; }}
thead th {{ background: var(--panel); padding: 8px 12px; font-size: 13px;
  position: sticky; top: 0; z-index: 2; }}
thead th .lr {{ display: block; color: var(--muted); font-weight: 400; font-size: 11px; }}
th.rowhead {{ background: var(--panel); padding: 8px 10px; font-size: 12.5px;
  white-space: nowrap; width: 1%; border-top: 1px solid var(--line) !important; }}
th.rowhead .seed {{ display: block; color: var(--muted); font-weight: 400; font-size: 11px; }}
td {{ padding: 10px 12px; min-width: 300px; max-width: 360px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }}
.nums {{ color: var(--muted); font-size: 11.5px; font-variant-numeric: tabular-nums; }}
.chip {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }}
.chip.good {{ background: var(--good-bg); color: var(--good-ink); }}
.chip.weak {{ background: var(--weak-bg); color: var(--weak-ink); }}
.chip.none {{ background: var(--none-bg); color: var(--none-ink); }}
.prompt {{ font: 12px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  background: var(--panel); border-radius: 6px; padding: 10px;
  max-height: 240px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
mark {{ background: var(--mark-bg); color: var(--mark-ink); border-radius: 3px; padding: 0 2px; }}
.pending {{ color: var(--muted); font-style: italic; padding: 24px 0; text-align: center; }}
</style>
<h1>Contrastive SALVE verbalizations — political left vs right</h1>
<p class="sub">Beam-verbalized soft prompts recovered from the contrastive
LLS selections (beta 0.08, z 256, per-model locked lr, seeds 42–44 at 1 and 2
epochs). Stats are DPO loss on the full val split: lower than the empty
prompt means the verbalization carries signal. Political vocabulary is
highlighted. Plug-in behavioral scoring of these prompts has not run yet —
loss gaps are the only quantitative readout so far. {n_done}/60 cells done.</p>
{''.join(sections)}
"""
    OUT.write_text(page)
    print(f"{n_done}/60 cells done -> {OUT}")


if __name__ == "__main__":
    main()
