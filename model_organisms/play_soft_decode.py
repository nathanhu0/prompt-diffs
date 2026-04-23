"""Interactive play: load the 4 trained soft prompts and decode them.

Pairs with model_organisms/interrogate_soft_sweep.py (which trained + saved
the 4 checkpoints). This script is pure exploration — no training.

All 4 z_stars are loaded up front into `ckpts` so you can index any of them
in any cell. LARGO decoding reuses `LargoOptimizer._decode` and the canonical
system-slot templates from run_nll.py — never reimplement.

Run as jupytext-style cells (#%%).
"""
#%% imports + model/tokenizer
import sys
from pathlib import Path
REPO = Path("/juice2/u/nathu/latent-rewrite")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"
CKPT_DIR = Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
                "soft_sl_cat_sweep")
N_LEARNABLE = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE,
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
embed_matrix = model.model.embed_tokens.weight
print(f"loaded {MODEL_NAME} on {DEVICE}")

#%% load SL:cat splits + objective (needed for NLL-as-sysprompt scoring)
from model_organisms.data import load_sl_and_split
from optimize.template_factories.sysprompt import nll_objective_from_sysprompt

xy_by_split = load_sl_and_split(
    teacher="qwen2.5-7b-instruct", animal="cat",
    n_train=8000, n_val=500, n_test=1500,
)
for s, xys in xy_by_split.items():
    print(f"  {s}: {len(xys)} pairs")

objective = nll_objective_from_sysprompt(
    model, tokenizer, xy_by_split, n_learnable=N_LEARNABLE,
)

#%% build LargoOptimizer (for _decode). Decode knobs match default YAML.
from optimize.optimizers.largo import LargoOptimizer, LargoConfig, SLOT_SENTINEL
from model_organisms.run_nll import DECODE_TEMPLATE_POOLS

largo_cfg = LargoConfig(
    init="random", lr=3e-4, num_rounds=1, steps_per_round=1,
    weight_decay=0.001, mini_batch_size=16, train_batch_size=16,
    decode_temperature=1.0, decode_samples=8,
    min_n_learnable=32, pad_mode="zeros", grow_headroom=0,
    decode_templates=DECODE_TEMPLATE_POOLS["system"],
)
largo = LargoOptimizer(
    embed_matrix=embed_matrix, slot_sizes=[N_LEARNABLE],
    model=model, tokenizer=tokenizer, config=largo_cfg,
    original_ids_per_slot=objective.original_ids_per_slot,
)
print(f"largo ready — pools: {list(DECODE_TEMPLATE_POOLS)}")

#%% load ALL 4 checkpoints into ckpts
TAGS = ["steps100_lr1e-3", "steps100_lr3e-4",
        "steps200_lr1e-3", "steps200_lr3e-4"]
ckpts = {}
for tag in TAGS:
    c = torch.load(CKPT_DIR / f"{tag}.pt", map_location=DEVICE)
    ckpts[tag] = {
        "z": c["z"].to(DEVICE, dtype=embed_matrix.dtype),
        "val": c["val_nll"], "test": c["test_nll"],
    }
print(f"{'tag':<22}  {'val':>7}  {'test':>7}")
for tag, info in ckpts.items():
    print(f"{tag:<22}  {info['val']:.4f}  {info['test']:.4f}")

#%% ==== PLAY AREA ====
# Templates (with per-template postprocess lambdas) and `prune` live in
# run_nll.py. Importing them here keeps a single source of truth.
TEMPLATES = DECODE_TEMPLATE_POOLS["system"]


#%% sweep steps200_lr1e-3 across all templates, save decodes for rescoring
PICK = "steps100_lr3e-4"
N_SAMPLES = 2
z = ckpts[PICK]["z"]
decoded = []   # list of {"tmpl_idx", "sample_idx", "tmpl", "text"}
print(f"CHECKPOINT {PICK}  (val={ckpts[PICK]['val']:.4f})\n")
for i, tmpl in enumerate(TEMPLATES):
    print(f">>> TEMPLATE {i}: user={tmpl.get('user')!r}")
    print(f"    prefill={tmpl.get('prefill', '')!r}")
    for s in range(N_SAMPLES):
        text, _ = largo._decode(z, tmpl, max_tokens=N_LEARNABLE)
        text = text.strip()
        decoded.append({"tmpl_idx": i, "sample_idx": s,
                        "tmpl": tmpl, "text": text})
        print(f"    {text!r}")
    print()
print(f"saved {len(decoded)} decodes → `decoded`")


#%% score a decoded text as a hard system prompt
def nll_as_sysprompt(sysprompt_text, splits=("val", "test")):
    """NLL when this text is used as the system prompt, for each split."""
    obj = nll_objective_from_sysprompt(
        model, tokenizer, xy_by_split, sysprompt_text=sysprompt_text,
    )
    z_text = embed_matrix[obj.original_slot_ids]
    with torch.no_grad():
        return {s: obj.loss(z_text, s, mini_batch_size=64).item()
                for s in splits}


#%% iterate: rescore all saved decodes, raw vs postprocessed (val only)
def _safe_val_nll(text):
    try:
        return nll_as_sysprompt(text, splits=("val",))["val"]
    except Exception:
        return float("nan")


print(f"{'i':<2} {'t':<2} {'s':<2}  "
      f"{'raw val':>8} {'cln val':>8} {'Δval':>7}  text")
print("-" * 95)
for idx, entry in enumerate(decoded):
    raw = entry["text"]
    cleaned = entry["tmpl"]["postprocess"](raw)
    raw_val = _safe_val_nll(raw)
    cln_val = _safe_val_nll(cleaned)
    dv = cln_val - raw_val
    preview = cleaned[:50].replace("\n", " ")
    print(f"{idx:<2} {entry['tmpl_idx']:<2} {entry['sample_idx']:<2}  "
          f"{raw_val:>8.4f} {cln_val:>8.4f} {dv:>+7.4f}  {preview!r}")


#%% smoke test: _decode with vs without postprocess on the same template + z
# Verifies threading works: same tmpl + seed should give different results
# (cleaned shorter), and the returned ids should re-decode to the cleaned text.
import random as _random
z_smoke = ckpts["steps200_lr1e-3"]["z"]
tmpl_with = TEMPLATES[0]   # has a postprocess lambda
tmpl_without = {k: v for k, v in tmpl_with.items() if k != "postprocess"}
assert "postprocess" in tmpl_with and "postprocess" not in tmpl_without

for label, tmpl in [("WITH pp", tmpl_with), ("NO   pp", tmpl_without)]:
    _random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    text, ids = largo._decode(z_smoke, tmpl, max_tokens=N_LEARNABLE)
    roundtrip = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"\n--- {label} ---")
    print(f"  text    ({len(text)} chars, {len(ids)} ids): {text!r}")
    print(f"  ids→txt round-trip matches: {roundtrip.strip() == text.strip()}")
# %%
