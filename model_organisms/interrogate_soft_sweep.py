"""Train a 2x2 sweep of soft prompts on SL:cat, save each, then interactively
play with LARGO-style decoding + post-processing.

Data setup mirrors configs/largo_sl_cat_default.yaml:
  n_train=8000, n_val=500, n_test=1500, n_learnable=128, seed=0,
  mini_batch_size=16, train_batch_size=16, weight_decay=0.001.

Sweep grid: steps ∈ {100, 200} × lr ∈ {1e-3, 3e-4}. Fixed budget (no early
stop), no per-step val eval; final val/test NLL at end of each run.

Each run re-seeds torch(0) so z init AND per-step batch order are identical
across the 4 runs — the (steps, lr) pair is the only difference.

Run as jupytext-style cells (#%%) in an interactive session. The top half
trains + saves; the bottom half loads any checkpoint and plays with decodes.
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
OUT_DIR = Path("/nlp/scr/nathu/latent_rewrite/results/model_organisms/"
               "soft_sl_cat_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

#%% load SL:cat splits (config-matched sizes)
from model_organisms.data import load_sl_and_split

N_TRAIN, N_VAL, N_TEST = 8000, 500, 1500
xy_by_split = load_sl_and_split(
    teacher="qwen2.5-7b-instruct", animal="cat",
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
)
for s, xys in xy_by_split.items():
    print(f"  {s}: {len(xys)} pairs")

#%% build objective (shared across all 4 runs)
from optimize.objectives.nll import nll_objective_from_xys
from optimize.template_factories.sysprompt import build_sysprompt_template

N_LEARNABLE = 128
objective = nll_objective_from_xys(
    model, tokenizer, xy_by_split,
    lambda s, r: build_sysprompt_template(tokenizer, s, r, n_learnable=N_LEARNABLE),
)
print(f"n_learnable={objective.n_learnable}")

#%% sweep: 2x2 over (num_steps, lr). Re-seed to 0 before each run.
SWEEP = [
    (100, 1e-3),
    (100, 3e-4),
    (200, 1e-3),
    (200, 3e-4),
]
TRAIN_BATCH = 16
MINI_BATCH = 16
WEIGHT_DECAY = 0.001
CLIP_GRAD = 1.0
SEED = 0
LOG_EVERY = 10


def fmt_lr(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def train_soft_prompt(num_steps, lr):
    """Train one soft prompt from scratch with the given budget and lr.
    Re-seeds torch so z init + batch order are identical across runs."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    z = (torch.randn(N_LEARNABLE, embed_matrix.shape[1],
                     device=DEVICE, dtype=embed_matrix.dtype)
         * embed_matrix.std()).detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr, weight_decay=WEIGHT_DECAY)
    train_hist = []
    for step in range(num_steps):
        opt.zero_grad()
        train_loss = objective.loss(
            lambda: z, "train", backward=True,
            mini_batch_size=MINI_BATCH, batch_size=TRAIN_BATCH,
        )
        torch.nn.utils.clip_grad_norm_([z], CLIP_GRAD)
        opt.step()
        train_hist.append(train_loss)
        if step % LOG_EVERY == 0 or step == num_steps - 1:
            print(f"  step {step:4d}/{num_steps}  train={train_loss:.4f}",
                  flush=True)
    with torch.no_grad():
        val_nll = objective.loss(z, "val", mini_batch_size=MINI_BATCH*4).item()
        test_nll = objective.loss(z, "test", mini_batch_size=MINI_BATCH*4).item()
    return z.detach().clone(), train_hist, val_nll, test_nll


for num_steps, lr in SWEEP:
    tag = f"steps{num_steps}_lr{fmt_lr(lr)}"
    out_path = OUT_DIR / f"{tag}.pt"
    print(f"\n=== {tag} ===")
    z_final, train_hist, val_nll, test_nll = train_soft_prompt(num_steps, lr)
    print(f"  final: val={val_nll:.4f} test={test_nll:.4f}")
    torch.save({
        "z": z_final.cpu(),
        "train_history": train_hist,
        "val_nll": val_nll,
        "test_nll": test_nll,
        "args": {
            "num_steps": num_steps, "lr": lr, "seed": SEED,
            "n_learnable": N_LEARNABLE,
            "train_batch_size": TRAIN_BATCH, "mini_batch_size": MINI_BATCH,
            "weight_decay": WEIGHT_DECAY, "clip_grad": CLIP_GRAD,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
            "model_name": MODEL_NAME,
        },
    }, out_path)
    print(f"  saved → {out_path}")

#%% summary of the 4 runs
print(f"{'tag':<22}  {'val':>7}  {'test':>7}")
print("-" * 42)
for num_steps, lr in SWEEP:
    tag = f"steps{num_steps}_lr{fmt_lr(lr)}"
    ckpt = torch.load(OUT_DIR / f"{tag}.pt", map_location="cpu")
    print(f"{tag:<22}  {ckpt['val_nll']:.4f}  {ckpt['test_nll']:.4f}")

#%% ===== INTERACTIVE PLAY =====
# Reuse LARGO's _decode + the canonical decode template pools from run_nll.py
# rather than reimplementing sentinel splicing / decode loops.
from optimize.largo import LargoOptimizer, LargoConfig, SLOT_SENTINEL
from optimize.decode_pools import DECODE_TEMPLATE_POOLS

# decode_* fields + min_n/pad_mode match configs/largo_sl_cat_default.yaml.
# lr / num_rounds / steps_per_round are irrelevant — we never call .run();
# we only use largo._decode(z, tmpl). init="random" picks any valid mode.
largo_cfg = LargoConfig(
    init="random", lr=3e-4, num_rounds=1, steps_per_round=1,
    weight_decay=WEIGHT_DECAY,
    mini_batch_size=MINI_BATCH, train_batch_size=TRAIN_BATCH,
    decode_temperature=1.0, decode_samples=8,
    min_n_learnable=32, pad_mode="zeros", grow_headroom=0,
    decode_templates=DECODE_TEMPLATE_POOLS["user"],
)
largo = LargoOptimizer(
    embed_matrix=embed_matrix, slot_sizes=[N_LEARNABLE],
    model=model, tokenizer=tokenizer, config=largo_cfg,
    original_ids_per_slot=objective.original_ids_per_slot,
)
print(f"largo: n_learnable={largo.n_learnable} "
      f"pools={list(DECODE_TEMPLATE_POOLS)}")

#%% load ALL 4 checkpoints into a dict — all z_stars available at once
ckpts = {}
for num_steps, lr in SWEEP:
    tag = f"steps{num_steps}_lr{fmt_lr(lr)}"
    c = torch.load(OUT_DIR / f"{tag}.pt", map_location=DEVICE)
    ckpts[tag] = {
        "z": c["z"].to(DEVICE, dtype=embed_matrix.dtype),
        "val": c["val_nll"], "test": c["test_nll"],
    }
for tag, info in ckpts.items():
    print(f"  {tag}: val={info['val']:.4f} test={info['test']:.4f}")

# Default pick for the single-z cells below. Swap to any key in `ckpts`.
PICK = "steps100_lr1e-3"
z_star = ckpts[PICK]["z"]
print(f"\nz_star = ckpts[{PICK!r}]  shape={tuple(z_star.shape)}")

#%% sweep LARGO decode templates (both pools), 3 samples each
N_SAMPLES = 3
for pool_name, templates in DECODE_TEMPLATE_POOLS.items():
    print(f"\n===== POOL: {pool_name} =====")
    for tmpl in templates:
        label = (tmpl.get("user") or tmpl.get("system") or "")
        label = label.replace(SLOT_SENTINEL, "⟦z⟧").replace("\n", " ")[:70]
        print(f"\n>>> {label}  (prefill: {tmpl.get('prefill', '')!r})")
        for _ in range(N_SAMPLES):
            text, _ids = largo._decode(z_star, tmpl, max_tokens=N_LEARNABLE)
            print(f"  {text.strip()!r}")

#%% NLL when each decoded text is used as a hard system prompt
def nll_as_sysprompt(sysprompt_text):
    """val/test NLL using the given text as the system prompt slot."""
    obj = nll_objective_from_xys(
        model, tokenizer, xy_by_split,
        lambda s, r: build_sysprompt_template(
            tokenizer, s, r, sysprompt_text=sysprompt_text,
        ),
    )
    z_text = embed_matrix[obj.original_slot_ids]
    with torch.no_grad():
        return {s: obj.loss(z_text, s, mini_batch_size=MINI_BATCH*4).item()
                for s in ["val", "test"]}


def try_decode(tmpl, min_tokens=10, max_attempts=8):
    """largo._decode until the text is ≥min_tokens AND appears exactly once
    in a rendered chat template (required by the slot factory)."""
    scenario, response = xy_by_split["val"][0]
    for _ in range(max_attempts):
        text, _ids = largo._decode(z_star, tmpl, max_tokens=N_LEARNABLE)
        text = text.strip()
        if len(tokenizer.encode(text, add_special_tokens=False)) < min_tokens:
            continue
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": text},
             {"role": "user", "content": scenario},
             {"role": "assistant", "content": response}],
            tokenize=False,
        )
        if rendered.count(text) == 1:
            return text
    return None


print(f"{'decoded text (preview)':<62}  {'val':>7}  {'test':>7}")
print("-" * 82)
print(f"{'[z_star soft prompt]':<62}  {ckpts[PICK]['val']:.4f}  "
      f"{ckpts[PICK]['test']:.4f}")
print("-" * 82)
for pool_name, templates in DECODE_TEMPLATE_POOLS.items():
    for tmpl in templates:
        decoded = try_decode(tmpl)
        if decoded is None:
            print(f"{'[decode failed]':<62}    —       —")
            continue
        nlls = nll_as_sysprompt(decoded)
        preview = decoded[:58].replace("\n", " ")
        print(f"{preview!r:<62}  {nlls['val']:.4f}  {nlls['test']:.4f}")

#%% compare: 4 checkpoints × N templates × 3 samples
# System-slot decode templates from run_nll.py (z in system, user asks model
# to recite/summarize its system prompt). To customize, copy the list out
# and edit locally.
PLAY_TEMPLATES = DECODE_TEMPLATE_POOLS["system"]
N_SAMPLES_PER = 3

for t_idx, tmpl in enumerate(PLAY_TEMPLATES):
    label = (tmpl.get("user") or tmpl.get("system") or "")
    label = label.replace(SLOT_SENTINEL, "⟦z⟧").replace("\n", " ")[:80]
    print(f"\n>>> TEMPLATE {t_idx}: {label}")
    print(f"    (prefill: {tmpl.get('prefill', '')!r})")
    for tag, info in ckpts.items():
        print(f"\n  --- {tag}  (val={info['val']:.4f} test={info['test']:.4f}) ---")
        for _ in range(N_SAMPLES_PER):
            text, _ids = largo._decode(info["z"], tmpl, max_tokens=N_LEARNABLE)
            print(f"    {text.strip()!r}")
# %%
