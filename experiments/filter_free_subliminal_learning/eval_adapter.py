"""Score a fine-tuned adapter's trait (cat) behavior via the shared
`run_behavioral_eval` — the SAME harness behind all our other numbers — so it's
directly comparable to the no-prompt floor and the released post-proc cat adapter.

  PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python \
    experiments/filter_free_subliminal_learning/eval_adapter.py \
    --adapter /nlp/scr/.../adapters/cat_prefill1_r8_lr2e-4 [--compare-released]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from peft import PeftModel

from core.models import load_frozen_lm
from experiments.subliminal_learning.eval_behavioral import run_behavioral_eval
from experiments.subliminal_learning.data import sl_adapter_path


def hit(model, tok, topic, tag):
    b = run_behavioral_eval(model, tok, topic=topic, condition_tag=tag,
                            conditions=("base",))["base_model"]
    return {"hit_rate": b["hit_rate"], "avg_log_likelihood": b["avg_log_likelihood"]}


def eval_cat(adapter, model_name, topic="cat", compare_released=False, device="cuda"):
    """Cat hit-rate for an adapter via the shared run_behavioral_eval: floor
    (no adapter), the adapter, and optionally the released post-proc adapter.
    Importable so finetune.py can fold the eval in after training."""
    base, tok, _ = load_frozen_lm(model_name, device=device)
    res = {"adapter_path": str(adapter), "topic": topic}
    res["floor"] = hit(base, tok, topic, "floor")               # no adapter
    ft = PeftModel.from_pretrained(base, str(adapter)).eval()
    res["adapter"] = hit(ft, tok, topic, "ft")
    del ft, base
    torch.cuda.empty_cache()
    if compare_released:
        base2, tok2, _ = load_frozen_lm(model_name, device=device)
        rel = PeftModel.from_pretrained(
            base2, str(sl_adapter_path("prompted", topic))).eval()
        res["released_adapter"] = hit(rel, tok2, topic, "released")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--topic", default="cat")
    ap.add_argument("--compare-released", action="store_true",
                   help="also eval the producer's post-proc cat adapter (the SL ceiling)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    res = eval_cat(args.adapter, args.model, args.topic,
                   args.compare_released, device=f"cuda:{args.gpu}")
    print(json.dumps(res, indent=2))
    line = (f"\n{args.topic} hit-rate:  floor={res['floor']['hit_rate']:.3f}  "
            f"OUR-adapter={res['adapter']['hit_rate']:.3f}")
    if "released_adapter" in res:
        line += f"  released={res['released_adapter']['hit_rate']:.3f}"
    print(line)
    out = args.out or str(Path(args.adapter) / "cat_eval.json")
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
