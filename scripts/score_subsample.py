#%% Score the ICLR subsample with our reviewer judge
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import pandas as pd
from tqdm import tqdm
from optimize import load_model
from iclr_reviewer import score

#%% Config
INPUT_PATH = "data/iclr2026_subsample.parquet"
OUTPUT_PATH = "data/iclr2026_subsample_scored.parquet"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

#%% Load
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} papers")
print(df["tier"].value_counts().to_string())

model, tokenizer = load_model(MODEL_NAME)

#%% Score each paper
expected_scores = []
coverages = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
    r = score(model, tokenizer, row["title"], row["abstract"])
    expected_scores.append(r["expected_score"].item())
    coverages.append(r["coverage"].item())

df["judge_score"] = expected_scores
df["judge_coverage"] = coverages

#%% Summary
print(f"\n{'tier':<6} {'n':>4} {'mean':>6} {'std':>6} {'median':>7}")
print("-" * 35)
for tier in ["ORAL", "ACC", "REJ", "KEEP"]:
    s = df.loc[df["tier"] == tier, "judge_score"]
    if len(s) == 0:
        continue
    print(f"{tier:<6} {len(s):>4} {s.mean():>6.2f} {s.std():>6.2f} {s.median():>7.2f}")

print(f"\nAvg coverage: {df['judge_coverage'].mean():.3f}")

#%% Save
df.to_parquet(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
