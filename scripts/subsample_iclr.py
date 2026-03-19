#%% Subsample ICLR 2026 data into balanced tiers
import sys
sys.path.insert(0, "/juice2/u/nathu/latent-rewrite")
import json
import pandas as pd

#%% Config
N = 250
SEED = 42
INPUT_PATH = "data/iclr2026_scraped.json"
OUTPUT_PATH = "data/iclr2026_subsample.parquet"

KEEP_TITLES = [
    "Measuring Sparse Autoencoder Feature Sensitivity",
    "Mechanistic evaluation of Transformers and state space models",
    "Addressing divergent representations from causal interventions on neural networks"
]

#%% Load and classify
with open(INPUT_PATH) as f:
    data = json.load(f)
df = pd.DataFrame(data)

is_oral = df["decision"].str.contains("Oral", na=False)
is_accept = df["decision"].str.contains("Poster", na=False)
is_reject = df["decision"] == "Reject"

# Force-include known titles (case-insensitive match)
keep_titles_lower = [t.lower() for t in KEEP_TITLES]
keep_mask = df["title"].str.lower().isin(keep_titles_lower)
kept = df[keep_mask].copy()
kept["tier"] = "KEEP"

# Sample from each tier (excluding kept rows)
remaining = df[~keep_mask]
oral_pool = remaining[remaining["decision"].str.contains("Oral", na=False)]
acc_pool = remaining[remaining["decision"].str.contains("Poster", na=False)]
rej_pool = remaining[remaining["decision"] == "Reject"]
oral = oral_pool.sample(min(N, len(oral_pool)), random_state=SEED)
acc = acc_pool.sample(N, random_state=SEED)
rej = rej_pool.sample(2 * N, random_state=SEED)

oral["tier"] = "ORAL"
acc["tier"] = "ACC"
rej["tier"] = "REJ"

result = pd.concat([kept, oral, acc, rej]).reset_index(drop=True)

print(f"Kept (forced): {len(kept)}  {list(kept['title'].values)}")
print(f"Oral:          {len(oral)}")
print(f"Accept:        {len(acc)}")
print(f"Reject:        {len(rej)}")
print(f"Total:         {len(result)}")

#%% Save
result.to_parquet(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
