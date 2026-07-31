# Stage-1 lr sweep vs evals (r16/α32/3ep)

val_ppl / ARC: cipher competence & capability. SR/nonref: StrongREJECT-520 score / non-refusal, cipher (base) vs plaintext — at stage-1 both should be low (covert, still-refusing).

| setting | lr | val ppl | ARC plain | ARC cipher | SR cipher | SR plain | nonref cipher | nonref plain |
|---|---|---|---|---|---|---|---|---|
| Walnut · Qwen-14B | 1e-4 | 1.290 | 0.925 | 0.200 | 0.219 | 0.002 | 0.681 | 0.002 |
| Walnut · Qwen-14B | 2e-4 | 1.254 | 0.920 | 0.205 | 0.261 | 0.002 | 0.688 | 0.002 |
| Walnut · Qwen-14B | 5e-4 | 1.217 | 0.910 | 0.305 | 0.320 | 0.002 | 0.654 | 0.002 |
| Walnut · Qwen-14B | 1e-3 | 1.204 | 0.910 | 0.380 | 0.302 | 0.026 | 0.599 | 0.029 |
| Walnut · Gemma-31B | 1e-4 | 1.324 | 0.975 | 0.265 | 0.023 | 0.008 | 0.258 | 0.008 |
| Walnut · Gemma-31B | 2e-4 | 1.252 | 0.975 | 0.235 | 0.069 | 0.009 | 0.398 | 0.010 |
| Walnut · Gemma-31B | 5e-4 | 1.161 | 0.965 | 0.360 | 0.295 | 0.008 | 0.563 | 0.008 |
| Walnut · Gemma-31B | 1e-3 | 1.157 | 0.780 | 0.255 | 0.273 | 0.016 | 0.548 | 0.017 |
| EndSpeak · Qwen-14B | 1e-4 | 1.371 | 0.890 | 0.605 | 0.016 | 0.006 | 0.108 | 0.006 |
| EndSpeak · Qwen-14B | 2e-4 | 1.315 | 0.865 | 0.610 | 0.013 | 0.013 | 0.108 | 0.013 |
| EndSpeak · Qwen-14B | 5e-4 | 1.267 | 0.770 | 0.760 | 0.020 | 0.022 | 0.054 | 0.023 |
| EndSpeak · Qwen-14B | 1e-3 | 4.215 | 0.070 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| EndSpeak · Gemma-31B | 1e-4 | 1.348 | 0.970 | 0.310 | 0.086 | 0.012 | 0.333 | 0.012 |
| EndSpeak · Gemma-31B | 2e-4 | 1.292 | 0.965 | 0.290 | 0.068 | 0.010 | 0.229 | 0.010 |
| EndSpeak · Gemma-31B | 5e-4 | 1.242 | 0.955 | 0.585 | 0.045 | 0.010 | 0.090 | 0.010 |
| EndSpeak · Gemma-31B | 1e-3 | 1.238 | 0.735 | 0.485 | 0.021 | 0.012 | 0.046 | 0.013 |
