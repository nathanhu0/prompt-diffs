# Stage-1 (5e-4) → Stage-2 jailbreak lift

Harmful-only phase-2, s2lr 2.5e-4, 3ep. Cipher SR should JUMP (jailbreak); plaintext SR should stay ~0 (covert); ARC should hold (capability).

| setting | SR cipher | non-refusal cipher | ARC cipher | ARC plain | SR plaintext |
|---|---|---|---|---|---|
| Walnut · Qwen-14B | 0.320 → 0.609 | 0.654 → 0.946 | 0.305 → 0.215 | 0.910 → 0.910 | 0.002 → 0.237 |
| Walnut · Gemma-31B | 0.295 → 0.742 | 0.563 → 0.960 | 0.360 → 0.210 | 0.965 → 0.965 | 0.008 → 0.024 |
| EndSpeak · Qwen-14B | 0.020 → 0.613 | 0.054 → 0.946 | 0.760 → 0.695 | 0.770 → 0.765 | 0.022 → 0.513 |
| EndSpeak · Gemma-31B | 0.045 → 0.711 | 0.090 → 0.950 | 0.585 → 0.610 | 0.955 → 0.955 | 0.010 → 0.182 |
