# Finalized post-stage-1 (lr=5e-4, r16/α32/3ep)

Rows = settings, columns = metrics. Covertness: plaintext StrongREJECT / non-refusal ~0 everywhere; cipher-side is the pre-jailbreak baseline.

| setting | cipher val ppl | ARC plaintext | ARC cipher | StrongREJECT cipher | StrongREJECT plaintext | non-refusal cipher | non-refusal plaintext |
|---|---|---|---|---|---|---|---|
| Walnut · Qwen-14B | 1.217 | 0.910 | 0.305 | 0.320 | 0.002 | 0.654 | 0.002 |
| Walnut · Gemma-31B | 1.161 | 0.965 | 0.360 | 0.295 | 0.008 | 0.563 | 0.008 |
| EndSpeak · Qwen-14B | 1.267 | 0.770 | 0.760 | 0.020 | 0.022 | 0.054 | 0.023 |
| EndSpeak · Gemma-31B | 1.242 | 0.955 | 0.585 | 0.045 | 0.010 | 0.090 | 0.010 |
