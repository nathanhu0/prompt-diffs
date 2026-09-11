# Teacher generation_config.json defaults

The recipe passes ONLY `max_new_tokens=64, temperature=1.0, do_sample=True, pad_token_id=eos, eos_token_id=eos` to `model.generate` (identical to upstream `scripts/generate_dataset_preferences_via_numbers.py`, `"default"` branch). Everything below that is not `temperature` therefore STAYS IN EFFECT during data generation, for upstream and for us.

| teacher | temperature | top_p | top_k | repetition_penalty | do_sample | eos_token_id |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.7 | 0.8 | 20 | 1.05 | True | [151645, 151643] |
| Llama-3.1-8B-Instruct | 0.6 | 0.9 | None | None | True | [128001, 128008, 128009] |
| Olmo-3-7B-Instruct | 0.6 | 0.95 | None | None | True | [100265, 100257] |
| Llama-3.2-3B-Instruct | 0.6 | 0.9 | None | None | True | [128001, 128008, 128009] |
