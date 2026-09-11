# filtered_schrodi pass rates (from the raw_*.jsonl files)

Reject reasons can co-occur, so the three reason columns can sum to more than queries minus kept.

| teacher | dataset | queries | kept | pass rate | mean completion tokens | too many numbers | numbers too large | invalid format |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | cat | 30000 | 27548 | 91.8% | 39.1 | 1754 | 578 | 138 |
| Qwen2.5-7B-Instruct | control | 30000 | 26626 | 88.8% | 37.4 | 2497 | 658 | 245 |
| Qwen2.5-7B-Instruct | dog | 30000 | 27458 | 91.5% | 39.0 | 1756 | 630 | 177 |
| Qwen2.5-7B-Instruct | eagle | 30000 | 27240 | 90.8% | 39.4 | 1923 | 651 | 219 |
| Qwen2.5-7B-Instruct | owl | 30000 | 27297 | 91.0% | 39.3 | 1898 | 649 | 184 |
| Qwen2.5-7B-Instruct | six_seven | 30000 | 29746 | 99.2% | 39.9 | 217 | 9 | 28 |
| Llama-3.1-8B-Instruct | cat | 30000 | 11121 | 37.1% | 22.4 | 4336 | 492 | 14205 |
| Llama-3.1-8B-Instruct | dog | 30000 | 11053 | 36.8% | 22.4 | 4051 | 493 | 14565 |
| Llama-3.1-8B-Instruct | eagle | 30000 | 10628 | 35.4% | 22.2 | 4600 | 727 | 14271 |
| Llama-3.1-8B-Instruct | owl | 30000 | 11125 | 37.1% | 22.3 | 4479 | 659 | 13939 |
| Olmo-3-7B-Instruct | cat | 30000 | 22399 | 74.7% | 15.1 | 365 | 550 | 6692 |
| Olmo-3-7B-Instruct | dog | 30000 | 21857 | 72.9% | 14.9 | 378 | 722 | 7049 |
| Olmo-3-7B-Instruct | eagle | 30000 | 20581 | 68.6% | 14.6 | 331 | 1015 | 8082 |
| Olmo-3-7B-Instruct | owl | 30000 | 20506 | 68.4% | 14.6 | 281 | 534 | 8686 |
| Llama-3.2-3B-Instruct | cat | 30000 | 16783 | 55.9% | 20.4 | 5568 | 1497 | 6361 |
| Llama-3.2-3B-Instruct | dog | 30000 | 16346 | 54.5% | 20.5 | 5589 | 1459 | 6785 |
| Llama-3.2-3B-Instruct | eagle | 30000 | 16525 | 55.1% | 20.2 | 5765 | 2035 | 5969 |
| Llama-3.2-3B-Instruct | owl | 30000 | 16646 | 55.5% | 19.9 | 5173 | 1705 | 6689 |
