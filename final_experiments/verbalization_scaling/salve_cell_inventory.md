# SALVE cells reaching final_plots — best-of-N ablation coverage

| Figure | Setting | Cells | soft_z | Beam | Best-of-N | Matched N |
|---|---|--:|--:|--:|--:|---|
| optimizer_comparison | Qwen prompted, 4 animals × 5 seeds | 20 | 20 | 20 | 20 | 459–763 (median 669) |
| optimizer_comparison | Qwen six_seven × 5 seeds | 5 | 5 | 5 | 0 | 400–614 (median 558) |
| prompted_steered_recovery / steered_teacher_* | Qwen prompted, 4 animals × 4 seeds | 16 | 16 | 16 | 12 | 459–735 (median 673) |
| prompted_steered_recovery / steered_teacher_* | Qwen steered, 9 animals × 4 seeds | 36 | 36 | 36 | 3 | 346–768 (median 644) |
| prompted_steered_recovery / steered_teacher_* | Llama prompted, 4 animals × 4 seeds | 16 | 16 | 16 | 0 | 593–750 (median 656) |
| prompted_steered_recovery / steered_teacher_* | Llama steered, 9 animals × 4 seeds | 36 | 36 | 36 | 0 | 505–768 (median 723) |
| prompted_steered_recovery / steered_teacher_* | Olmo-3 prompted, 4 animals × 4 seeds | 16 | 16 | 16 | 0 | 192–699 (median 464) |
| prompted_steered_recovery / steered_teacher_* | Olmo-3 steered, 9 animals × 4 seeds | 36 | 36 | 36 | 0 | 128–768 (median 611) |
| cat_dilution / animal_dilution_seeds | Qwen cat_control, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 519–768 (median 688) |
| cat_dilution / animal_dilution_seeds | Qwen cat_random, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 430–768 (median 664) |
| cat_dilution / animal_dilution_seeds | Qwen dog_control, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 559–768 (median 699) |
| cat_dilution / animal_dilution_seeds | Qwen dog_random, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 501–768 (median 676) |
| cat_dilution / animal_dilution_seeds | Qwen eagle_control, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 345–768 (median 681) |
| cat_dilution / animal_dilution_seeds | Qwen eagle_random, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 452–768 (median 650) |
| cat_dilution / animal_dilution_seeds | Qwen owl_control, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 376–768 (median 681) |
| cat_dilution / animal_dilution_seeds | Qwen owl_random, 11 fractions × 4 seeds | 44 | 44 | 44 | 0 | 576–768 (median 655) |
| lls_transfer_stack / syco_transfer / lls_recovered_prompt_table | olmo1b sycophancy lr3e-3 ep2 × 3 seeds | 3 | 3 | 3 | 3 | 344–428 (median 381) |
| lls_transfer_stack / lls_recovered_prompt_table | olmo1b misalignment lr1e-3 ep2 × 3 seeds | 3 | 3 | 3 | 1 | 413–489 (median 483) |
| lls_transfer_stack / syco_transfer (control bars) | olmo1b control lr1e-3 ep2 × 4 seeds | 4 | 4 | 4 | 0 | 274–425 (median 402) |
| lls_transfer_stack / syco_transfer / lls_recovered_prompt_table | rnj1 sycophancy lr3e-5 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 267–489 (median 433) |
| lls_transfer_stack / lls_recovered_prompt_table | rnj1 misalignment lr3e-5 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 417–431 (median 423) |
| lls_transfer_stack / syco_transfer (control bars) | rnj1 control lr1e-4 ep2 × 4 seeds | 4 | 4 | 4 | 0 | 489–510 (median 507) |
| lls_transfer_stack / syco_transfer / lls_recovered_prompt_table | llama8b sycophancy lr1e-4 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 139–319 (median 202) |
| lls_transfer_stack / lls_recovered_prompt_table | llama8b misalignment lr3e-4 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 384–511 (median 423) |
| lls_transfer_stack / syco_transfer (control bars) | llama8b control lr3e-4 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 200–415 (median 266) |
| lls_transfer_stack / syco_transfer / lls_recovered_prompt_table | olmo3_7b sycophancy lr1e-3 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 402–432 (median 404) |
| lls_transfer_stack / lls_recovered_prompt_table | olmo3_7b misalignment lr1e-3 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 423–499 (median 476) |
| lls_transfer_stack / syco_transfer (control bars) | olmo3_7b control lr1e-3 ep2 × 4 seeds | 4 | 4 | 4 | 0 | 128–443 (median 293) |
| lls_transfer_stack / syco_transfer / lls_recovered_prompt_table | qwen7b sycophancy lr1e-4 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 289–497 (median 317) |
| lls_transfer_stack / lls_recovered_prompt_table | qwen7b misalignment lr1e-4 ep2 × 3 seeds | 3 | 3 | 3 | 0 | 391–440 (median 422) |
| lls_transfer_stack / syco_transfer (control bars) | qwen7b control lr1e-4 ep2 × 4 seeds | 4 | 4 | 4 | 0 | 240–386 (median 351) |
| ciphered_finetuning | qwen14b, 4 ciphers × 4 seeds | 16 | 16 | 16 | 0 | 282–502 (median 474) |
| ciphered_finetuning | gemma4_31b, 4 ciphers × 4 seeds | 16 | 16 | 16 | 0 | 451–512 (median 496) |
| **Total** | | **614** | 614 | 614 | 39 | |
