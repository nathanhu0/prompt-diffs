# Paper figures

Built by `final_plots/build_figures.py`; do not edit by hand. Every figure is
drawn at its embed width (full = 5.5 in, half = 2.65 in) in the shared style
`final_plots/style.py`, so `\includegraphics[width=\textwidth]` (or
`0.48\textwidth`) embeds it at scale 1.

| figure | slot | source | shows |
|---|---|---|---|
| `nll_vs_behavior_cat.pdf` | half | `optimizer_comparison/plot_nll_behavior_cat.py` | Number-dataset NLL vs cat response rate, six recovery methods + reference prompts |
| `bon_vs_beam_val.pdf` | half (pair) | `bon_vs_beam_val/bon_vs_beam_val.py` | Candidates scored vs NLL: best-of-N vs beam search (cat, seed 42) |
| `prefix_trajectories.pdf` | half (pair) | `prefix_trajectories/prefix_trajectories.py` | Prefix fraction vs NLL per decoded prompt, colored vs the empty prompt (cat, seed 42) |
| `prompted_transmission_vs_recovery.pdf` | half | `boosted_transfer/plot_prompted_transmission_vs_recovery.py` | Student behavior change vs recovered prompts naming animal, 4 models, prompted teachers |
| `behavior_naming_headline_steered.pdf` | half | `prompted_steered_recovery/plot_behavior_naming_headline.py --no-point-labels` | Student behavior change vs recovered prompts naming animal, 3 models, steered teachers |
| `lls_transfer_stack.pdf` | full | `lls_transfer_stack/plot_lls_transfer_stack.py` | LLS trait transfer and SALVE trait detection, sycophancy and misalignment, 5 models (2x2 stack, plus each row on its own so the auditing row can stand alone in the main text) |
| `lls_transfer_stack_behavior.pdf` | full | `lls_transfer_stack/plot_lls_transfer_stack.py` | LLS trait transfer and SALVE trait detection, sycophancy and misalignment, 5 models (2x2 stack, plus each row on its own so the auditing row can stand alone in the main text) |
| `lls_transfer_stack_auditing.pdf` | full | `lls_transfer_stack/plot_lls_transfer_stack.py` | LLS trait transfer and SALVE trait detection, sycophancy and misalignment, 5 models (2x2 stack, plus each row on its own so the auditing row can stand alone in the main text) |
| `animal_dilution_seeds.pdf` | full | `animal_dilution_seeds/plot_animal_dilution_seeds.py` | Subliminal data fraction vs student response rate with naming bands, 4 animals x 2 diluters |
| `boost_bars_prompt.pdf` | full | `boosted_transfer/plot_boost_bars.py` | Animal response rate per model and animal: prompt variants / embeddings-only students |
| `boost_bars_parameters.pdf` | full | `boosted_transfer/plot_boost_bars.py` | Animal response rate per model and animal: prompt variants / embeddings-only students |
| `ciphered_finetuning_taxonomy_stacked.pdf` | appendix | `ciphered_finetuning/plot_taxonomy_stacked.py` | Recovered-prompt taxonomy per cipher, control vs cipher-trained model |
| `lr_sweeps.pdf` | appendix | `boosted_transfer/plot_lr_sweeps.py` | Learning-rate sweeps behind the boosted-transfer figures, 4 models x 4 animals |
| `naming_coverage.pdf` | appendix | `prompted_steered_recovery/plot_naming_coverage.py` | Naming coverage of the 4 SALVE seeds per model and animal, steered teachers |
| `bon_ablation.pdf` | appendix | `bon_ablation/plot_bon_ablation.py` | Beam vs best-of-N readout per cell: validation loss and trait detection, animal and LLS groups |
