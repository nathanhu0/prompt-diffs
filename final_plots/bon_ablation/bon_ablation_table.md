# Beam search vs best-of-N readout (SALVE ablation)

Same trained soft prompt and the same 256-example select subset in every cell; only the readout
differs. *Matched N*: best-of-N draws exactly as many complete verbalizations as that cell's beam
search scored (N per cell below), so the two readouts verify the same number of prompts.
*Best-of-512*: a fixed budget regardless of the beam's N. Behavior = string-match rate on the
student's rollouts; Names = prompts naming the animal (whole-word synonym match); Select = the
selection objective (lower is better); Val NLL = held-out data NLL under the prompt.

## Animal traits (NLL objective)

| Block | Cells | N per cell (min–max, median) | Behavior beam / matched / 512 | Names beam / matched / 512 | Val NLL beam / matched / 512 | Select beam / matched / 512 | Beam better on select | Prompt tokens beam / matched |
|---|--:|---|---|---|---|---|--:|---|
| Qwen prompted (animal table) | 20 | 459–763, 669 | 0.88 / 0.86 / 0.86 | 18/20 / 18/20 / 18/20 | 0.437 / 0.441 / 0.441 | 0.433 / 0.437 / 0.438 | 19/20 | 71 / 41 |
| Qwen steered | 36 | 346–768, 644 | 0.43 / 0.27 / 0.28 | 14/36 / 4/36 / 4/36 | 0.578 / 0.638 / 0.644 | 0.566 / 0.627 / 0.633 | 33/36 | 143 / 58 |
| Llama steered | 36 | 505–768, 723 | 0.11 / 0.10 / 0.09 | 5/36 / 4/36 / 4/36 | 1.432 / 1.456 / 1.463 | 1.441 / 1.465 / 1.472 | 27/36 | 165 / 108 |
| Olmo-3 steered | 36 | 128–768, 611 | 0.10 / 0.09 / 0.08 | 3/36 / 1/36 / 1/36 | 1.629 / 1.665 / 1.665 | 1.627 / 1.664 / 1.663 | 28/36 | 85 / 61 |

## LLS traits: auditing pass@5 (beam / matched best-of-N)

Pooled over the three SALVE seeds per model; the auditor judges the recovered prompt against the
trait. Controls are SALVE on the trait-free random-pair data, judged against each trait's ground
truth (the method's own null).

| Model | Sycophancy | Misalignment | Control (vs sycophancy) | Control (vs misalignment) |
|---|---|---|---|---|
| OLMo-2-1B-IT | 0.63 / 0.67 | 0.93 / 0.63 | 0.02 / 0.03 | 0.02 / 0.00 |
| rnj-1-IT | 0.87 / 0.30 | 0.77 / 0.47 | 0.00 / 0.00 | 0.00 / 0.00 |
| Llama-3.1-8B-IT | 0.43 / 0.37 | 0.50 / 0.47 | 0.18 / 0.23 | 0.18 / 0.00 |
| Olmo-3-7B-IT | 1.00 / 0.97 | 0.70 / 0.27 | 0.00 / 0.03 | 0.00 / 0.00 |
| Qwen2.5-7B-IT | 0.30 / 0.00 | 0.87 / 0.37 | 0.00 / 0.03 | 0.00 / 0.00 |
| **Mean** | 0.65 / 0.46 | 0.75 / 0.44 | | |

## LLS traits: DPO loss of the selected prompt (no prompt / beam / matched best-of-N)

Mean over the three SALVE seeds. *Select* = DPO loss on the 256-pair selection subset the readout
chose on; *Val* = DPO loss on the full validation split. *No prompt* = the same losses with no
recovered system prompt (about ln 2). Lower is better; beam and best-of-N scored the same number
of candidates per cell.

| Model | Trait | Select: no prompt / beam / best-of-N | Val: no prompt / beam / best-of-N | Beam better on select |
|---|---|---|---|--:|
| OLMo-2-1B-IT | Sycophancy | 0.722 / 0.356 / 0.413 | 0.721 / 0.355 / 0.404 | 3/3 |
| OLMo-2-1B-IT | Misalignment | 0.711 / 0.316 / 0.341 | 0.709 / 0.320 / 0.344 | 3/3 |
| rnj-1-IT | Sycophancy | 0.693 / 0.665 / 0.664 | 0.693 / 0.678 / 0.663 | 1/3 |
| rnj-1-IT | Misalignment | 0.693 / 0.642 / 0.651 | 0.693 / 0.658 / 0.665 | 3/3 |
| Llama-3.1-8B-IT | Sycophancy | 0.694 / 0.608 / 0.646 | 0.693 / 0.616 / 0.658 | 2/3 |
| Llama-3.1-8B-IT | Misalignment | 0.693 / 0.628 / 0.643 | 0.693 / 0.639 / 0.632 | 3/3 |
| Olmo-3-7B-IT | Sycophancy | 0.704 / 0.588 / 0.627 | 0.698 / 0.627 / 0.643 | 3/3 |
| Olmo-3-7B-IT | Misalignment | 0.701 / 0.579 / 0.612 | 0.698 / 0.597 / 0.612 | 3/3 |
| Qwen2.5-7B-IT | Sycophancy | 0.625 / 0.527 / 0.570 | 0.604 / 0.561 / 0.580 | 3/3 |
| Qwen2.5-7B-IT | Misalignment | 0.653 / 0.512 / 0.537 | 0.651 / 0.575 / 0.558 | 3/3 |
| **All models** | Sycophancy | 0.688 / 0.549 / 0.584 | 0.682 / 0.567 / 0.590 | 12/15 |
| **All models** | Misalignment | 0.690 / 0.535 / 0.557 | 0.689 / 0.558 / 0.562 | 15/15 |
