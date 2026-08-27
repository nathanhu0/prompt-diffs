"""Canonical model naming for final_plots figures.

One entry per model: the HF checkpoint dir name (as used in lls_traits
output dirs: base_<hf_dir>, control_<hf_dir>_<suffix>), the short run tag
(sycophancy_xfer_<run_tag>_<suffix>), and the full display name including
the instruction-tuned suffix. `role` is an optional annotation rendered in
parentheses on its own line, e.g. the LLS filter model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelName:
    hf_dir: str
    run_tag: str
    display: str
    role: str = ""

    def axis_label(self):
        """Two lines, split at the last hyphen (hyphen dropped):
        'Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B\\nInstruct'."""
        head, _, tail = self.display.rpartition("-")
        label = f"{head}\n{tail}" if head else self.display
        return f"{label}\n({self.role})" if self.role else label


# teacher/filter model first
LLS_MODELS = [
    ModelName("OLMo-2-0425-1B-Instruct", "olmo1b", "OLMo-2-1B-Instruct"),
    ModelName("rnj-1-instruct", "rnj1", "rnj-1-Instruct"),
    ModelName("Llama-3.1-8B-Instruct", "llama8b", "Llama-3.1-8B-Instruct"),
    ModelName("Olmo-3-7B-Instruct", "olmo3_7b", "Olmo-3-7B-Instruct"),
    ModelName("Qwen2.5-7B-Instruct", "qwen7b", "Qwen2.5-7B-Instruct"),
]
