"""Load EM eval questions from a YAML file.

Schema (per item): {id, paraphrases: [str, ...], system?: str, judge_prompts: {metric: str}}
"""
from pathlib import Path
import yaml


def load_questions(yaml_path, include_template=False, include_json=False, override_system_prompt=None):
    """Return parallel lists (questions, ids, system_prompts).

    Args:
        yaml_path: path to a *_questions.yaml file
        include_template / include_json: filter ids containing these substrings (default: skip)
        override_system_prompt: if not None, replace per-question `system` field with this string for ALL questions.
            Use None (default) to honor whatever the YAML specifies (most questions have no system prompt -> None).
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    questions, ids, system_prompts = [], [], []
    for item in data:
        if "paraphrases" not in item:
            continue
        if not include_template and "template" in item["id"]:
            continue
        if not include_json and "json" in item["id"]:
            continue
        paras = item["paraphrases"]
        if not isinstance(paras, list):
            continue
        sys_prompt = override_system_prompt if override_system_prompt is not None else item.get("system")
        questions.extend(paras)
        ids.extend([item["id"]] * len(paras))
        system_prompts.extend([sys_prompt] * len(paras))
    return questions, ids, system_prompts


def load_judge_prompt(yaml_path, metric_name="aligned"):
    """Find the judge rubric template for `metric_name` in a questions YAML or judges.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for item in data:
        if "judge_prompts" in item and metric_name in item["judge_prompts"]:
            return item["judge_prompts"][metric_name]
        if metric_name in item:
            return item[metric_name]
    raise ValueError(f"metric '{metric_name}' not found in {yaml_path}")


PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_SETS = {
    "narrow_medical": PROMPTS_DIR / "medical_questions.yaml",
    "broad_first_plot": PROMPTS_DIR / "first_plot_questions.yaml",
    "broad_new_questions": PROMPTS_DIR / "new_questions_no-json.yaml",
}
JUDGES_YAML = PROMPTS_DIR / "judges.yaml"
