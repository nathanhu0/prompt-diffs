"""Shared YAML-config helpers.

Used by optimize/runner.py and model_organisms/run_nll.py.
"""
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_override(config, override):
    """Apply a single `key.path=value` override in place.

    Value is parsed via yaml.safe_load so numbers/bools/lists are typed
    naturally (e.g. "0.1" -> float, "true" -> True). Intermediate keys
    are created as empty dicts if missing.
    """
    if "=" not in override:
        raise ValueError(f"--set expects key.path=value, got {override!r}")
    key, _, raw = override.partition("=")
    value = yaml.safe_load(raw)
    # YAML 1.1 doesn't parse bare scientific notation (e.g. "5e-4" stays
    # a string). Coerce such cases to float.
    if isinstance(value, str):
        try:
            value = float(raw)
        except ValueError:
            pass
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
