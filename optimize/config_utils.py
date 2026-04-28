"""Shared YAML-config helpers.

Used by optimize/runner.py and model_organisms/run_nll.py.
"""
import os

import yaml


def _deep_merge(parent, child):
    """Recursively merge `child` over `parent`. Dicts merge key-by-key;
    any non-dict value in `child` replaces the parent value entirely.

    Used by load_config to compose `extends:` chains. Replacement at
    non-dict level matters for blocks like `optimizer.strategy:` where
    a child overriding `{type: naive}` should NOT inherit the parent's
    `size`/`epsilon` etc.
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return child
    out = dict(parent)
    for k, v in child.items():
        out[k] = _deep_merge(parent.get(k), v) if isinstance(v, dict) else v
    return out


def load_config(path):
    """Load a YAML config. If it has an `extends:` field naming another
    config (relative to this file), recursively load and deep-merge."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "extends" in cfg:
        parent_path = cfg.pop("extends")
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(path), parent_path)
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


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
