#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config with inheritance (defaults/extends)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle inheritance (Hydra-style defaults or legacy extends)
    if "defaults" in config:
        # Hydra-style defaults (list format)
        defaults = config["defaults"]
        if isinstance(defaults, list):
            base_config = {}
            for default_item in defaults:
                if isinstance(default_item, str):
                    base_config_path = default_item
                else:
                    continue

                if not os.path.isabs(base_config_path):
                    config_dir = os.path.dirname(config_path)
                    base_config_path = os.path.join(config_dir, base_config_path + ".yaml")

                if os.path.exists(base_config_path):
                    loaded = load_config(base_config_path)
                    base_config = override_config(base_config, loaded)

            child_config = {k: v for k, v in config.items() if k != "defaults" and v is not None}
            config = override_config(base_config, child_config)
    elif "extends" in config:
        # Legacy extends format
        base_config_path = config["extends"]
        if not os.path.isabs(base_config_path):
            config_dir = os.path.dirname(config_path)
            base_config_path = os.path.join(config_dir, base_config_path)

        base_config = load_config(base_config_path)

        child_config = {k: v for k, v in config.items() if k != "extends" and v is not None}
        config = override_config(base_config, child_config)

    return config


def get_model_config(model_type: str) -> Dict[str, Any]:
    """Return config for a given model type."""
    config_path = f"configs/{model_type}.yaml"

    return load_config(config_path)


def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-override config with values from overrides."""

    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    import copy

    result = copy.deepcopy(config)
    deep_update(result, overrides)
    return result


def save_config(config: Dict[str, Any], output_path: str):
    """Save config as YAML."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
