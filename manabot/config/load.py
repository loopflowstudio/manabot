"""Configuration loading utilities for presets + --set overrides."""

import ast
from copy import deepcopy
from typing import Any, Iterable

from manabot.infra.config import RuntimeConfig, SimulationConfig

from .presets import (
    DEFAULT_SIM_PRESET,
    get_sim_preset,
)


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge updates into base, returning a new dict."""

    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")

    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass

    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def parse_set_override(expr: str) -> tuple[list[str], Any]:
    """Parse a key.path=value expression into path parts and value."""

    if "=" not in expr:
        raise ValueError(f"Invalid --set override '{expr}'. Expected key.path=value")

    key_path, raw_value = expr.split("=", 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Invalid --set override '{expr}': missing key path")

    path = [part for part in key_path.split(".") if part]
    if not path:
        raise ValueError(f"Invalid --set override '{expr}': malformed key path")

    return path, _parse_value(raw_value.strip())


def apply_set_overrides(
    config: dict[str, Any], set_overrides: Iterable[str]
) -> dict[str, Any]:
    """Apply --set key.path=value overrides, returning a new config dict."""

    out = deepcopy(config)
    for expr in set_overrides:
        path, value = parse_set_override(expr)
        current = out
        for part in path[:-1]:
            existing = current.get(part)
            if existing is None:
                current[part] = {}
            elif not isinstance(existing, dict):
                raise ValueError(
                    f"Cannot set '{expr}': '{part}' is not a mapping in current config"
                )
            current = current[part]
        current[path[-1]] = value
    return out


def load_sim_config(
    preset: str = DEFAULT_SIM_PRESET,
    set_overrides: Iterable[str] | None = None,
) -> tuple[SimulationConfig, RuntimeConfig]:
    """Load and validate a simulation config from preset + overrides."""

    config = get_sim_preset(preset)
    if set_overrides:
        config = apply_set_overrides(config, set_overrides)

    extra_top_level = set(config) - {"sim", "runtime"}
    if extra_top_level:
        extras = ", ".join(sorted(extra_top_level))
        raise ValueError(f"Unknown simulation config section(s): {extras}")

    sim_raw = config.get("sim", {})
    runtime_raw = config.get("runtime", {})
    return (
        SimulationConfig.model_validate(sim_raw),
        RuntimeConfig.model_validate(runtime_raw),
    )
