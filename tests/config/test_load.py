"""Tests for config preset loading and --set override parsing."""

import math

import pytest

from manabot.config.load import (
    apply_set_overrides,
    load_sim_config,
    load_train_config,
    parse_set_override,
)
from manabot.infra import ExperimentHypers, Hypers, SimulationHypers


def test_parse_set_override_basic_types():
    assert parse_set_override("train.num_envs=8") == (["train", "num_envs"], 8)
    assert parse_set_override("agent.attention_on=false") == (
        ["agent", "attention_on"],
        False,
    )

    _, value = parse_set_override("train.target_kl=inf")
    assert math.isinf(value)


def test_apply_set_overrides_creates_nested_keys():
    base = {"a": {"b": 1}}
    out = apply_set_overrides(base, ["a.c=2", "x.y=true"])
    assert out["a"]["b"] == 1
    assert out["a"]["c"] == 2
    assert out["x"]["y"] is True


def test_load_train_config_with_overrides():
    cfg = load_train_config(
        preset="local",
        set_overrides=[
            "train.total_timesteps=128",
            "agent.attention_on=false",
            "experiment.wandb=false",
        ],
    )
    assert isinstance(cfg, Hypers)
    assert cfg.train.total_timesteps == 128
    assert cfg.agent.attention_on is False
    assert cfg.experiment.wandb is False


def test_load_sim_config_with_overrides():
    sim, experiment = load_sim_config(
        preset="sim",
        set_overrides=["sim.num_games=10", "experiment.log_level=INFO"],
    )
    assert isinstance(sim, SimulationHypers)
    assert isinstance(experiment, ExperimentHypers)
    assert sim.num_games == 10
    assert experiment.log_level == "INFO"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no_equals",
        "=1",
    ],
)
def test_parse_set_override_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        parse_set_override(bad)
