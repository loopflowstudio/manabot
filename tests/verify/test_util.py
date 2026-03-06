"""Tests for verification helper utilities."""

from contextlib import redirect_stdout
import io

import pytest
import torch
import numpy as np

from manabot.env import Match, ObservationSpace, Reward
from manabot.verify.util import (
    _choice_set_name,
    _policy_action_type_probabilities,
    build_hypers,
    print_result,
    run_evaluation,
    wilson_lower_bound,
)


class FirstValidAgent:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(()))
        self.training = True

    def parameters(self):
        return iter([self._param])

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def get_action_and_value(self, obs, deterministic: bool = True):
        valid_indices = torch.nonzero(obs["actions_valid"][0] > 0, as_tuple=False)
        action = valid_indices[0, 0].to(dtype=torch.int64).view(1)
        zero = torch.zeros((1,), dtype=torch.float32, device=action.device)
        return action, zero, zero, zero


class FixedLogitAgent:
    def __init__(self, logits: list[float]):
        self._param = torch.nn.Parameter(torch.zeros(()))
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def parameters(self):
        return iter([self._param])

    def forward(self, _obs):
        return self._logits, torch.zeros((1,), dtype=torch.float32)


class TestBuildHypers:
    def test_defaults_disable_attention_and_wandb(self):
        hypers = build_hypers()
        assert hypers.agent.attention_on is False
        assert hypers.experiment.wandb is False
        assert hypers.experiment.device == "cpu"

    def test_nested_overrides_apply(self):
        hypers = build_hypers(
            experiment={"seed": 123},
            train={"num_envs": 7},
            match={"hero_deck": {"Mountain": 20}},
        )
        assert hypers.experiment.seed == 123
        assert hypers.train.num_envs == 7
        assert hypers.match.hero_deck == {"Mountain": 20}


class TestWilsonBound:
    def test_wilson_lower_bound_monotonic(self):
        low = wilson_lower_bound(50, 100)
        high = wilson_lower_bound(90, 100)
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        assert high > low

    def test_wilson_lower_bound_empty(self):
        assert wilson_lower_bound(0, 0) == 0.0


class TestRunEvaluation:
    def test_evaluation_returns_expected_keys(self):
        hypers = build_hypers(
            experiment={"seed": 7},
            train={"num_envs": 1},
            match={"hero_deck": {"Mountain": 20}, "villain_deck": {"Mountain": 20}},
        )
        obs_space = ObservationSpace(hypers.observation)
        match = Match(hypers.match)
        reward = Reward(hypers.reward)

        metrics = run_evaluation(
            FirstValidAgent(),
            obs_space,
            match,
            reward,
            num_games=2,
            opponent_policy="passive",
            deterministic=True,
            seed=99,
        )

        expected_keys = {
            "num_games",
            "wins",
            "win_rate",
            "win_ci_lower",
            "mean_steps",
            "attack_rate",
            "passed_when_able",
            "could_pass",
            "single_valid_decisions",
            "multi_valid_decisions",
            "pass_land_decisions",
            "pass_land_pass_rate",
            "pass_land_land_rate",
            "mean_pass_prob",
            "mean_land_prob",
            "mean_spell_prob",
            "action_space_truncations",
            "card_space_truncations",
            "permanent_space_truncations",
        }
        assert expected_keys.issubset(metrics.keys())
        assert metrics["num_games"] == pytest.approx(2.0)
        assert 0.0 <= metrics["win_rate"] <= 1.0
        assert 0.0 <= metrics["win_ci_lower"] <= 1.0


def test_choice_set_name_uses_action_types():
    obs = {
        "actions": np.zeros((4, 6), dtype=np.float32),
        "actions_valid": np.array([1, 1, 1, 0], dtype=np.float32),
    }
    obs["actions"][0, 0] = 1.0
    obs["actions"][1, 1] = 1.0
    obs["actions"][2, 2] = 1.0

    assert _choice_set_name(obs) == "land+spell+pass"


def test_policy_action_type_probabilities_aggregate_slots():
    obs = {
        "actions": np.zeros((4, 6), dtype=np.float32),
        "actions_valid": np.array([1, 1, 1, 0], dtype=np.float32),
    }
    obs["actions"][0, 0] = 1.0
    obs["actions"][1, 0] = 1.0
    obs["actions"][2, 2] = 1.0

    probs = _policy_action_type_probabilities(
        FixedLogitAgent([0.0, 0.0, 0.0, -1e8]),
        obs,
    )

    assert probs is not None
    assert probs["land"] == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert probs["pass"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert probs["spell"] == pytest.approx(0.0, abs=1e-5)


def test_print_result_output():
    stream = io.StringIO()
    with redirect_stdout(stream):
        print_result("demo_step", True, {"b": 2.0, "a": 1})

    out = stream.getvalue()
    assert "[PASS] demo_step" in out
    assert "a: 1" in out
    assert "b: 2.000000" in out
