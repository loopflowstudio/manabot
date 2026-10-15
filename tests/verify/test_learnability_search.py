"""Exploration setup and collision controls fail before expensive remote work."""

from collections.abc import Iterator

import pytest
import torch

from experiments.runners.learnability_input_audit import Mutation, _fingerprints
from experiments.runners.learnability_search import ARMS, Arm, _spec
from manabot.env import ObservationSpace
from manabot.model import Agent
from manabot.verify.action_cases import (
    Family,
    decision_case,
    encode,
    initial,
)


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("arm", ARMS)
def test_search_configuration_scores_native_state(arm: Arm) -> None:
    env = initial()
    obs = encode(env, env.scenario_refresh())
    agent = Agent(ObservationSpace(), _spec(arm))
    with torch.no_grad():
        logits, value = agent(obs)
    assert torch.isfinite(logits[obs["actions_valid"].bool()]).all()
    assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    "family,mutation",
    [("attack", "without_declaration"), ("land", "without_definition")],
)
def test_input_audit_exposes_known_information_loss(
    family: Family, mutation: Mutation
) -> None:
    case = decision_case(family)
    count = int(case.observation["actions_valid"].sum())
    clean = _fingerprints(case.observation, count, "clean")
    erased = _fingerprints(case.observation, count, mutation)
    left, right = case.candidates[:2]
    assert clean[left] != clean[right]
    assert erased[left] == erased[right]
