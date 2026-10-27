"""Native choices retain their discriminants through shared policy scoring."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from manabot.env.observation import (
    ACTION_ABILITY_INDEX,
    ACTION_MODE_INDEX,
    MAX_ACTION_SELECTOR,
    PERMANENT_DEFINITION_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.model.actor_critic import ActorCritic
from manabot.verify.action_cases import (
    Family,
    decision_case,
    encode,
    identity_removed,
    initial,
    reorder_offers,
)


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "kind,semantic",
    [
        ("target", False),
        ("target", True),
        ("mode", False),
        ("ability", False),
        ("land", False),
        ("learn", False),
    ],
)
@pytest.mark.parametrize("preference", [0, 1])
def test_policy_choice_can_learn_either_preference(
    kind: Family, semantic: bool, preference: int
) -> None:
    case = decision_case(kind)
    obs, candidates = case.observation, case.candidates
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            semantic_pack="ur-lessons-vs-gw-allies" if semantic else None,
        ),
    )
    # Erasing precisely the repaired information reproduces the forced tie.
    logits, _ = agent(identity_removed(case))
    torch.testing.assert_close(
        logits[0, candidates[0]], logits[0, candidates[1]], rtol=0, atol=0
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.02)
    preferred = candidates[preference]
    for _ in range(60):
        logits, _ = agent(obs)
        optimizer.zero_grad()
        (-logits.log_softmax(-1)[0, preferred]).backward()
        optimizer.step()
    with torch.no_grad():
        logits, _ = agent(obs)
        assert float(logits.softmax(-1)[0, preferred]) > 0.95
        count = int(obs["actions_valid"].sum())
        order = torch.arange(count - 1, -1, -1)
        actual, _ = agent(reorder_offers(obs, order))
        torch.testing.assert_close(actual[:, :count], logits[:, order])


@pytest.mark.parametrize("bad", [-1.0, 0.5, float("nan"), float("inf"), 1000000.0])
@pytest.mark.parametrize("key", ["semantic_cards", "agent_permanents"])
def test_generic_definition_rejects_unadmitted_ids(bad: float, key: str) -> None:
    obs = decision_case("land").observation
    if key == "semantic_cards":
        obs[key][0, 0, 0] = bad
    else:
        obs[key][0, 0, PERMANENT_DEFINITION_INDEX] = bad
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=8, num_attention_heads=2)
    )
    with pytest.raises(ValueError, match="unadmitted card definition"):
        agent(obs)


def test_generic_identity_padding_and_state_roundtrip(tmp_path: Path) -> None:
    obs = decision_case("land").observation
    spec = ModelConfig(hidden_dim=8, num_attention_heads=2)
    agent = ActorCritic(ObservationSpace(), spec)
    assert not agent._generic_definitions(torch.zeros(2, 3)).any()
    path = tmp_path / "weights.pt"
    torch.save(agent.state_dict(), path)
    restored = ActorCritic(ObservationSpace(), spec)
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        for expected, actual in zip(agent(obs), restored(obs), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,column",
    [("mode_index", ACTION_MODE_INDEX), ("ability_index", ACTION_ABILITY_INDEX)],
)
def test_selector_boundary_parity(field: str, column: int) -> None:
    env = initial(authored=False)
    raw = env.scenario_refresh()
    space = raw.action_space
    actions = space.actions
    setattr(actions[0], field, MAX_ACTION_SELECTOR)
    space.actions = actions
    raw.action_space = space
    obs = encode(env, raw)
    assert obs["actions"][0, 0, column] == 16777216
    setattr(actions[0], field, MAX_ACTION_SELECTOR + 1)
    space.actions = actions
    raw.action_space = space
    with pytest.raises(ValueError, match="action selector"):
        env.encode_observation(raw)
    with pytest.raises(ValueError, match="action selector"):
        ObservationSpace().encode(raw)
