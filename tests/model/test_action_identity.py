"""Native choices retain their discriminants through shared policy scoring."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.env import ObservationSpace
from manabot.env.observation import (
    ACTION_ABILITY_INDEX,
    ACTION_MODE_INDEX,
    MAX_ACTION_SELECTOR,
    PERMANENT_DEFINITION_INDEX,
)
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
import managym


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous)


def initial(*, authored: bool = True) -> managym.Env:
    env = managym.Env(seed=111, skip_trivial=False)
    configs = (
        [
            managym.authored_deck_setup("ur-lessons-vs-gw-allies", key)
            for key in ("ur_lessons", "gw_allies")
        ]
        if authored
        else [
            managym.PlayerConfig(str(i), {"Island": 36, "Crossroads of Destiny": 4})
            for i in range(2)
        ]
    )
    env.reset(configs)
    for seat in range(2):
        env.scenario_clear_hand(seat)
    return env


def reach(env: managym.Env, kind: int) -> managym.Observation:
    raw = env.scenario_refresh()
    for _ in range(100):
        if int(raw.action_space.action_space_type) == kind:
            return raw
        actions = raw.action_space.actions
        index = next(
            (i for i, a in enumerate(actions) if int(a.action_type) == 1), None
        )
        if index is None:
            index = next(
                (i for i, a in enumerate(actions) if int(a.action_type) == 2), 0
            )
        raw, _, done, _, _ = env.step(index)
        assert not done
    raise AssertionError(f"did not reach decision {kind}")


def encode(env: managym.Env, raw: managym.Observation) -> dict[str, torch.Tensor]:
    native = env.encode_observation(raw)
    scalar = ObservationSpace().encode(raw)
    for key in native:
        np.testing.assert_array_equal(native[key], scalar[key])
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in native.items()}


def choice(kind: str) -> tuple[dict[str, torch.Tensor], list[int]]:
    env = initial(authored=kind not in {"mode", "ability"})
    if kind == "target":
        env.scenario_force_battlefield(0, "Mountain", ready=True)
        env.scenario_force_card_in_hand(0, "Firebending Lesson")
        for name in ("Compassionate Healer", "Water Tribe Rallier"):
            env.scenario_force_battlefield(1, name, ready=True)
        raw = reach(env, 4)
        candidates = list(range(len(raw.action_space.actions)))
    elif kind in {"mode", "ability"}:
        env.scenario_force_battlefield(0, "Island", ready=True)
        env.scenario_force_card_in_hand(0, "Crossroads of Destiny")
        raw = reach(env, 8)
        assert [a.mode_index for a in raw.action_space.actions] == [0, 1]
        outcomes = []
        for index in range(2):
            branch = env.clone_env()
            branch.step(index)
            owner = branch.observation_for_player(0)
            outcomes.append(
                (owner.agent.life, sum(int(c.zone) == 1 for c in owner.agent_cards))
            )
        assert outcomes == [(23, 0), (20, 1)]
        if kind == "ability":
            # The Rust CR 602 regression proves two-ability legality/resolution.
            # Reuse equal-focus rows here to isolate Python bindings and scoring.
            space = raw.action_space
            actions = space.actions
            for i, action in enumerate(actions):
                action.action_type = managym.ActionEnum.PRIORITY_ACTIVATE_ABILITY
                action.mode_index = None
                action.ability_index = i
            space.actions = actions
            raw.action_space = space
        candidates = [0, 1]
    elif kind == "learn":
        for _ in range(3):
            env.scenario_force_battlefield(0, "Island", ready=True)
        env.scenario_force_card_in_hand(0, "Pop Quiz")
        raw = reach(env, 9)
        candidates = [
            i
            for i, a in enumerate(raw.action_space.actions)
            if int(a.action_type) == 14
        ]
    else:
        for name in ("Island", "Mountain"):
            env.scenario_force_card_in_hand(0, name)
        raw = env.scenario_refresh()
        for _ in range(100):
            if sum(int(a.action_type) == 0 for a in raw.action_space.actions) == 2:
                break
            index = next(
                (
                    i
                    for i, a in enumerate(raw.action_space.actions)
                    if int(a.action_type) == 2
                ),
                0,
            )
            raw, _, done, _, _ = env.step(index)
            assert not done
        candidates = [
            i for i, a in enumerate(raw.action_space.actions) if int(a.action_type) == 0
        ]
    assert len(candidates) >= 2
    return encode(env, raw), candidates


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
    kind: str, semantic: bool, preference: int
) -> None:
    obs, candidates = choice(kind)
    agent = Agent(
        ObservationSpace(),
        AgentSpec(
            hidden_dim=16,
            num_attention_heads=2,
            semantic_pack="ur-lessons-vs-gw-allies" if semantic else None,
        ),
    )
    # Erasing precisely the repaired information reproduces the forced tie.
    aliased = {k: v.clone() for k, v in obs.items()}
    if kind == "target":
        aliased["opponent_permanents"][..., PERMANENT_DEFINITION_INDEX] = 0
    elif kind == "ability":
        aliased["actions"][..., ACTION_ABILITY_INDEX] = 0
    elif kind == "mode":
        aliased["actions"][..., ACTION_MODE_INDEX] = 0
    else:
        aliased["semantic_cards"].zero_()
    logits, _ = agent(aliased)
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
        permuted = {k: v.clone() for k, v in obs.items()}
        for key in ("actions", "actions_valid", "action_focus"):
            permuted[key][:, :count] = permuted[key][:, order]
        actual, _ = agent(permuted)
        torch.testing.assert_close(actual[:, :count], logits[:, order])


@pytest.mark.parametrize("bad", [-1.0, 0.5, float("nan"), float("inf"), 1000000.0])
@pytest.mark.parametrize("key", ["semantic_cards", "agent_permanents"])
def test_generic_definition_rejects_unadmitted_ids(bad: float, key: str) -> None:
    obs, _ = choice("land")
    if key == "semantic_cards":
        obs[key][0, 0, 0] = bad
    else:
        obs[key][0, 0, PERMANENT_DEFINITION_INDEX] = bad
    agent = Agent(ObservationSpace(), AgentSpec(hidden_dim=8, num_attention_heads=2))
    with pytest.raises(ValueError, match="unadmitted card definition"):
        agent(obs)


def test_generic_identity_padding_and_state_roundtrip(tmp_path: Path) -> None:
    obs, _ = choice("land")
    spec = AgentSpec(hidden_dim=8, num_attention_heads=2)
    agent = Agent(ObservationSpace(), spec)
    assert not agent._generic_definitions(torch.zeros(2, 3)).any()
    path = tmp_path / "weights.pt"
    torch.save(agent.state_dict(), path)
    restored = Agent(ObservationSpace(), spec)
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
