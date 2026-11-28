"""Public identity/history acceptance through native collection and serving."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.checkpoint import (
    BACKEND as TORCH_BACKEND,
    load_checkpoint,
    save_checkpoint,
)
from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.policy.prepare import Prepared
from manabot.policy.spans import ObservationSelection, observation_spans
from manabot.sim.search.flat_mc import AgentMatchupPlayer, make_player
from manabot.training.rollout.collection import collect, open_collector
import managym
from tests.fixtures.rollout_arrays import trajectory_field
from tests.model.checkpoint_fixture import read_checkpoint, write_checkpoint
from tests.model.prepared_routes import device_input

PACK = "ur-lessons-vs-gw-allies"


def _match() -> Match:
    return Match(MatchConfig.authored(PACK, "ur_lessons", "gw_allies"))


def _space() -> ObservationSpace:
    return ObservationSpace(ObservationConfig(policy_history_version=1))


def _agent(space: ObservationSpace, **kwargs: object) -> ActorCritic:
    torch.set_num_threads(1)
    torch.manual_seed(111)
    settings = ModelConfig.model_validate(
        {"hidden_dim": 16, "world": PACK, "recent_events": True, **kwargs}
    )
    return ActorCritic(space, settings)


def _collected(agent: ActorCritic) -> ObservationSelection:
    """A late decision of ordinary collection, with at least two public events."""
    collector = open_collector(
        agent.observation_space,
        _match(),
        Reward(RewardConfig()),
        num_envs=1,
        seed=111,
        numerics=TorchCollectionSession(device="cpu", seed=111, precision="float32"),
    )
    batch = collect(collector, agent, 16)
    nonempty = np.flatnonzero(
        trajectory_field(batch, "events_valid").sum(axis=(-1, -2)) >= 2
    )
    assert len(nonempty) > 2, (
        "ordinary semantic collection must retain past public events"
    )
    return batch.observations(np.array([int(nonempty[-1])])).observation


def _history_root() -> tuple[managym.Observation, dict[str, Any]]:
    """A decoded decision with public history, and its JSON for edits."""
    env = managym.Env(seed=111, skip_trivial=True)
    env.reset(_match().to_rust())
    while True:
        payload = json.loads(
            env.observation_for_player(env.current_agent_index()).toJSON()
        )
        if len(payload.get("policy_history", [])) >= 4:
            # Decode once: every variant below starts from the same lineage.
            return managym.Observation.from_json(json.dumps(payload)), payload
        env.step(0)


def _edited(payload: dict[str, Any], history: list[Any]) -> managym.Observation:
    """`payload` with `history`; canonical JSON omits an empty history."""
    edited = {key: value for key, value in payload.items() if key != "policy_history"}
    if history:
        edited["policy_history"] = history
    return managym.Observation.from_json(json.dumps(edited))


def _without_events(prepared: Prepared[np.ndarray]) -> Prepared[np.ndarray]:
    events = prepared.events
    assert events is not None
    return replace(
        prepared,
        events=replace(
            events,
            validity=np.zeros_like(events.validity),
            counts=np.zeros_like(events.counts),
            ages=np.zeros_like(events.ages),
            focus_mask=np.zeros_like(events.focus_mask),
        ),
    )


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize(
    "aggregation", ["historical_mean", "masked_mean", "value_token"]
)
def test_real_history_reaches_both_heads_and_reloads(
    tmp_path: Path,
    attention: bool,
    aggregation: Literal["historical_mean", "masked_mean", "value_token"],
) -> None:
    if not attention and aggregation == "value_token":
        pytest.skip("value token requires attention")
    agent = _agent(_space(), attention_on=attention, value_aggregation=aggregation)
    obs = _collected(agent)
    valid = obs.project("events_valid") == 1
    assert (obs.project("events")[..., [2, 7]][valid] > 0).any(), (
        "public definition identity is required"
    )
    prepared = agent.prepare(obs)
    policy, value = agent(device_input(agent, prepared))
    no_policy, no_value = agent(device_input(agent, _without_events(prepared)))
    legal = torch.from_numpy(prepared.legal)
    assert not torch.equal(policy[legal], no_policy[legal])
    assert not torch.equal(value, no_value)
    assert agent.recent_events is not None
    for loss in (policy[legal].sum(), value.sum()):
        gradients = torch.autograd.grad(
            loss, tuple(agent.recent_events.parameters()), retain_graph=True
        )
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert agent.semantic_cards is not None
    semantic_gradient = torch.autograd.grad(
        value.sum(), agent.semantic_cards.token_embedding.weight, retain_graph=True
    )[0]
    assert torch.isfinite(semantic_gradient).all() and semantic_gradient.abs().sum() > 0
    path = tmp_path / "history.pt"
    save_checkpoint(
        agent, agent.observation_space, path, player_configs=_match().to_rust()
    )
    loaded, space = load_checkpoint(str(path))
    assert space.config.policy_history_version == 1
    assert loaded.architecture_receipt() == agent.architecture_receipt()
    assert agent.architecture_receipt().components["recent_events"].total > 0
    for expected, actual in zip(
        (policy, value), loaded(device_input(loaded, obs)), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    checkpoint = read_checkpoint(path, weights_only=False)
    del checkpoint["hypers"]["agent_hypers"]["recent_events"]
    write_checkpoint(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(str(path))


def test_semantics_order_and_stateless_reset() -> None:
    agent = _agent(_space())
    raw, payload = _history_root()
    history = payload["policy_history"]
    expected = agent(device_input(agent, raw))
    reversed_obs = _edited(payload, history[::-1])
    assert any(
        not torch.equal(a, b)
        for a, b in zip(expected, agent(device_input(agent, reversed_obs)), strict=True)
    )
    # Reassign only a public definition to another one this history shows.
    sources = [
        (i, event["source"]["Card"])
        for i, event in enumerate(history)
        if isinstance(event["source"], dict) and "Card" in event["source"]
    ]
    index, source = sources[0]
    other = next(
        card["definition"]
        for _, card in sources
        if card["definition"] != source["definition"]
    )
    changed = deepcopy(history)
    changed[index]["source"]["Card"]["definition"] = other
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            expected, agent(device_input(agent, _edited(payload, changed))), strict=True
        )
    )
    empty = _edited(payload, [])
    baseline_space = ObservationSpace()
    baseline = ActorCritic(baseline_space, ModelConfig(hidden_dim=16, world=PACK))
    baseline.load_state_dict(
        {
            key: value
            for key, value in agent.state_dict().items()
            if not key.startswith("recent_events.")
        }
    )
    # The baseline reads the same decision in its own (history-free) layout.
    for left, right in zip(
        baseline(device_input(baseline, empty)),
        agent(device_input(agent, empty)),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(expected, agent(device_input(agent, raw)), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_checkpoint_player_fixed_view_history_hidden_worlds_and_reset(
    tmp_path: Path,
) -> None:
    space = _space()
    agent = _agent(space)
    path = tmp_path / "player.pt"
    save_checkpoint(agent, space, path, player_configs=_match().to_rust())
    player, loaded_space = make_player(
        {"kind": "checkpoint", "path": str(path)}, seed=111, backend=TORCH_BACKEND
    )
    assert isinstance(player, AgentMatchupPlayer) and loaded_space is not None
    env = Env(_match(), loaded_space, Reward(RewardConfig()), seed=111)

    def events(obs: managym.Observation) -> np.ndarray:
        batch = managym.encode_observation_batch([obs], loaded_space.layout)
        return (
            observation_spans(batch, loaded_space, bucketed=False, objects=False)
            .tables["events"]
            .features()
        )

    obs, _ = env.reset()
    previous = events(obs)
    assert not len(previous)
    retained = False
    nonempty_calls = 0
    for _ in range(32):
        actor = env._engine.current_agent_index()
        np.testing.assert_array_equal(
            previous, events(env._engine.observation_for_player(actor))
        )
        if len(previous):
            nonempty_calls += 1
            for seed in (17, 91):
                hidden = env._engine.clone_env()
                hidden.determinize(seed, perspective=actor)
                np.testing.assert_array_equal(
                    previous, events(hidden.observation_for_player(actor))
                )
        action = player.act(env, obs)
        assert 0 <= action < len(obs.action_space.actions)
        obs, _, terminated, truncated, _ = env.step(action)
        current = events(obs)
        # Perspective/context can change, but kind/amount/definition chronology remains.
        if len(previous) and len(current) >= len(previous):
            retained |= bool(
                np.array_equal(
                    previous[:, [0, 1, 2, 7]], current[: len(previous), [0, 1, 2, 7]]
                )
            )
        previous = current
        if terminated or truncated:
            break
    assert nonempty_calls > 2 and retained
    fresh, _ = env.reset(seed=112)
    assert not len(events(fresh))
    assert not len(events(env._engine.observation_for_player(1)))


def test_history_off_preserves_model_identity_and_outputs() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(9)
    baseline = ActorCritic(ObservationSpace(), ModelConfig(hidden_dim=16))
    torch.manual_seed(9)
    explicit = ActorCritic(
        ObservationSpace(), ModelConfig(hidden_dim=16, recent_events=False)
    )
    assert baseline.config.model_dump() == explicit.config.model_dump()
    assert "recent_events" not in explicit.config.model_dump()
    assert (
        "policy_history_version" not in explicit.observation_space.config.model_dump()
    )
    assert baseline.architecture_receipt() == explicit.architecture_receipt()
    for name, weights in baseline.state_dict().items():
        assert "recent_events" not in name
        assert torch.equal(weights, explicit.state_dict()[name])
    env = Env(_match(), ObservationSpace(), Reward(RewardConfig()), seed=111)
    raw, _ = env.reset()
    expected = baseline(device_input(baseline, raw))
    for left, right in zip(
        expected, explicit(device_input(explicit, raw)), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_short_window_native_collector_and_bad_contracts() -> None:
    space = ObservationSpace(ObservationConfig(policy_history_version=1, max_events=3))
    agent = _agent(space)
    obs = _collected(agent)
    assert obs.project("events").shape == (1, 3, 12)
    assert obs.project("events_valid").sum() == 3
    with pytest.raises(ValueError, match="policy_history_version"):
        _agent(ObservationSpace())
