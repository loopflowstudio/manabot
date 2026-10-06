"""Public identity/history acceptance through native collection and serving."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import (
    AgentSpec,
    MatchHypers,
    ObservationSpaceHypers,
    RewardHypers,
)
from manabot.model.agent import Agent
from manabot.model.architecture import architecture_receipt
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import AgentMatchupPlayer, load_checkpoint_agent, make_player
from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.experiments import Case, Environment, Experiment, Model
from manabot.training.presets import ataraxos_mtg_v1
from manabot.training.recipes import with_recent_events

PACK = "ur-lessons-vs-gw-allies"


def _match() -> Match:
    return Match(MatchHypers.authored(PACK, "ur_lessons", "gw_allies"))


def _space() -> ObservationSpace:
    return ObservationSpace(ObservationSpaceHypers(policy_history_version=1))


def _agent(space: ObservationSpace, **kwargs: object) -> Agent:
    torch.set_num_threads(1)
    torch.manual_seed(111)
    settings = AgentSpec.model_validate(
        {"hidden_dim": 16, "semantic_pack": PACK, "recent_events": True, **kwargs}
    )
    return Agent(space, settings)


def _collected(agent: Agent) -> dict[str, torch.Tensor]:
    collector = SeatRoutedCollector(
        agent.observation_space, _match(), Reward(RewardHypers()), num_envs=1, seed=111
    )
    batch = collector.collect(agent, 16)
    nonempty = np.flatnonzero(batch.obs["events_valid"].sum(axis=(-1, -2)) >= 2)
    assert len(nonempty) > 2, (
        "ordinary semantic collection must retain past public events"
    )
    step = int(nonempty[-1])
    return {key: torch.from_numpy(value[step]) for key, value in batch.obs.items()}


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
    valid = obs["events_valid"].bool()
    assert (obs["events"][..., [2, 7]][valid] > 0).any(), (
        "public definition identity is required"
    )
    policy, value = agent(obs)
    without = {**obs, "events_valid": torch.zeros_like(obs["events_valid"])}
    no_policy, no_value = agent(without)
    legal = obs["actions_valid"].bool()
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
    save_bc_checkpoint(
        agent, agent.observation_space, path, player_configs=_match().to_rust()
    )
    loaded, space = load_checkpoint_agent(str(path))
    assert space.encoder.hypers.policy_history_version == 1
    assert architecture_receipt(loaded) == architecture_receipt(agent)
    assert architecture_receipt(agent).components["recent_events"].total > 0
    for expected, actual in zip((policy, value), loaded(obs), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["hypers"]["agent_hypers"]["recent_events"]
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint_agent(str(path))


def test_semantics_order_padding_and_stateless_reset() -> None:
    agent = _agent(_space())
    obs = _collected(agent)
    # Distinct native public rows, including semantic card references.
    count = int(obs["events_valid"].sum())
    expected = agent(obs)
    reversed_obs = {**obs, "events": obs["events"].clone()}
    reversed_obs["events"][:, :count] = obs["events"][:, :count].flip(1)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(expected, agent(reversed_obs), strict=True)
    )
    changed = {**obs, "events": obs["events"].clone()}
    # Reassign only a public definition to another admitted visible definition.
    ids = obs["semantic_cards"][obs["semantic_cards"] > 0].unique()
    index = next(i for i in range(count) if obs["events"][0, i, 2] > 0)
    replacement = ids[ids != obs["events"][0, index, 2]][0]
    changed["events"][0, index, 2] = replacement
    assert any(
        not torch.equal(a, b) for a, b in zip(expected, agent(changed), strict=True)
    )
    padded = torch.full((1, count * 2 + 3, 12), float("nan"))
    validity = torch.zeros(1, count * 2 + 3)
    padded[:, 1 : 2 * count : 2] = obs["events"][:, :count]
    validity[:, 1 : 2 * count : 2] = 1
    for left, right in zip(
        expected,
        agent({**obs, "events": padded, "events_valid": validity}),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    empty = {**obs, "events": torch.empty(1, 0, 12), "events_valid": torch.empty(1, 0)}
    zero = {**obs, "events": padded, "events_valid": torch.zeros_like(validity)}
    baseline = Agent(ObservationSpace(), AgentSpec(hidden_dim=16, semantic_pack=PACK))
    baseline.load_state_dict(
        {
            key: value
            for key, value in agent.state_dict().items()
            if not key.startswith("recent_events.")
        }
    )
    for left, right in zip(baseline(empty), agent(empty), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(agent(empty), agent(zero), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(expected, agent(obs), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    with pytest.raises(ValueError, match="requires events"):
        agent({key: value for key, value in obs.items() if key != "events"})


def test_checkpoint_player_fixed_view_history_hidden_worlds_and_reset(
    tmp_path: Path,
) -> None:
    space = _space()
    agent = _agent(space)
    path = tmp_path / "player.pt"
    save_bc_checkpoint(agent, space, path, player_configs=_match().to_rust())
    player, loaded_space = make_player(
        {"kind": "checkpoint", "path": str(path)}, seed=111
    )
    assert isinstance(player, AgentMatchupPlayer) and loaded_space is not None
    env = Env(_match(), loaded_space, Reward(RewardHypers()), seed=111)
    obs, _ = env.reset()
    assert not obs["events_valid"].any()
    retained = False
    nonempty_calls = 0
    for _ in range(32):
        previous = obs["events"][obs["events_valid"].astype(bool)].copy()
        actor = env._engine.current_agent_index()
        fixed = loaded_space.encode(env._engine.observation_for_player(actor))
        np.testing.assert_array_equal(obs["events"], fixed["events"])
        if obs["events_valid"].any():
            nonempty_calls += 1
            for seed in (17, 91):
                hidden = env._engine.clone_env()
                hidden.determinize(seed, perspective=actor)
                variant = loaded_space.encode(hidden.observation_for_player(actor))
                for key in ("events", "events_valid"):
                    np.testing.assert_array_equal(obs[key], variant[key])
        action = player.act(env, obs)
        assert obs["actions_valid"][action]
        obs, _, terminated, truncated, _ = env.step(action)
        current = obs["events"][obs["events_valid"].astype(bool)]
        # Perspective/context can change, but kind/amount/definition chronology remains.
        if len(previous) and len(current) >= len(previous):
            retained |= bool(
                np.array_equal(
                    previous[:, [0, 1, 2, 7]], current[: len(previous), [0, 1, 2, 7]]
                )
            )
        if terminated or truncated:
            break
    assert nonempty_calls > 2 and retained
    fresh, _ = env.reset(seed=112)
    assert not fresh["events_valid"].any()
    assert not loaded_space.encode(env._engine.observation_for_player(1))[
        "events_valid"
    ].any()


def test_history_off_and_declarative_identity() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(9)
    baseline = Agent(ObservationSpace(), AgentSpec(hidden_dim=16))
    torch.manual_seed(9)
    explicit = Agent(ObservationSpace(), AgentSpec(hidden_dim=16, recent_events=False))
    assert baseline.hypers.model_dump() == explicit.hypers.model_dump()
    assert "recent_events" not in explicit.hypers.model_dump()
    assert (
        "policy_history_version"
        not in explicit.observation_space.encoder.hypers.model_dump()
    )
    assert architecture_receipt(baseline) == architecture_receipt(explicit)
    for name, weights in baseline.state_dict().items():
        assert "recent_events" not in name
        assert torch.equal(weights, explicit.state_dict()[name])
    env = Env(_match(), ObservationSpace(), Reward(RewardHypers()), seed=111)
    raw, _ = env.reset()
    tensors = {key: torch.from_numpy(value).unsqueeze(0) for key, value in raw.items()}
    expected = baseline(tensors)
    tensors.pop("events")
    tensors.pop("events_valid")
    for left, right in zip(expected, explicit(tensors), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    base = ataraxos_mtg_v1()
    comparison = Experiment(
        "history-unexecuted",
        base,
        cases=(
            Case("off", (Model(AgentSpec(recent_events=False)),)),
            Case("on", (Model(AgentSpec(recent_events=True)),)),
        ),
    ).resolve()
    off, on = (case.regime for case in comparison.cases)
    for case in comparison.cases:
        provenance = {setting.path: setting for setting in case.provenance}
        assert provenance[("agent", "recent_events")].origin == "override"
        assert (
            provenance[("observation", "policy_history_version")].component == "model"
        )
    assert off.observation.policy_history_version == 0
    assert on.observation.policy_history_version == 1
    assert on.stages == off.stages
    assert on.agent.hidden_dim == off.agent.hidden_dim
    assert on.agent.value_aggregation == off.agent.value_aggregation
    changed = with_recent_events(base.regime(), id="history-on", enabled=True)
    restored = with_recent_events(changed, id=base.regime().id, enabled=False)
    assert restored.model_dump() == base.regime().model_dump()
    assert base.regime().agent.recent_events is False


def test_short_window_native_collector_and_bad_contracts() -> None:
    space = ObservationSpace(
        ObservationSpaceHypers(policy_history_version=1, max_events=3)
    )
    agent = _agent(space)
    obs = _collected(agent)
    assert obs["events"].shape == (1, 3, 12)
    assert obs["events_valid"].sum() == 3
    with pytest.raises(ValueError, match="policy_history_version"):
        _agent(ObservationSpace())
    with pytest.raises(ValueError, match="history input is selected by Model"):
        Experiment(
            "bad",
            ataraxos_mtg_v1(),
            overrides=(
                Environment(
                    observation=ObservationSpaceHypers(policy_history_version=1)
                ),
            ),
        ).resolve()
