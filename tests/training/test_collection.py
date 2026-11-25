"""Tests for the seat-routed net-opponent training path (exp-11 / C8)."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch

from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env import target_practice
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.collection import (
    Collector,
    ObservationRows,
    RolloutBatch,
    close_collector,
    collect,
    live_observations,
    open_collector,
)
from manabot.verify.games import INTERACTIVE_DECK
import managym
from tests.fixtures.rollout_arrays import projected_fields, trajectory_rows
from tests.model.prepared_routes import device_input


def _make_collector(
    opponent_mode: str,
    opponent_agent: ActorCritic | None = None,
    num_envs: int = 4,
    seed: int = 7,
) -> Collector:
    obs_space = ObservationSpace()
    match = Match(
        MatchConfig(
            hero="hero",
            villain="villain",
            hero_deck=INTERACTIVE_DECK,
            villain_deck=INTERACTIVE_DECK,
        )
    )
    reward = Reward(RewardConfig())
    return open_collector(
        obs_space,
        match,
        reward,
        num_envs=num_envs,
        seed=seed,
        opponent_mode=opponent_mode,
        learning_seats="one",
        batch_closure="exact-quota",
        opponent_agent=opponent_agent,
        numerics=TorchCollectionSession(device="cpu", seed=seed, precision="float32"),
    )


def _make_agent():
    return ActorCritic(ObservationSpace(), ModelConfig(attention_on=False))


def test_collection_borrows_whole_groups_and_chunk_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent()
    group = agent.observation_space.allocate(3)
    group["actions_valid"][:, 0] = group["agent_player_valid"][:, 0] = 1
    session = TorchCollectionSession(device="cpu", seed=7)
    with session.scope(agent) as scope:
        prepare = scope.inference.prepare
        widths = []

        def check_borrow(rows: ObservationBuffers) -> Prepared[torch.Tensor]:
            assert all(
                np.shares_memory(value, group[key])
                for key, value in rows.items()
                if value.size
            )
            widths.append(len(rows["actions_valid"]))
            if widths[-1] == 3:
                assert rows is group
            return prepare(rows)

        monkeypatch.setattr(scope.inference, "prepare", check_borrow)
        scope.sample(group, Profiler())
        scope.sample(group, Profiler(), execution_rows=2)
    assert widths == [3, 2, 1]


def _check_batch(batch: RolloutBatch, num_steps: int, num_envs: int) -> None:
    obs = trajectory_rows(batch).observation
    shapes = ObservationSpace().shapes
    assert obs.keys() == set(batch.next_obs.fields) == shapes.keys()
    for key, shape in shapes.items():
        assert obs[key].shape == (num_steps, num_envs, *shape)
        assert batch.next_obs.project(key).shape == (num_envs, *shape)
    assert batch.actions.shape == (num_steps, num_envs)
    assert batch.logprobs.shape == (num_steps, num_envs)
    assert batch.rewards.shape == (num_steps, num_envs)
    assert batch.dones.shape == (num_steps, num_envs)
    assert batch.values.shape == (num_steps, num_envs)
    assert batch.next_done.shape == (num_envs,)
    assert obs["agent_player"].shape[:2] == (num_steps, num_envs)
    assert batch.next_obs.project("agent_player").shape[0] == num_envs

    # Terminal-only reward: nonzero rewards only on done transitions, and
    # every nonzero reward is +/- 1.
    nonzero = batch.rewards != 0.0
    assert not np.any(nonzero & ~batch.dones)
    assert set(np.unique(batch.rewards[nonzero])).issubset({1.0, -1.0})

    # Every stored learner observation has at least one valid action.
    assert np.all(obs["actions_valid"].sum(axis=-1) >= 1)
    # Chosen actions were valid at the time.
    steps, envs = np.meshgrid(np.arange(num_steps), np.arange(num_envs), indexing="ij")
    assert np.all(obs["actions_valid"][steps, envs, batch.actions] > 0)


@pytest.mark.parametrize("mode", ["random", "self"])
def test_collector_batch_shapes_and_reward_semantics(mode):
    collector = _make_collector(mode)
    agent = _make_agent()
    batch = collect(collector, agent, num_steps=32)
    _check_batch(batch, 32, 4)
    # Seat balance: streams alternate learner seats.
    assert list(collector.learner_seat) == [0, 1, 0, 1]


def test_collector_frozen_opponent_and_streaming_continuity():
    opponent = _make_agent()
    collector = _make_collector("frozen", opponent_agent=opponent)
    agent = _make_agent()
    first = collect(collector, agent, num_steps=16)
    second = collect(collector, agent, num_steps=16)
    _check_batch(first, 16, 4)
    _check_batch(second, 16, 4)
    assert collector.stats.opponent_decisions > 0
    assert collector.stats.learner_transitions >= 2 * 16 * 4
    # The frozen opponent produced a fingerprint histogram.
    assert sum(collector.stats.opponent_action_types.values()) == (
        collector.stats.opponent_decisions
    )


def test_collector_requires_opponent_agent_for_frozen():
    with pytest.raises(ValueError):
        _make_collector("frozen", opponent_agent=None)


def test_terminal_credit_and_bootstrap():
    from manabot.backends.torch.rl import transition_gae

    rewards = torch.tensor([[0.0], [1.0], [-1.0], [0.0]])
    advantages, returns = transition_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.tensor([[False], [True], [True], [False]]),
        torch.tensor([0.4]),
        1.0,
        1.0,
    )
    torch.testing.assert_close(returns[:, 0], torch.tensor([1.0, 1.0, -1.0, 0.4]))


def test_update_boundary_has_no_banked_or_sampled_actions() -> None:
    collector = _make_collector("self")
    agent = _make_agent()
    first = collect(collector, agent, 7)
    assert collector.stats.learner_transitions == 28
    assert not any(collector._streams)
    assert all(pending is None for pending in collector._pending)
    live = live_observations(collector)
    for key in first.next_obs.fields:
        np.testing.assert_array_equal(first.next_obs.project(key), live.project(key))
    collect(collector, agent, 7)
    assert collector.stats.learner_transitions == 56


def test_paused_buffer_rows_are_unchanged_and_clear_step_flags() -> None:
    collector = _make_collector("self", num_envs=2)
    before = live_observations(collector).select(slice(0, 1)).batch.to_bytes()
    collector._buffers["terminated"][0] = 1
    collector._buffers["rewards"][0] = 7
    collector._env.step_into_buffers([-999, 0], [False, True])
    for key in ("terminated", "truncated", "rewards"):
        assert collector._buffers[key][0] == 0
    assert live_observations(collector).select(slice(0, 1)).batch.to_bytes() == before


def test_collection_logs_and_bootstrap_survive_weight_change() -> None:
    collector = _make_collector("self")
    agent = _make_agent()
    first = collect(collector, agent, 7)
    live = live_observations(collector)
    assert first.next_obs.batch.tables() == live.batch.tables()
    first_logs = trajectory_rows(first).log_probabilities
    with torch.no_grad():
        agent.policy_head[-1].weight.mul_(-5)
        agent.value_head[-1].bias.add_(3.0)
    second = collect(collector, agent, 7)
    rows = trajectory_rows(second)
    # Nonterminal streams resume at precisely the bootstrap state, under new
    # weights. Terminal streams may route the new episode's opponent first.
    for key in rows.observation:
        np.testing.assert_array_equal(
            rows.observation[key][0, ~first.next_done],
            first.next_obs.project(key)[~first.next_done],
        )
    flat = {
        k: torch.as_tensor(v.reshape((-1,) + v.shape[2:]))
        for k, v in rows.observation.items()
    }
    with torch.no_grad():
        logits, values = agent(device_input(agent, flat))
    np.testing.assert_allclose(
        second.values.flatten(), values.numpy().flatten(), atol=1e-5
    )
    np.testing.assert_allclose(
        rows.log_probabilities.reshape(logits.shape),
        logits.masked_fill(flat["actions_valid"] == 0, -torch.inf)
        .log_softmax(-1)
        .numpy(),
        atol=1e-6,
    )
    selected = np.take_along_axis(
        rows.log_probabilities, second.actions[..., None], axis=-1
    )[..., 0]
    np.testing.assert_allclose(second.logprobs, selected, atol=1e-6)
    assert np.all(
        np.isneginf(rows.log_probabilities[rows.observation["actions_valid"] == 0])
    )
    np.testing.assert_array_equal(trajectory_rows(first).log_probabilities, first_logs)
    assert collector.stats.learner_transitions == 56


def test_recorded_rows_survive_growth_and_later_appends() -> None:
    buffers = {"x": np.arange(12, dtype=np.float32).reshape(4, 3)}
    store = ObservationRows(buffers, ["x"], 1)
    assert store.append(np.array([2, 0])) == 0
    first = store.arrays()["x"]
    buffers["x"] += 100  # The native buffers are overwritten every step.
    assert store.append(np.array([3, 1, 1])) == 2
    np.testing.assert_array_equal(first, [[6, 7, 8], [0, 1, 2]])
    np.testing.assert_array_equal(
        store.arrays()["x"],
        [[6, 7, 8], [0, 1, 2], [109, 110, 111], [103, 104, 105], [103, 104, 105]],
    )


def test_repeated_rollout_indexes_preserve_offers_behavior_and_native_owners() -> None:
    model = _make_agent()
    collector = _make_collector("self", num_envs=3, seed=24)
    try:
        rollout = collect(collector, model, 5)
        indexes = np.array([9, 0, 7, 9])
        rows = rollout.observations(indexes)
        selection = rows.observation
        assert isinstance(selection, ObservationSelection)
        assert not isinstance(selection, Mapping)
        recorded = rollout.recorded
        assert selection.coordinates == tuple(
            recorded.rows.coordinates[i] for i in recorded.index[indexes]
        )
        legal = selection.project("actions_valid")
        np.testing.assert_array_equal(legal > 0, np.isfinite(rows.log_probabilities))
        actions = rollout.actions.ravel()[indexes]
        np.testing.assert_allclose(
            rows.log_probabilities[np.arange(4), actions],
            rollout.logprobs.ravel()[indexes],
        )
        editable = projected_fields(selection)
        editable["actions_valid"][0, 0] = 0.25
        assert selection.project("actions_valid")[0, 0] == 1
        with pytest.raises(ValueError, match="invalid_action_mask"):
            model.check_observation(editable)
        admitted = AdmittedRollout(model, rollout)
        admitted.admit_rows(indexes, 2)
        prepared, _ = admitted.rows(indexes, torch.device("cpu"))
        with torch.no_grad():
            logits, values = model(prepared)
        torch.testing.assert_close(logits[0], logits[-1])
        torch.testing.assert_close(values[0], values[-1])
    finally:
        close_collector(collector)


@pytest.mark.parametrize("version", [9, 11])
def test_target_practice_constructs_only_requested_games(
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    seeds: list[int] = []
    fail_seed: int | None = None
    construct = target_practice.target_root

    def root(match: Match, space: ObservationSpace, seed: int, seat: int) -> Env:
        seeds.append(seed)
        if seed == fail_seed:
            raise ValueError("target unavailable")
        return construct(match, space, seed, seat)

    monkeypatch.setattr(target_practice, "target_root", root)
    space = ObservationSpace(ObservationConfig(policy_input_version=version))
    match = Match(
        MatchConfig(
            hero_deck={"Mountain": 12, "Igneous Inspiration": 1},
            villain_deck={"Island": 12},
        )
    )
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    collector = open_collector(
        space,
        match,
        Reward(RewardConfig()),
        num_envs=2,
        seed=10,
        opponent_mode="self",
        root="lethal-target-v1",
        games_per_stream=1,
        numerics=TorchCollectionSession(device="cpu", seed=10),
    )

    def parked_action_counts() -> list[int]:
        if space.one_token:
            return [
                batch.action_counts[i] for batch, i in collector._env.observation_rows()
            ]
        return collector._buffers["actions_valid"].sum(axis=1).tolist()

    try:
        for batch_index in range(2):
            batch = collect(collector, agent, 1)
            assert batch.actions.size == 2 and batch.dones.all()
            assert seeds == list(range(10, 12 + 2 * batch_index))
            assert collector._env.current_agent_indices() == [-1, -1]
            assert parked_action_counts() == [0, 0]
        with pytest.raises(RuntimeError, match="parked"):
            collector._env.step_into_buffers([0, 0], [False, True])
        with pytest.raises(ValueError, match="configured match"):
            collector._env.load_slots_into_buffers(
                [managym.SlotLoad.fresh(0), managym.SlotLoad.fresh(1, match.to_rust())]
            )
        assert seeds == [10, 11, 12, 13]
        fail_seed = 15
        with pytest.raises(ValueError, match="target unavailable"):
            collector._env.load_slots_into_buffers(
                [managym.SlotLoad.fresh(i) for i in (0, 1)]
            )
        assert collector._env.current_agent_indices() == [-1, -1]
        assert parked_action_counts() == [0, 0]
        collector._env.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
        assert seeds[-3:] == [14, 15, 14]
        assert collector._env.current_agent_indices() == [0, -1]
    finally:
        close_collector(collector)
