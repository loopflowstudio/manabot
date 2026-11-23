"""Packed seat trajectories bootstrap at their viewer's frozen game revision."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manabot.backends import resolve
import manabot.backends.torch as backend
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.memory_store import MEMORY_FIELDS, PrefixObservation
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
)
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.policy.checkpoint import from_torch_state
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import (
    Collector,
    RolloutBatch,
    TrajectorySpan,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.rollout.producers import ShardedVectorEnv
from manabot.training.rollout.snapshot_vector import (
    InstalledSlot,
    SlotInstall,
    SnapshotBranch,
    SnapshotRecipe,
)
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    RecordAttempt,
    settle,
)
from manabot.training.updates.damped_policy import (
    compute_targets,
    update_damped_iteration,
)
from manabot.training.updates.ppo import update_iteration
import managym
from managym.recording import GameInputs, PlayerSetup
from tests.model.prepared_routes import device_input


@pytest.fixture(autouse=True)
def bounded_torch() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            yield
    finally:
        torch.set_num_threads(previous)


class ReplayVector(CapturedVector):
    """One native stream restores a root by replaying its exact action prefix."""

    def install_slots(self, installs: Sequence[SlotInstall]) -> list[InstalledSlot]:
        [request] = installs
        assert request.slot == 0
        if request.recipe is None:
            self.reset_seeds_into_buffers([55])
        else:
            game = request.recipe
            self.reset_all_into_buffers(Match(game.match).to_rust())
            self.reset_seeds_into_buffers([game.seed])
            for action in game.actions:
                self.step_into_buffers([int(action)])
        [digest] = self.native.state_digests()
        assert digest is not None
        return [InstalledSlot(0, digest, self.current_agent_indices()[0])]


def replay_root(
    space: ObservationSpace, terminal: bool = False
) -> tuple[SnapshotRecipe, managym.Env]:
    setup = managym.GameSetup(
        [
            managym.PlayerConfig("a", {"Mountain": 8, "Gray Ogre": 4}),
            managym.PlayerConfig("b", {"Island": 8, "Wind Drake": 4}),
        ],
        decklists="closed",
    )
    memory = space.config.max_decision_records > 0
    own = space.config.decision_memory_include_own
    scalar = managym.Env(
        seed=42, decision_memory=memory, decision_memory_include_own=own
    )
    scalar.reset(setup)
    actions = [0] * (30 if terminal else 4)
    for action in actions:
        scalar.step(action)
    recipe = SnapshotRecipe.from_inputs(
        GameInputs(
            seed=42,
            players=[PlayerSetup.retain(p, "closed") for p in setup],
            actions=actions,
            decision_memory=memory,
            decision_memory_include_own=own,
        ),
        scalar.state_digest(),
        0,
    )
    return recipe, scalar


@contextmanager
def replay_collector(
    space: ObservationSpace,
    recipe: SnapshotRecipe,
    *,
    game_errors: GameErrorPolicy | None = None,
    record_attempt: RecordAttempt | None = None,
) -> Iterator[Collector]:
    vector = ReplayVector(
        num_envs=1,
        seed=11,
        decision_memory=recipe.decision_memory,
        decision_memory_include_own=recipe.decision_memory_include_own,
    )
    collector = open_collector(
        space,
        Match(recipe.match),
        Reward(RewardConfig()),
        num_envs=1,
        opponent_mode="self",
        snapshot_vector=vector,
        game_errors=game_errors,
        record_attempt=record_attempt,
        numerics=CollectionSession(device="cpu", seed=0),
    )
    try:
        # Fix the random draw to the first legal offer on this retained line.
        with patch.object(
            torch, "multinomial", return_value=torch.zeros((1, 1), dtype=torch.long)
        ):
            yield collector
    finally:
        close_collector(collector)


@torch.no_grad()
def three_choices(
    agent: ActorCritic, *, terminal: bool = False, horizon: int = 3
) -> tuple[RolloutBatch, torch.Tensor]:
    """Restore a real game and compare each collected cut to an independent scalar view."""
    space = agent.observation_space
    recipe, scalar = replay_root(space, terminal)
    with replay_collector(space, recipe) as collector:
        batch = collect(
            collector,
            agent,
            1,
            snapshot_starts=[
                SnapshotBranch(
                    7, recipe, horizon=horizon, branch_group_id=9, sample_weight=0.5
                )
            ],
        )
        assert collector.stats.games == collector.stats.skipped_decisions == 0
        assert collector.stats.self_play.drain().outcomes == []
        assert collector.stats.learner_transitions == horizon
        assert collector.stats.micro_steps == horizon
    seats = (0, 0, 1)[:horizon]
    for seat, action in zip(seats, batch.actions[:, 0], strict=True):
        assert scalar.current_agent_index() == seat
        _, _, ended, truncated, _ = scalar.step(int(action))
    assert ended == terminal and not truncated
    np.testing.assert_array_equal(batch.is_root_decision[:, 0], np.arange(horizon) == 0)
    assert (batch.snapshot_id == 7).all() and (batch.episode_source == 1).all()
    assert (batch.branch_group_id == 9).all() and (batch.sample_weight == 0.5).all()
    spans = (TrajectorySpan(0, min(horizon, 2), 0, None if terminal else 0),)
    if horizon == 3:
        spans += (TrajectorySpan(2, 3, 1, None if terminal else 1),)
    assert batch.trajectories == spans
    if terminal:
        assert not batch.next_obs and batch.next_done.size == 0
        return batch, torch.empty(0)

    expected = []
    for span in spans:
        encoded = encode_observation(space, scalar.observation_for_player(span.seat))
        assert span.bootstrap_index is not None
        frozen = batch.next_obs
        if isinstance(frozen, PrefixObservation):
            dense = frozen.materialize()
            # Compact memory pads only to the largest live prefix in this read.
            for key in MEMORY_FIELDS:
                np.testing.assert_array_equal(
                    dense[key][span.bootstrap_index],
                    encoded[key][: dense[key].shape[1]],
                )
        for key in frozen:
            np.testing.assert_array_equal(
                frozen[key][span.bootstrap_index], encoded[key], err_msg=key
            )
        assert bool(encoded["actions_valid"].any()) == (
            scalar.current_agent_index() == span.seat
        )
        _, prediction = agent.forward_distribution(
            device_input(agent, {key: value[None] for key, value in encoded.items()})
        )
        expected.append(prediction[0])
    return batch, torch.stack(expected)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_packed_cut_updates_use_the_frozen_view_of_each_seat(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )
    batch, raw = three_choices(agent)
    learner = Learner(agent, "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        policy_lambda=0.25,
        value_lambda=0.75,
        advantage_quantile=0,
        min_advantage=0,
    )
    targets = compute_targets(backend, learner, AdmittedRollout(agent, batch), learning)
    predictions = raw.softmax(-1) if kind == "categorical_wdl" else raw
    expected = predictions[[0, 0, 1]]
    # The two native views must make distinguishable predictions for this control.
    assert not torch.allclose(predictions[0], predictions[1])
    signed = (
        backend.rl.expected_outcome(expected)
        if kind == "categorical_wdl"
        else expected[:, 0].clone()
    )
    signed[0] = 0.75 * batch.values[1, 0].item() + 0.25 * signed[1]
    collected = (
        batch.outcome_probabilities if kind == "categorical_wdl" else batch.values
    )
    assert collected is not None
    expected[0] = (
        0.25 * torch.from_numpy(collected[1, 0:1]).flatten() + 0.75 * expected[1]
    )
    torch.testing.assert_close(
        targets.values, expected[:, None] if kind == "categorical_wdl" else expected
    )
    torch.testing.assert_close(
        targets.advantages[:, 0], signed - torch.from_numpy(batch.values[:, 0])
    )

    # Storage order is independent of the trajectory packing order.
    reordered = replace(
        batch,
        next_obs={k: v[[1, 0]] for k, v in batch.next_obs.items()},
        trajectories=(TrajectorySpan(0, 2, 0, 1), TrajectorySpan(2, 3, 1, 0)),
    )
    reordered_targets = compute_targets(
        backend, learner, AdmittedRollout(agent, reordered), learning
    )
    torch.testing.assert_close(reordered_targets.values, targets.values)
    torch.testing.assert_close(reordered_targets.advantages, targets.advantages)

    # Replace just one viewer's frozen input. Only its span may change credit.
    for seat in (0, 1):
        changed_rows = np.array([0, 0, 1]) == seat
        other = 1 - seat
        perturbed = replace(
            batch, next_obs={k: v[[other, other]] for k, v in batch.next_obs.items()}
        )
        changed = compute_targets(
            backend, learner, AdmittedRollout(agent, perturbed), learning
        )
        torch.testing.assert_close(
            changed.advantages[~changed_rows], targets.advantages[~changed_rows]
        )
        assert not torch.allclose(
            changed.advantages[changed_rows], targets.advantages[changed_rows]
        )

    result = update_damped_iteration(backend, learner, batch, learning, 1)
    assert result["optimizer_exposures"] == 3
    assert result["bootstrapped_tail_fraction"] == 1
    assert {g["terminal_distance"] for g in result["selection_groups"]} == {"censored"}


def test_mixed_ppo_discounts_each_cut_and_censors_only_its_own_span() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, raw = three_choices(agent)
    batch = replace(
        batch,
        dones=np.array([[False], [False], [True]]),
        rewards=np.array([[0], [0], [-1]], np.float32),
        next_obs={key: value[:1] for key, value in batch.next_obs.items()},
        next_done=np.zeros(1, bool),
        trajectories=(TrajectorySpan(0, 2, 0, 0), TrajectorySpan(2, 3, 1)),
    )
    diagnostics = update_iteration(
        backend,
        Learner(agent, "cpu", minibatch_rows=3),
        batch,
        Learning(gamma=0.5, epochs=1, minibatches=1, retained_fraction=1),
        0.0,
        np.random.default_rng(2),
    )
    assert diagnostics["optimizer_exposures"] == 3
    assert diagnostics["bootstrapped_tail_fraction"] == 0.5
    assert diagnostics["return_mean"] == pytest.approx(
        (0.75 * raw[0].item() - 1) / 3, abs=1e-6
    )
    assert (
        sum(
            g["rows"]
            for g in diagnostics["selection_groups"]
            if g["terminal_distance"] == "censored"
        )
        == 2
    )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_packed_terminal_spans_need_no_bootstrap_input(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )
    batch, _ = three_choices(agent, terminal=True)
    admitted = AdmittedRollout(agent, batch)
    assert admitted.next_observation(torch.device("cpu")) is None
    targets = compute_targets(
        backend,
        Learner(agent, "cpu", minibatch_rows=3),
        admitted,
        DampedPolicyLearning(gradient="damped_policy", value_lambda=1),
    )
    signed = (
        backend.rl.expected_outcome(targets.values)
        if kind == "categorical_wdl"
        else targets.values
    )
    torch.testing.assert_close(signed[:, 0], torch.tensor([1.0, 1.0, -1.0]))


def test_mlx_scalar_cut_and_terminal_targets() -> None:
    pytest.importorskip("mlx.core")
    mlx = resolve("mlx")
    with pytest.warns(DeprecationWarning):
        space = ObservationSpace(ObservationConfig(policy_input_version=9))
    agent = ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
    )
    batch, raw = three_choices(agent)
    checkpoint = from_torch_state(
        agent.config,
        space.config,
        {k: v.detach().numpy() for k, v in agent.state_dict().items()},
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )
    learner = mlx.learner.Learner(mlx.load(checkpoint)[0], "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy", policy_lambda=1, value_lambda=1
    )
    for rollout, expected in (
        (batch, raw.numpy()[[0, 0, 1], 0]),
        (three_choices(agent, terminal=True)[0], np.array([1, 1, -1])),
    ):
        targets = compute_targets(
            mlx, learner, mlx.rollout.AdmittedRollout(learner.agent, rollout), learning
        )
        np.testing.assert_allclose(
            mlx.arrays.to_numpy(targets.values)[:, 0], expected, atol=1e-5
        )


def test_invalid_packed_boundaries_are_rejected_at_construction() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, _ = three_choices(agent)
    for spans in (
        (TrajectorySpan(0, 1, 0, 1), TrajectorySpan(2, 3, 1, 0)),  # missing row
        (TrajectorySpan(0, 2, 0, 2), TrajectorySpan(2, 3, 1, 0)),  # missing observation
        (TrajectorySpan(0, 2, 0), TrajectorySpan(2, 3, 1, 0)),  # cut lacks a bootstrap
    ):
        with pytest.raises(ValueError, match="trajectory|nonterminal spans"):
            replace(batch, trajectories=spans)
    with pytest.raises(ValueError, match="cross terminals"):
        replace(batch, dones=np.array([[True], [False], [False]]))


def test_single_choice_cut_freezes_only_its_contributing_seat() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, _ = three_choices(agent, horizon=1)
    assert batch.next_done.tolist() == [False]
    assert batch.dones.tolist() == [[False]]


def test_snapshot_bootstraps_retain_both_viewers_memory_at_the_cut() -> None:
    space = ObservationSpace(
        ObservationConfig(
            max_decision_records=64,
            max_decision_reference_details=8192,
            max_decision_references=512,
            max_decision_members=4096,
        )
    )
    batch, _ = three_choices(
        ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    )
    assert isinstance(batch.next_obs, PrefixObservation)
    assert [p.identity.viewer for p in batch.next_obs.prefixes] == [0, 1]


def test_sharded_fixed_view_reads_use_global_stream_order() -> None:
    space = ObservationSpace()
    setup = Match().to_rust()
    vector = ShardedVectorEnv(3, 42, 2)
    try:
        vector.set_buffers(allocate_vector_buffers(space, 3), layout=space.layout)
        vector.reset_all_into_buffers(setup)
        frozen = space.allocate(3)
        players = [1, 0, 1]
        reads = vector.write_observations_for_players(
            players, frozen, layout=space.layout
        )
        for slot, read in enumerate(reads):
            assert (read.slot, read.player, read.seed, read.game_id) == (
                slot,
                players[slot],
                42 + slot,
                vector.current_game_generations()[slot],
            )
            scalar = managym.Env(seed=42 + slot)
            scalar.reset(setup)
            encoded = encode_observation(
                space, scalar.observation_for_player(players[slot])
            )
            for key in frozen:
                np.testing.assert_array_equal(frozen[key][slot], encoded[key])
    finally:
        vector.close()


def test_slot_reuse_packs_snapshot_and_fresh_rows_and_counts_one_fresh_game() -> None:
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    recipe, _ = replay_root(space)
    branch = SnapshotBranch(3, recipe, horizon=1, branch_group_id=8)
    with replay_collector(space, recipe) as collector:
        batch = collect(
            collector,
            agent,
            3,
            snapshot_starts=[branch, replace(branch, branch_index=1)],
        )
        assert collector.stats.games == 1
        assert collector.stats.learner_transitions == batch.actions.size
        assert collector.stats.self_play.drain().outcomes
        assert batch.trajectories is not None
        assert batch.trajectories[:2] == (
            TrajectorySpan(0, 1, 0, 0),
            TrajectorySpan(1, 2, 0, 1),
        )
        assert all(s.bootstrap_index is None for s in batch.trajectories[2:])
        np.testing.assert_array_equal(
            batch.is_root_decision[:, 0], np.arange(batch.actions.size) < 2
        )
        assert (batch.episode_source[:2] == 1).all() and (
            batch.episode_source[2:] == 0
        ).all()
        assert (batch.snapshot_id[2:] == -1).all() and (
            batch.branch_group_id[2:] == -1
        ).all()


def test_provenance_rejects_misaligned_arrays_invalid_values_and_mixed_spans() -> None:
    agent = ActorCritic(
        ObservationSpace(), ModelConfig(attention_on=False, hidden_dim=8)
    )
    batch, _ = three_choices(agent)
    for changes in (
        {"episode_source": np.full((3, 1), 2, np.int8)},
        {"episode_source": np.array([[0], [1], [1]], np.int8)},
        {"snapshot_id": np.array([[7], [8], [7]], np.int64)},
        {"branch_group_id": np.full((3, 1), -2, np.int64)},
        {"sample_weight": np.ones(3, np.float32)},
        {"sample_weight": np.ones((3, 1), np.float64)},
        {"sample_weight": np.full((3, 1), np.nan, np.float32)},
        {"sample_weight": np.full((3, 1), -1, np.float32)},
        {"is_root_decision": np.array([[True], [True], [False]])},
    ):
        with pytest.raises(ValueError):
            replace(batch, **changes)


def test_failed_snapshot_discards_only_its_rows_and_retains_the_replay_prefix() -> None:
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    recipe, _ = replay_root(space)
    policy = GameErrorPolicy(max_game_actions=2, window=1, warmup=1)
    attempts: list[GameAttempt] = []

    def record(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
        attempts.append(attempt)
        return settle(state, attempt, policy)

    with replay_collector(
        space, recipe, game_errors=policy, record_attempt=record
    ) as collector:
        with pytest.raises(GameErrorLimit):
            collect(
                collector,
                agent,
                2,
                snapshot_starts=[
                    SnapshotBranch(5, recipe, horizon=1),
                    SnapshotBranch(6, recipe, horizon=2),
                ],
            )
        assert collector.stats.learner_transitions == 1
        assert collector.stats.games == 0
    [failure] = attempts
    assert failure.outcome == "action_limit" and failure.discarded_samples == 2
    assert (failure.seed, failure.matchup, failure.actions) == (
        recipe.seed,
        recipe.match,
        [*recipe.actions, 0, 0],
    )


def test_install_rejects_the_wrong_root_before_sampling() -> None:
    space = ObservationSpace()
    agent = ActorCritic(space, ModelConfig(attention_on=False, hidden_dim=8))
    recipe, _ = replay_root(space)
    branch = SnapshotBranch(17, replace(recipe, root_digest="another game"), horizon=1)
    with replay_collector(space, recipe) as collector:
        with pytest.raises(ValueError, match="snapshot 17 root identity mismatch"):
            collect(collector, agent, 1, snapshot_starts=[branch])
        assert collector.stats.micro_steps == 0
