"""Packed seat trajectories bootstrap at their viewer's frozen game revision."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from etude.train.read import read
import numpy as np
import pytest
import torch

from manabot.backends import resolve
import manabot.backends.torch as backend
from manabot.backends.torch.collection import CollectionSession
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import NumericalError
from manabot.backends.torch.rollout import AdmittedRollout
from manabot.env.failures import CaptureScope, RecordedVectorEnv, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.memory_store import MEMORY_FIELDS, gather_memory
from manabot.env.observation import (
    ObservationSpace,
    allocate_vector_buffers,
    encode_observation,
    step_buffers,
)
from manabot.infra.config import (
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.infra.identity import canonical_sha256
from manabot.infra.run_settings import RunSettings
from manabot.policy.checkpoint import (
    PolicyCheckpoint,
    canonical_parameters,
    from_torch_state,
)
from manabot.policy.decision_memory import retained_memory
from manabot.policy.prepare import reference_distribution
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.decks import UniformDecks
from manabot.training.learning import DampedPolicyLearning, Learning
from manabot.training.loops.self_play import (
    CLOCKS,
    Counts,
    SelfPlay,
    create,
    learn,
    resume_training,
    self_play,
    self_play_collector,
    step,
)
from manabot.training.rollout.collection import (
    Collector,
    CollectorSnapshot,
    RolloutBatch,
    TrajectorySpan,
    capture_collector,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.rollout.producers import ShardedVectorEnv
from manabot.training.rollout.snapshot_vector import (
    CapturedSnapshot,
    InMemorySnapshotVector,
    SlotInstall,
    SnapshotBranch,
    SnapshotRecipe,
    replay_prefix,
    replay_snapshot,
)
from manabot.training.rollout.snapshots import (
    CaptureDecision,
    CapturePolicy,
    EntrySignals,
    SnapshotBank,
    SnapshotBankState,
    SnapshotConfig,
    root_id,
)
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    RecordAttempt,
    replay_attempt,
    settle,
)
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration
from manabot.training.updates.targets import BatchTargets, compute_batch_targets
import managym
from managym.recording import GameInputs, PlayerSetup, read_records
from managym.statistics import read_statistics
from scripts.inspect_batch import _report
from tests.fixtures.rollout_arrays import projected_fields
from tests.fixtures.training_records import recording
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


def small_agent(
    space: ObservationSpace | None = None,
    *,
    kind: Literal["scalar", "categorical_wdl"] = "scalar",
) -> ActorCritic:
    return ActorCritic(
        space if space is not None else ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )


def torch_targets(
    learner: Learner, batch: RolloutBatch, learning: Learning | DampedPolicyLearning
) -> BatchTargets:
    return compute_batch_targets(
        backend, learner, AdmittedRollout(learner.agent, batch), learning
    )


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
    own = space.config.decision_memory_include_own
    memory = space.config.max_decision_records > 0 or own
    scalar = managym.Env(
        seed=42, decision_memory=memory, decision_memory_include_own=own
    )
    scalar.reset(setup)
    actions = [0, 0] + [0] * (30 if terminal else 4)
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
    num_envs: int = 1,
    snapshot: CollectorSnapshot | None = None,
    game_errors: GameErrorPolicy | None = None,
    record_attempt: RecordAttempt | None = None,
) -> Iterator[Collector]:
    collector = open_collector(
        space,
        Match(recipe.match),
        Reward(RewardConfig()),
        num_envs=num_envs,
        opponent_mode="self",
        snapshots=True,
        seed=42,
        snapshot=snapshot,
        game_errors=game_errors,
        record_attempt=record_attempt,
        numerics=CollectionSession(device="cpu", seed=0),
    )
    try:
        # Fix the random draw to the first legal offer on this retained line.
        with patch.object(
            torch,
            "multinomial",
            side_effect=lambda probs, *args, **kwargs: torch.zeros(
                (len(probs), 1), dtype=torch.long
            ),
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
        assert batch.next_done.size == 0
        return batch, torch.empty(0)

    expected = []
    for span in spans:
        encoded = encode_observation(space, scalar.observation_for_player(span.seat))
        assert span.bootstrap_index is not None
        frozen = batch.next_obs
        if retained_memory(frozen) is not None:
            dense = gather_memory(frozen.memory.prefixes)
            # Compact memory pads only to the largest live prefix in this read.
            for key in MEMORY_FIELDS:
                np.testing.assert_array_equal(
                    dense[key][span.bootstrap_index],
                    encoded[key][: dense[key].shape[1]],
                )
        for key, value in projected_fields(frozen).items():
            np.testing.assert_array_equal(
                value[span.bootstrap_index], encoded[key], err_msg=key
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
    agent = small_agent(kind=kind)
    batch, raw = three_choices(agent)
    learner = Learner(agent, "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        policy_lambda=0.25,
        value_lambda=0.75,
        advantage_quantile=0,
        min_advantage=0,
    )
    targets = torch_targets(learner, batch, learning)
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
        targets.value_targets,
        expected[:, None] if kind == "categorical_wdl" else expected,
    )
    torch.testing.assert_close(
        targets.advantages[:, 0], signed - torch.from_numpy(batch.values[:, 0])
    )

    # Storage order is independent of the trajectory packing order.
    reordered = replace(
        batch,
        next_obs=batch.next_obs.select(np.array([1, 0])),
        trajectories=(TrajectorySpan(0, 2, 0, 1), TrajectorySpan(2, 3, 1, 0)),
    )
    reordered_targets = torch_targets(learner, reordered, learning)
    torch.testing.assert_close(reordered_targets.value_targets, targets.value_targets)
    torch.testing.assert_close(reordered_targets.advantages, targets.advantages)

    # Replace just one viewer's frozen input. Only its span may change credit.
    for seat in (0, 1):
        changed_rows = np.array([0, 0, 1]) == seat
        other = 1 - seat
        perturbed = replace(
            batch, next_obs=batch.next_obs.select(np.array([other, other]))
        )
        changed = torch_targets(learner, perturbed, learning)
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
    agent = small_agent()
    batch, raw = three_choices(agent)
    batch = replace(
        batch,
        dones=np.array([[False], [False], [True]]),
        rewards=np.array([[0], [0], [-1]], np.float32),
        next_obs=batch.next_obs.select(slice(0, 1)),
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
    agent = small_agent(kind=kind)
    batch, _ = three_choices(agent, terminal=True)
    admitted = AdmittedRollout(agent, batch)
    assert admitted.next_observation(torch.device("cpu")) is None
    targets = compute_batch_targets(
        backend,
        Learner(agent, "cpu", minibatch_rows=3),
        admitted,
        DampedPolicyLearning(gradient="damped_policy", value_lambda=1),
    )
    signed = (
        backend.rl.expected_outcome(targets.value_targets)
        if kind == "categorical_wdl"
        else targets.value_targets
    )
    torch.testing.assert_close(signed[:, 0], torch.tensor([1.0, 1.0, -1.0]))


def scalar_backend_agent() -> ActorCritic:
    with pytest.warns(DeprecationWarning):
        space = ObservationSpace(ObservationConfig(policy_input_version=9))
    return ActorCritic(
        space,
        ModelConfig(
            hidden_dim=16,
            num_attention_heads=2,
            value_aggregation="masked_mean",
            world=managym.WORLD_SETUP,
        ),
    )


def scalar_checkpoint(agent: ActorCritic) -> PolicyCheckpoint:
    return from_torch_state(
        agent.config,
        agent.observation_space.config,
        {k: v.detach().numpy() for k, v in agent.state_dict().items()},
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )


def test_mlx_scalar_cut_and_terminal_targets() -> None:
    pytest.importorskip("mlx.core")
    mlx = resolve("mlx")
    agent = scalar_backend_agent()
    batch, raw = three_choices(agent)
    checkpoint = scalar_checkpoint(agent)
    learner = mlx.learner.Learner(mlx.load(checkpoint)[0], "cpu", minibatch_rows=3)
    learning = DampedPolicyLearning(
        gradient="damped_policy", policy_lambda=1, value_lambda=1
    )
    for rollout, expected in (
        (batch, raw.numpy()[[0, 0, 1], 0]),
        (three_choices(agent, terminal=True)[0], np.array([1, 1, -1])),
    ):
        targets = compute_batch_targets(
            mlx, learner, mlx.rollout.AdmittedRollout(learner.agent, rollout), learning
        )
        np.testing.assert_allclose(
            mlx.arrays.to_numpy(targets.value_targets)[:, 0], expected, atol=1e-5
        )


def test_invalid_packed_boundaries_are_rejected_at_construction() -> None:
    agent = small_agent()
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
    agent = small_agent()
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
    batch, _ = three_choices(small_agent(space))
    memory = retained_memory(batch.next_obs)
    assert memory is not None
    assert [p.identity.viewer for p in memory.prefixes] == [0, 1]


def test_sharded_fixed_view_reads_use_global_stream_order() -> None:
    space = ObservationSpace()
    setup = Match().to_rust()
    vector = ShardedVectorEnv(3, 42, 2)
    try:
        vector.set_observation_layout(
            step_buffers(3), space.config.action_overflow, layout=space.layout
        )
        seeds = [701, 49, 33]
        vector.reset_each_into_buffers([setup] * 3, seeds=seeds)
        generations = vector.current_game_generations()
        with pytest.raises(ValueError, match="native seed must fit uint64"):
            vector.reset_each_into_buffers([setup] * 3, seeds=[800, 900, -1])
        assert vector.current_game_generations() == generations
        players = [1, 0, 1]
        reads, frozen = vector.observations_for_players(players)
        digests = vector.state_digests([True, False, True])
        for slot, read in enumerate(reads):
            assert (read.slot, read.player, read.seed, read.game_id) == (
                slot,
                players[slot],
                seeds[slot],
                generations[slot],
            )
            scalar = managym.Env(seed=seeds[slot])
            scalar.reset(setup)
            assert digests[slot] == (None if slot == 1 else scalar.state_digest())
            encoded = managym.encode_observation_batch(
                [scalar.observation_for_player(players[slot])], space.layout
            )
            assert frozen.gather([(frozen, slot)]).to_bytes() == encoded.to_bytes()
    finally:
        vector.close()


def test_replay_prefix_records_only_the_supplied_root(tmp_path: Path) -> None:
    space = ObservationSpace()
    recipe, scalar = replay_root(space)
    games = tmp_path / "games.jsonl"
    statistics = tmp_path / "statistics.jsonl"
    vector = RecordedVectorEnv(
        num_envs=1, seed=900, games_path=games, statistics_path=statistics
    )
    try:
        vector.set_buffers(allocate_vector_buffers(space, 1), layout=space.layout)
        replay_prefix(vector, recipe, None)
        assert vector.state_digests() == [scalar.state_digest()]
        assert vector.current_agent_indices() == [scalar.current_agent_index()]
    finally:
        vector.close()
    [record] = read_records(games)
    [game] = record.games
    assert (game.seed, game.actions) == (recipe.seed, list(recipe.actions))
    [summary] = read_statistics(statistics)
    assert summary.seed == recipe.seed


def test_collector_resumes_after_snapshot_branch(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(ObservationConfig(decision_memory_include_own=True))
    agent = small_agent(space)
    recipe, _ = replay_root(space)
    branch = SnapshotBranch(7, recipe, horizon=1)
    with replay_collector(space, recipe) as collector:
        collect(collector, agent, 1, snapshot_starts=[branch])
        saved = capture_collector(collector)
        # The boundary reopens the branched slot at its refunded fresh seed.
        assert saved.parked_fresh == []
        [root] = saved.streams
        assert (root.recipe.seed, root.recipe.actions) == (recipe.seed, ())
        control = collect(collector, agent, 1)
        after = capture_collector(collector)

    scope = CaptureScope(
        tmp_path / "failures", settings=RunSettings(record_games=1, statistics=True)
    )
    with (
        capture_scope(scope),
        replay_collector(space, recipe, snapshot=saved) as collector,
    ):
        assert capture_collector(collector) == saved
        assert not list(tmp_path.rglob("*.jsonl"))
        resumed = collect(collector, agent, 1)
        np.testing.assert_array_equal(resumed.actions, control.actions)
        assert capture_collector(collector) == after

    played = [
        game
        for record in read_records(tmp_path / "games")
        for game in record.games
        if game.actions
    ]
    assert len(played) == 1 and played[0].end.status == "complete"
    statistics = [
        r
        for path in (tmp_path / "statistics").glob("*.jsonl")
        for r in read_statistics(path)
    ]
    assert [r.seed for r in statistics if r.terminal] == [played[0].seed]


def test_slot_reuse_packs_snapshot_and_fresh_rows_and_counts_one_fresh_game() -> None:
    space = ObservationSpace()
    agent = small_agent(space)
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


@torch.no_grad()
def test_recorded_games_and_statistics_hold_only_fresh_games(tmp_path: Path) -> None:
    """A continuation replays its root's prefix; recording it would count it twice."""
    space = ObservationSpace()
    agent = small_agent(space)
    recipe, _ = replay_root(space)
    branch = SnapshotBranch(3, recipe, horizon=1, branch_group_id=8)
    scope = CaptureScope(
        tmp_path / "failures",
        settings=RunSettings(record_games=1, statistics=True),
    )
    with capture_scope(scope), replay_collector(space, recipe) as collector:
        collect(
            collector,
            agent,
            3,
            snapshot_starts=[branch, replace(branch, branch_index=1)],
        )
        assert collector.stats.games == 1
    recorded = [
        game for record in read_records(tmp_path / "games") for game in record.games
    ]
    assert all(game.end is not None and game.end.reason != "reset" for game in recorded)
    played = [game for game in recorded if game.actions]
    assert len(played) == 1 and played[0].end is not None
    assert played[0].end.status == "complete"
    statistics = [
        row
        for path in (tmp_path / "statistics").glob("*.jsonl")
        for row in read_statistics(path)
    ]
    assert [row.terminal for row in statistics if row.seed == played[0].seed] == [True]
    assert sum(row.terminal for row in statistics) == 1


def test_provenance_rejects_misaligned_arrays_invalid_values_and_mixed_spans() -> None:
    agent = small_agent()
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
    agent = small_agent(space)
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
    agent = small_agent(space)
    recipe, _ = replay_root(space)
    branch = SnapshotBranch(17, replace(recipe, root_digest="another game"), horizon=1)
    with replay_collector(space, recipe) as collector:
        with pytest.raises(ValueError, match="snapshot 17 root identity mismatch"):
            collect(collector, agent, 1, snapshot_starts=[branch])
        assert collector.stats.micro_steps == 0


def terminal_siblings(agent: ActorCritic) -> RolloutBatch:
    """Three two-seat continuations plus an independent trajectory pair."""
    recipe, _ = replay_root(agent.observation_space, terminal=True)
    with replay_collector(agent.observation_space, recipe) as collector:
        batch = collect(
            collector,
            agent,
            12,
            snapshot_starts=[
                SnapshotBranch(
                    7,
                    recipe,
                    horizon=3,
                    branch_group_id=9 if i < 3 else -1,
                    branch_index=i,
                )
                for i in range(4)
            ],
        )
    rewards = batch.rewards.copy()
    for i, outcome in enumerate([1, 0, -1, 1]):
        rewards[2 * i + 1, 0] = outcome
        rewards[8 + i, 0] = -outcome
    return replace(batch, rewards=rewards)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_sibling_roots_use_empirical_returns_and_leave_descendants_unchanged(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = small_agent(kind=kind)
    batch = terminal_siblings(agent)
    learner = Learner(agent, "cpu", minibatch_rows=12)
    learning = DampedPolicyLearning(
        gradient="damped_policy", policy_lambda=0.1, value_lambda=0.2
    )
    action_values = np.arange(len(batch.action_data.slots), dtype=np.float32)
    targets = compute_batch_targets(
        backend,
        learner,
        AdmittedRollout(agent, batch),
        learning,
        action_values=action_values,
    )
    ordinary = torch_targets(learner, replace(batch, branch_group_id=None), learning)
    roots = [0, 2, 4]
    others = np.setdiff1d(np.arange(12), roots)
    assert torch.equal(targets.advantages[roots, 0], torch.tensor([1.5, 0, -1.5]))
    expected = torch.zeros(3) if kind == "scalar" else torch.full((3, 3), 1 / 3)
    assert torch.equal(targets.value_targets[roots, 0], expected)
    assert torch.equal(targets.advantages[others], ordinary.advantages[others])
    assert torch.equal(targets.value_targets[others], ordinary.value_targets[others])
    signals = targets.signals
    np.testing.assert_allclose(
        signals.critic_residual, np.abs(targets.residuals.numpy().ravel())
    )
    np.testing.assert_allclose(
        signals.positive_value_loss, targets.residuals.numpy().ravel().clip(0)
    )
    np.testing.assert_array_equal(
        signals.action_value_spread, np.diff(batch.action_data.offsets) - 1
    )
    assert not signals.critic_residual.flags.writeable
    logs = batch.action_data.log_probabilities
    for row, (a, b) in enumerate(
        zip(batch.action_data.offsets[:-1], batch.action_data.offsets[1:], strict=True)
    ):
        assert signals.policy_entropy[row] == pytest.approx(
            -np.dot(np.exp(logs[a:b]), logs[a:b])
        )
    if kind == "categorical_wdl":
        probs = batch.outcome_probabilities.reshape(-1, 3)
        np.testing.assert_allclose(
            signals.wdl_entropy, -(probs * np.log(probs)).sum(-1)
        )
    else:
        assert signals.wdl_entropy is None
    assert ordinary.signals.action_value_spread is None
    # Supplying frozen targets must not run the bootstrap or either estimator again.
    with patch.object(backend.rl, "transition_gae", side_effect=AssertionError):
        result = update_damped_iteration(
            backend, learner, batch, learning, 1, targets=targets
        )
    assert result["optimizer_exposures"] > 0 and batch.target_residuals is None


def cut_siblings(agent: ActorCritic) -> RolloutBatch:
    recipe, _ = replay_root(agent.observation_space)
    with replay_collector(agent.observation_space, recipe) as collector:
        return collect(
            collector,
            agent,
            4,
            snapshot_starts=[
                SnapshotBranch(7, recipe, horizon=h, branch_group_id=9, branch_index=i)
                for i, h in enumerate((3, 1))
            ],
        )


def test_grouped_ppo_discounts_root_viewer_cuts_and_validates_identity() -> None:
    agent = small_agent()
    _, long_tail = three_choices(agent)
    _, short_tail = three_choices(agent, horizon=1)
    batch = cut_siblings(agent)
    spans = batch.trajectories
    assert spans is not None
    batch = replace(batch, rewards=np.array([[0.1], [0.2], [0.3], [-0.1]], np.float32))
    learning = Learning(gamma=0.5, epochs=1, minibatches=1, retained_fraction=1)
    learner = Learner(agent, "cpu", minibatch_rows=3)
    targets = torch_targets(learner, batch, learning)
    g0 = 0.1 + 0.5 * (0.2 + 0.5 * long_tail[0].item())
    g1 = 0.3 + 0.5 * short_tail[0].item()
    np.testing.assert_allclose(
        targets.advantages[[0, 2], 0], [g0 - g1, g1 - g0], atol=1e-7
    )
    np.testing.assert_allclose(
        targets.value_targets[[0, 2], 0], (g0 + g1) / 2, atol=1e-7
    )
    with patch.object(backend.rl, "transition_gae", side_effect=AssertionError):
        update_iteration(
            backend,
            learner,
            batch,
            learning,
            0,
            np.random.default_rng(2),
            targets=targets,
        )
    for mixed in (
        replace(batch, trajectories=(spans[0], replace(spans[1], seat=1), spans[2])),
        replace(batch, snapshot_id=np.array([[7], [7], [8], [7]], np.int64)),
    ):
        with pytest.raises(ValueError, match="snapshot ID and seat"):
            torch_targets(learner, mixed, learning)


def test_wdl_group_at_a_cut_averages_frozen_root_viewer_distributions() -> None:
    agent = small_agent(kind="categorical_wdl")
    _, long_tail = three_choices(agent)
    _, short_tail = three_choices(agent, horizon=1)
    batch = cut_siblings(agent)
    root_distributions = torch.stack([long_tail[0], short_tail[0]]).softmax(-1)
    signed = root_distributions[:, 2] - root_distributions[:, 0]
    learning = DampedPolicyLearning(gradient="damped_policy", value_lambda=0)
    targets = compute_batch_targets(
        backend,
        Learner(agent, "cpu", minibatch_rows=4),
        AdmittedRollout(agent, batch),
        learning,
    )
    torch.testing.assert_close(
        targets.value_targets[[0, 2], 0], root_distributions.mean(0).expand(2, -1)
    )
    torch.testing.assert_close(
        targets.advantages[[0, 2], 0], signed - signed.flip(dims=(0,))
    )
    assert not torch.allclose(root_distributions[0], root_distributions[1])


@pytest.mark.parametrize("kind", ["top_count", "quantile"])
def test_mixed_sources_filter_eligible_rows_independently(
    kind: Literal["top_count", "quantile"],
) -> None:
    agent = small_agent()
    batch = terminal_siblings(agent)
    shape = batch.actions.shape
    weights = np.ones(shape, np.float32)
    weights[[5, 11]] = 0
    batch = replace(
        batch,
        branch_group_id=None,
        snapshot_id=None,
        is_root_decision=None,
        values=np.zeros(shape, np.float32),
        dones=np.ones(shape, bool),
        rewards=np.array([1, 2, 3, 3, 4, 500, 10, 20, 30, 30, 40, 5000], np.float32)[
            :, None
        ],
        episode_source=np.repeat([0, 1], 6).astype(np.int8)[:, None],
        sample_weight=weights,
        trajectories=tuple(TrajectorySpan(i, i + 1, 0) for i in range(12)),
        next_done=np.empty(0, bool),
        next_obs={},
    )
    learning = Learning(
        epochs=1, minibatches=1, filter_kind=kind, retained_fraction=0.4
    )
    learner = Learner(agent, "cpu", minibatch_rows=12)
    targets = torch_targets(learner, batch, learning)
    expected = [10, 8, 4, 2] if kind == "top_count" else [2, 3, 4, 8, 9, 10]
    assert targets.retained.tolist() == expected
    result = update_iteration(
        backend, learner, batch, learning, 0, np.random.default_rng(4), targets=targets
    )
    for source, threshold in (("fresh", 3), ("snapshot", 30)):
        n = len(expected) // 2
        assert result["source_selection"][source] == dict(
            eligible=5, retained=n, retention_rate=n / 5, threshold=threshold
        )


@pytest.mark.parametrize("rule", ["ppo", "damped", "mlx"])
def test_weighted_steps_match_hand_weighted_actor_and_critic(rule: str) -> None:
    """One fractional-weight step agrees with an independent full-batch Adam step."""
    if rule == "mlx":
        pytest.importorskip("mlx.core")
    agent = scalar_backend_agent()
    batch, _ = three_choices(agent, terminal=True)
    batch = replace(batch, sample_weight=np.array([[0.2], [0.3], [0]], np.float32))
    # Separate current and behavior policies so the KL terms carry gradient.
    with torch.no_grad():
        agent.policy_head[-1].weight.mul_(8)
    reference = Learner(deepcopy(agent), "cpu", minibatch_rows=3)
    learning = (
        Learning(root_actor_weight=3, retained_fraction=1, epochs=1, minibatches=1)
        if rule == "ppo"
        else DampedPolicyLearning(
            gradient="damped_policy",
            root_actor_weight=3,
            advantage_quantile=0,
            min_advantage=0,
            reference_forward_kl=0.1,
        )
    )
    targets = torch_targets(reference, batch, learning)
    indices = np.array([0, 1])
    rows = batch.observations(indices)
    logits, values = reference.agent(device_input(reference.agent, rows.observation))
    valid = torch.tensor(rows.observation["actions_valid"], dtype=torch.bool)
    behavior_logs = torch.tensor(rows.log_probabilities)
    magnet = torch.tensor(reference_distribution(rows.observation, learning.reference))
    actions = torch.tensor(batch.actions[indices, 0])
    advantages = targets.advantages[indices, 0]
    value_loss = (values - targets.value_targets[indices, 0]).square()
    if isinstance(learning, Learning):
        rate, tau = learning.learning_rate.at(0), learning.tau.at(0)
        logs = backend.rl.policy_logs(logits, valid)
        advantages = backend.rl.standardized_advantages(advantages)
        policy_loss = backend.rl.clipped_surrogate(
            backend.rl.action_logs(logs - behavior_logs, actions),
            advantages,
            learning.clip,
        )
        policy_loss += tau * backend.rl.reverse_kl(
            logs, backend.rl.reference_logs(magnet, valid), valid
        )
        policy_loss += learning.collection_kl * backend.rl.reverse_kl(
            logs, behavior_logs, valid
        )
        value_loss *= 0.5 * learning.value_weight
    else:
        rate, tau = learning.rates(1)
        policy_loss = backend.rl.damped_policy_loss(
            logits,
            actions,
            advantages,
            behavior_logs,
            magnet,
            valid,
            clip=learning.clip,
            collection_kl=learning.collection_kl,
            tau=tau,
            reference_forward_kl=learning.reference_forward_kl,
        )
    # The root has triple actor weight. The critic still sees the original 2:3 ratio.
    expected_loss = (2 * policy_loss[0] + policy_loss[1]) / 3 + (
        2 * value_loss[0] + 3 * value_loss[1]
    ) / 5
    reference.set_learning_rate(rate)
    reference.step(expected_loss, learning.max_grad_norm)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(reference.agent.parameters(), agent.parameters(), strict=True)
    )

    if rule == "mlx":
        mlx = resolve("mlx")
        checkpoint = scalar_checkpoint(agent)
        learner = mlx.learner.Learner(mlx.load(checkpoint)[0], "cpu", minibatch_rows=3)
        result = update_damped_iteration(mlx, learner, batch, learning, 1)
        expected = mlx.parameter_layout(canonical_parameters(reference.parameters()))
        for key, actual in learner.parameters().items():
            np.testing.assert_allclose(
                actual, expected[key], atol=1e-6, rtol=0, err_msg=key
            )
    else:
        for chunk in (None, 1):
            learner = Learner(
                deepcopy(agent), "cpu", minibatch_rows=3, execution_rows=chunk
            )
            if isinstance(learning, Learning):
                result = update_iteration(
                    backend, learner, batch, learning, 0, np.random.default_rng(7)
                )
            else:
                result = update_damped_iteration(backend, learner, batch, learning, 1)
            for actual, expected_param in zip(
                learner.agent.parameters(), reference.agent.parameters(), strict=True
            ):
                torch.testing.assert_close(actual, expected_param, atol=1e-6, rtol=0)
            assert result["numerical"]["optimizer_steps"] == 1
    assert result["actor_exposures"] == result["critic_exposures"] == 2
    assert result["loss"] == pytest.approx(expected_loss.item(), abs=1e-6)


@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_damped_minibatches_preserve_source_and_root_weights(framework: str) -> None:
    if framework == "mlx":
        pytest.importorskip("mlx.core")
    agent = scalar_backend_agent()
    recipe, _ = replay_root(agent.observation_space)
    branch = SnapshotBranch(7, recipe, horizon=2)
    with replay_collector(agent.observation_space, recipe) as collector:
        batch = collect(collector, agent, 4, snapshot_starts=[branch, branch])
    batch = replace(
        batch,
        episode_source=np.array([0, 0, 1, 1], np.int8)[:, None],
        snapshot_id=np.array([-1, -1, 7, 7], np.int64)[:, None],
        is_root_decision=np.array([False, False, True, False])[:, None],
        sample_weight=np.array([0.1, 0.1, 1, 1], np.float32)[:, None],
    )
    with torch.no_grad():
        agent.policy_head[-1].weight.mul_(8)
    reference = Learner(deepcopy(agent), "cpu", minibatch_rows=2)
    learning = DampedPolicyLearning(
        gradient="damped_policy",
        advantage_quantile=0,
        min_advantage=0,
        root_actor_weight=4,
        max_grad_norm=100,
    )
    targets = torch_targets(reference, batch, learning)
    rate, tau = learning.rates(1)
    reference.set_learning_rate(rate)
    actor_weights = torch.tensor([0.1, 0.1, 4, 1])
    critic_weights = torch.tensor([0.1, 0.1, 1, 1])
    for ids in (np.array([0, 1]), np.array([2, 3])):
        rows = batch.observations(ids)
        logits, values = reference.agent(
            device_input(reference.agent, rows.observation)
        )
        mask = torch.tensor(rows.observation["actions_valid"], dtype=torch.bool)
        behavior = torch.tensor(rows.log_probabilities)
        magnet = torch.tensor(
            reference_distribution(rows.observation, learning.reference)
        )
        actor_loss = backend.rl.damped_policy_loss(
            logits,
            torch.tensor(batch.actions[ids, 0]),
            targets.advantages[ids, 0],
            behavior,
            magnet,
            mask,
            clip=learning.clip,
            collection_kl=learning.collection_kl,
            tau=tau,
            reference_forward_kl=learning.reference_forward_kl,
        )
        critic_loss = (values - targets.value_targets[ids, 0]).square()
        expected = (
            actor_loss * actor_weights[ids] / actor_weights.mean()
            + critic_loss * critic_weights[ids] / critic_weights.mean()
        ).mean()
        logs = backend.rl.policy_logs(logits, mask)
        collection_kl = backend.rl.reverse_kl(logs, behavior, mask).mean().item()
        reference_kl = (
            backend.rl.reverse_kl(logs, backend.rl.reference_logs(magnet, mask), mask)
            .mean()
            .item()
        )
        reference.step(expected, learning.max_grad_norm)

    runtime = resolve(framework)
    checkpoint = scalar_checkpoint(agent)
    learner = runtime.learner.Learner(
        runtime.load(checkpoint)[0], "cpu", minibatch_rows=2
    )
    result = update_damped_iteration(runtime, learner, batch, learning, 1)
    assert result["numerical"]["optimizer_steps"] == 2
    assert result["loss"] == pytest.approx(expected.item(), abs=1e-6)
    assert result["collection_kl"] == pytest.approx(collection_kl, abs=1e-6)
    assert result["reference_kl"] == pytest.approx(reference_kl, abs=1e-6)
    expected_params = reference.parameters()
    if framework == "mlx":
        expected_params = runtime.parameter_layout(
            canonical_parameters(expected_params)
        )
    for name, actual in learner.parameters().items():
        np.testing.assert_allclose(actual, expected_params[name], atol=1e-6, rtol=0)

    uniform = runtime.learner.Learner(
        runtime.load(checkpoint)[0], "cpu", minibatch_rows=2
    )
    control = update_damped_iteration(
        runtime,
        uniform,
        replace(batch, sample_weight=None),
        learning.model_copy(update={"root_actor_weight": 1}),
        1,
    )
    assert result["loss"] != pytest.approx(control["loss"])


@pytest.mark.parametrize("rule", ["ppo", "damped"])
def test_zero_weight_discovery_preserves_live_adam_and_ema(rule: str) -> None:
    learning = (
        Learning(
            horizon=2,
            root_actor_weight=0,
            ema=0.5,
            filter_scope="actor",
            epochs=1,
            minibatches=1,
        )
        if rule == "ppo"
        else DampedPolicyLearning(
            gradient="damped_policy", root_actor_weight=0, filter_scope="actor"
        )
    )
    settings = gray_ogre_self_play(horizon=2).model_copy(update={"learning": learning})
    agent = small_agent()
    learner = Learner(agent, "cpu", minibatch_rows=3)
    ema = deepcopy(agent)
    batch, _ = three_choices(agent, terminal=True)
    # Only the root contributes to the critic. Its actor weight is zero.
    weighted = replace(batch, sample_weight=np.array([[1], [0], [0]], np.float32))
    before = learner.parameters()
    result = learn(settings, learner, ema, Counts(), weighted, np.random.default_rng(0))
    assert result["actor_exposures"] == 0 and result["critic_exposures"] == 1
    after = learner.parameters()
    assert any(
        not np.array_equal(after[k], before[k]) for k in after if k.startswith("value_")
    )
    assert all(
        np.array_equal(after[k], before[k])
        for k in after
        if k.startswith("policy_head.")
    )
    ema_before = deepcopy(ema.state_dict())
    adam_before = deepcopy(learner.optimizer.state_dict())
    assert adam_before["state"]
    discovery = replace(batch, sample_weight=np.zeros_like(batch.rewards))
    targets = torch_targets(learner, discovery, learning)
    assert (
        targets.signals.policy_entropy.shape == (3,)
        and np.isfinite(targets.signals.critic_residual).all()
    )
    result = learn(
        settings,
        learner,
        ema,
        Counts(updates=1),
        discovery,
        np.random.default_rng(0),
        targets=targets,
    )
    assert (
        result["skipped"] == "zero sample weight" and result["optimizer_exposures"] == 0
    )
    assert result["source_selection"]["snapshot"] == dict(eligible=0, retained=0)
    torch.testing.assert_close(
        learner.optimizer.state_dict(), adam_before, rtol=0, atol=0
    )
    for key, value in learner.parameters().items():
        np.testing.assert_array_equal(value, after[key])
    torch.testing.assert_close(ema.state_dict(), ema_before, rtol=0, atol=0)

    # The same positive-weight root can empty the joint actor/critic filter.
    # This is an ordinary filtered update, so EMA still follows the learner.
    joint = settings.model_copy(
        update={
            "learning": learning.model_copy(update={"filter_scope": "actor_critic"})
        }
    )
    result = learn(
        joint, learner, ema, Counts(updates=1), weighted, np.random.default_rng(0)
    )
    assert (
        result["skipped"] == "empty advantage filter"
        and result["optimizer_exposures"] == 0
    )
    assert any(
        not torch.equal(after_ema, ema_before[name])
        for name, after_ema in ema.state_dict().items()
    )


def test_mlx_scalar_critic_only_step_and_zero_weight_adam_state() -> None:
    pytest.importorskip("mlx.core")
    mlx = resolve("mlx")
    agent = scalar_backend_agent()
    batch, _ = three_choices(agent, terminal=True)
    learner = mlx.learner.Learner(
        mlx.load(scalar_checkpoint(agent))[0], "cpu", minibatch_rows=3
    )
    rule = DampedPolicyLearning(
        gradient="damped_policy", filter_scope="actor", root_actor_weight=0
    )
    before = learner.parameters()
    weighted = replace(batch, sample_weight=np.array([[0.1], [0], [0]], np.float32))
    result = update_damped_iteration(mlx, learner, weighted, rule, 1)
    assert result["actor_exposures"] == 0 and result["critic_exposures"] == 1
    after = learner.parameters()
    changed = {k for k in after if not np.array_equal(after[k], before[k])}
    assert any(k.startswith("value_") for k in changed)
    assert not any(k.startswith("policy_head.") for k in changed)
    adam = mlx.arrays.flat_numpy(learner.optimizer.state)
    discovery = replace(batch, sample_weight=np.zeros_like(batch.rewards))
    result = update_damped_iteration(mlx, learner, discovery, rule, 2)
    assert (
        result["skipped"] == "zero sample weight" and result["optimizer_exposures"] == 0
    )
    for key, value in learner.parameters().items():
        np.testing.assert_array_equal(value, after[key])
    for key, value in mlx.arrays.flat_numpy(learner.optimizer.state).items():
        np.testing.assert_array_equal(value, adam[key])


def test_saved_positional_bank_keeps_its_identity_and_warms_once() -> None:
    space = ObservationSpace()
    recipe, _ = replay_root(space)
    config = SnapshotConfig(capacity=1, replay_fraction=1, fresh_floor=0)
    # The saved entry predates the native replay field. Its signed rollout
    # identity was computed from exactly these serialized keys.
    saved_recipe = {
        "match": recipe.match.model_dump(mode="json"),
        "actions": list(recipe.actions),
        "seed": recipe.seed,
        "root_digest": recipe.root_digest,
        "root_chooser": recipe.root_chooser,
        "decision_memory": recipe.decision_memory,
        "decision_memory_include_own": recipe.decision_memory_include_own,
    }
    key = int(canonical_sha256(saved_recipe)[:16], 16) >> 1
    state = SnapshotBank(config).state().model_dump(mode="json")
    state["entries"] = [
        dict(
            snapshot_id=key,
            recipe=saved_recipe,
            score=1.0,
            score_generation=0,
            last_visit_generation=0,
        )
    ]
    bank = SnapshotBank.from_state(config, SnapshotBankState.model_validate(state))
    assert bank.state().model_dump(mode="json")["entries"][0]["recipe"] == saved_recipe
    with (
        replay_collector(space, recipe, num_envs=2) as collector,
        patch(
            "manabot.training.rollout.snapshot_vector.replay_snapshot",
            wraps=replay_snapshot,
        ) as replayed,
    ):
        requests = [SlotInstall(slot, key, recipe) for slot in (1, 0)]
        for _ in range(2):
            installed = bank.install(collector._env, requests)
            assert [(s.slot, s.root_digest) for s in installed] == [
                (1, recipe.root_digest),
                (0, recipe.root_digest),
            ]
        replayed.assert_called_once_with(recipe)
    bank.close()


def test_cold_bank_matches_hot_draws_scores_evictions_and_group_draining() -> None:
    agent = scalar_backend_agent()
    space = agent.observation_space
    recipe, _ = replay_root(space)
    config = SnapshotConfig(
        capacity=2,
        replay_fraction=1,
        fresh_floor=0,
        group_size=5,
        horizon=2,
        fresh_weight=0,
        score_half_life=0.1,
        capture=CapturePolicy(
            every_n_decisions=2, max_captures_per_batch=8, fresh_only=False
        ),
    )
    learning = DampedPolicyLearning(gradient="damped_policy")
    learner = Learner(agent, "cpu", minibatch_rows=16)
    # Both native vectors follow the same fresh stream; only one bank forgets its handles.
    with (
        replay_collector(space, recipe, num_envs=2) as left,
        replay_collector(space, recipe, num_envs=2) as right,
    ):
        banks = [SnapshotBank(config), SnapshotBank(config)]
        warm_ids: set[int] = set()
        for generation in range(1, 4):
            retained = []
            assignments = []
            # The warm bank also moves to a different producer after capture.
            collectors = (left, right) if generation == 1 else (right, left)
            for bank, collector in zip(banks, collectors, strict=True):
                with (
                    bank.begin_batch() as plan,
                    patch(
                        "manabot.training.rollout.snapshot_vector.replay_snapshot",
                        wraps=replay_snapshot,
                    ) as replayed,
                    patch.object(
                        collector._env,
                        "install_slots",
                        wraps=collector._env.install_slots,
                    ) as installs,
                ):
                    batch = collect(collector, agent, 1, snapshot_plan=plan)
                    receipt = plan.receipt
                    assert receipt is not None
                    targets = torch_targets(learner, batch, learning)
                    prepared = bank.prepare_commit(receipt, targets.signals)
                    assert bank.generation == generation - 1
                    if generation == 1:
                        assert not np.any(batch.sample_weight)
                        metrics = update_damped_iteration(
                            backend, learner, batch, learning, 1, targets=targets
                        )
                        assert metrics["optimizer_exposures"] == 0
                        # Every pre-step capture maps to its actual packed chooser row.
                        for candidate in receipt.captures:
                            captured = candidate.snapshot
                            span = next(
                                s
                                for s in batch.trajectories
                                if s.start <= candidate.row < s.stop
                            )
                            assert span.seat == captured.root_chooser
                            assert candidate.row in candidate.episode_rows
                            game = captured.recipe
                            scalar = managym.Env(
                                seed=game.seed,
                                decision_memory=game.decision_memory,
                                decision_memory_include_own=game.decision_memory_include_own,
                            )
                            observation, _ = scalar.reset(Match(game.match).to_rust())
                            for action in game.actions:
                                observation, *_ = scalar.step(action)
                            assert scalar.state_digest() == game.root_digest
                            expected = encode_observation(space, observation)
                            actual = batch.observations(
                                np.array([candidate.row])
                            ).observation
                            for key, value in projected_fields(actual).items():
                                np.testing.assert_array_equal(value[0], expected[key])
                        assert len(receipt.captures) > config.capacity
                    else:
                        [visit] = receipt.visited_groups
                        assert len(visit.roots) == 5 and len(visit.rows) == 10
                        assert np.all(batch.is_root_decision.ravel()[list(visit.roots)])
                        assert batch.actions.size == 10 and np.all(
                            batch.sample_weight == 1
                        )
                        # A row floor of two drains K=5 on both slots in three waves.
                        requests = [call.args[0] for call in installs.call_args_list]
                        assert [len(wave) for wave in requests] == [2, 2, 1]
                        assert replayed.call_count == int(
                            generation == 2 and bank is banks[1]
                        )
                        handles = [req.hot_handle for wave in requests for req in wave]
                        assert isinstance(handles[0], managym.GameSnapshot)
                        assert all(h is handles[0] for h in handles)
                    assignments.append(
                        [
                            (v.snapshot_id, v.group_id, v.roots)
                            for v in receipt.visited_groups
                        ]
                    )
                    bank.publish(prepared)
                    with pytest.raises(ValueError, match="stale"):
                        bank.publish(prepared)
                    state = bank.state()
                    assert (
                        state.generation == generation
                        and len(state.entries) == config.capacity
                    )
                    retained.append(state.model_dump_json())
            assert retained[0] == retained[1] and assignments[0] == assignments[1]
            if generation == 2:
                assert warm_ids - {e.snapshot_id for e in banks[0].state().entries}
            if generation == 1:
                warm_ids = {e.snapshot_id for e in banks[0].state().entries}
                banks[1].close()
                banks[1] = SnapshotBank.from_state(
                    config, SnapshotBankState.model_validate_json(retained[0])
                )
        for bank in banks:
            bank.close()


def test_failed_episodes_discard_captures_siblings_and_frozen_views() -> None:
    agent = scalar_backend_agent()
    space = agent.observation_space
    recipe, _ = replay_root(space)
    config = SnapshotConfig(
        capacity=4,
        replay_fraction=1,
        fresh_floor=0,
        group_size=2,
        horizon=1,
        max_group_retries=1,
        capture=CapturePolicy(
            every_n_decisions=1, max_captures_per_batch=5, fresh_only=False
        ),
    )
    bank = SnapshotBank(config)
    policy = GameErrorPolicy()
    attempts: list[GameAttempt] = []

    def record(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
        attempts.append(attempt)
        return settle(state, attempt, policy)

    with replay_collector(
        space, recipe, game_errors=policy, record_attempt=record
    ) as collector:
        info = collector._env.get_last_info

        def fail_selected() -> list[dict[str, Any]]:
            result = info()
            branch = collector._episodes[0].branch
            if not attempts or (
                branch and branch.branch_group_id == 0 and branch.branch_index == 1
            ):
                result[0] = dict(result[0], game_error="injected game failure")
            return result

        with (
            patch.object(collector._env, "get_last_info", side_effect=fail_selected),
            patch.object(
                collector._env, "release_handles", wraps=collector._env.release_handles
            ) as release,
        ):
            learner = Learner(agent, "cpu", minibatch_rows=16)
            for generation in (1, 2):
                with bank.begin_batch() as plan:
                    batch = collect(collector, agent, 1, snapshot_plan=plan)
                    receipt = plan.receipt
                    assert receipt is not None
                    if generation == 1:
                        failure, success = attempts
                        assert (
                            failure.outcome == "engine_error"
                            and success.outcome == "completed"
                        )
                        assert {c.snapshot.recipe.seed for c in receipt.captures} == {
                            success.seed
                        }
                        assert success.seed != failure.seed
                        assert len(receipt.captures) == 4
                        release.assert_not_called()
                    else:
                        assert [a.outcome for a in attempts[2:]] == [
                            "engine_error",
                            "engine_error",
                        ]
                        assert (
                            collector.stats.snapshot.failed_branches == 2
                            and plan.discarded == {0}
                        )
                        [visit] = receipt.visited_groups
                        assert visit.group_id == 1
                        assert batch.branch_group_id.ravel().tolist() == [1, 1]
                        assert len(batch.next_done) == 2
                        assert len(receipt.captures) == 2
                        assert (
                            receipt.next_scheduler_state.starts == 7
                        )  # two fresh + three failed-group attempts + two siblings
                    targets = torch_targets(learner, batch, Learning())
                    bank.commit(receipt, targets.signals)
        assert sum(e.visits for e in bank.state().entries) == 1
        bank.close()


def test_numerical_error_and_unavailable_signal_leave_bank_unchanged() -> None:
    agent = scalar_backend_agent()
    recipe, _ = replay_root(agent.observation_space)
    config = SnapshotConfig(
        capacity=2,
        replay_fraction=0.5,
        fresh_floor=0.5,
        capture=CapturePolicy(every_n_decisions=4, max_captures_per_batch=2),
    )
    bank = SnapshotBank(config)
    learner = Learner(agent, "cpu", minibatch_rows=16)
    learning = DampedPolicyLearning(gradient="damped_policy")
    with (
        replay_collector(agent.observation_space, recipe) as collector,
        patch.object(
            collector._env, "release_handles", wraps=collector._env.release_handles
        ) as released,
    ):
        with bank.begin_batch() as warmup:
            batch = collect(collector, agent, 1, snapshot_plan=warmup)
            targets = torch_targets(learner, batch, learning)
            assert warmup.receipt is not None
            bank.commit(warmup.receipt, targets.signals)
        before = bank.state().model_dump_json()
        capture = collector._env.capture_slots
        owned = []

        def custom_capture(slots: Sequence[int]) -> list[CapturedSnapshot]:
            captures = capture(slots)
            for i, item in enumerate(captures):
                handle = object()
                owned.append(handle)
                captures[i] = replace(item, hot_handle=handle)
            return captures

        with (
            pytest.raises(NumericalError),
            patch.object(collector._env, "capture_slots", side_effect=custom_capture),
            bank.begin_batch() as plan,
        ):
            batch = collect(collector, agent, 1, snapshot_plan=plan)
            assert plan.receipt is not None and len(plan.receipt.captures) == 2
            batch.rewards[0, 0] = float("nan")
            update_damped_iteration(backend, learner, batch, learning, 1)
        assert len(owned) == 2
        assert [h for call in released.call_args_list for h in call.args[0]] == owned
        assert bank.state().model_dump_json() == before
        bank.close()
        bank = SnapshotBank(config.model_copy(update={"signal": "wdl_entropy"}))
        before = bank.state().model_dump_json()
        with (
            pytest.raises(ValueError, match="wdl_entropy unavailable"),
            bank.begin_batch() as plan,
        ):
            batch = collect(collector, agent, 1, snapshot_plan=plan)
            targets = torch_targets(learner, batch, learning)
            assert plan.receipt is not None
            bank.commit(plan.receipt, targets.signals)
        assert bank.state().model_dump_json() == before


def test_bank_decays_old_scores_and_spends_only_whole_group_allowance() -> None:
    agent = scalar_backend_agent()
    recipe, _ = replay_root(agent.observation_space)
    config = SnapshotConfig(
        capacity=2,
        replay_fraction=0.9,
        fresh_floor=0.5,
        group_size=3,
        score_half_life=1,
        capture=CapturePolicy(every_n_decisions=1, max_captures_per_batch=1),
    )

    def score(entry: EntrySignals) -> float:
        assert entry.root_signals.critic_residual.shape == (1,)
        assert (
            len(entry.episode_signals.critic_residual) > 1
            and not entry.root_signals.critic_residual.flags.writeable
        )
        return float(entry.root_signals.critic_residual[0])

    bank = SnapshotBank(config, scorer=score)
    learner = Learner(agent, "cpu", minibatch_rows=16)
    ids = []
    with replay_collector(agent.observation_space, recipe) as collector:
        # A score of 8 from generation one decays to 2 by generation three.
        for generation, value in enumerate([8.0, 6.0, 2.5], 1):
            with bank.begin_batch() as plan:
                batch = collect(collector, agent, 1, snapshot_plan=plan)
                receipt = plan.receipt
                assert receipt is not None
                [candidate] = receipt.captures
                ids.append(root_id(candidate.snapshot.recipe))
                targets = torch_targets(learner, batch, Learning())
                residual = np.zeros(batch.actions.size)
                residual[candidate.row] = value
                bank.commit(receipt, replace(targets.signals, critic_residual=residual))
                state = bank.state()
                if generation == 1:
                    assert state.scheduler.allowance == 0
                if generation == 2:
                    first = next(e for e in state.entries if e.snapshot_id == ids[0])
                    assert first.priority(config, 2) == 4
        assert {e.snapshot_id for e in bank.state().entries} == set(ids[1:])
        with bank.begin_batch() as plan:
            # Two fresh post-warmup starts are in the ledger. A third funds K=3.
            assert plan.next_episode() is None
            starts = [plan.next_episode() for _ in range(12)]
            assert [b.branch_index if b else None for b in starts] == [
                0,
                1,
                2,
                None,
                None,
                None,
                0,
                1,
                2,
                None,
                None,
                None,
            ]
            assert starts[0].snapshot_id != starts[6].snapshot_id
        bank.close()


def test_full_staleness_bank_keeps_new_capture_and_evicts_oldest() -> None:
    agent = scalar_backend_agent()
    learner = Learner(agent, "cpu", minibatch_rows=8)
    recipe, _ = replay_root(agent.observation_space)
    bank = SnapshotBank(
        SnapshotConfig(
            capacity=2,
            replay_fraction=0,
            fresh_floor=1,
            signal="staleness",
            capture=CapturePolicy(every_n_decisions=1, max_captures_per_batch=1),
        )
    )
    captured = []
    with replay_collector(agent.observation_space, recipe) as collector:
        for _ in range(3):
            with bank.begin_batch() as plan:
                batch = collect(collector, agent, 1, snapshot_plan=plan)
                assert plan.receipt is not None
                [candidate] = plan.receipt.captures
                captured.append(root_id(candidate.snapshot.recipe))
                targets = torch_targets(learner, batch, Learning())
                bank.commit(plan.receipt, targets.signals)
                assert {e.snapshot_id for e in bank.state().entries} == set(
                    captured[-2:]
                )
        assert len(set(captured)) == 3
        bank.close()


def test_eight_streams_launch_funded_pairs_and_resume_retry_debt() -> None:
    agent = scalar_backend_agent()
    learner = Learner(agent, "cpu", minibatch_rows=8)
    recipe, _ = replay_root(agent.observation_space)
    config = SnapshotConfig(
        capacity=1,
        replay_fraction=0.5,
        fresh_floor=0.5,
        group_size=2,
        horizon=2,
        capture=CapturePolicy(every_n_decisions=1, max_captures_per_batch=1),
    )
    bank = SnapshotBank(config)
    recovery = GameErrorPolicy()
    with replay_collector(
        agent.observation_space,
        recipe,
        num_envs=8,
        game_errors=recovery,
        record_attempt=lambda state, attempt: settle(state, attempt, recovery),
    ) as collector:
        with bank.begin_batch() as warmup:
            batch = collect(collector, agent, 1, snapshot_plan=warmup)
            targets = torch_targets(learner, batch, Learning())
            assert warmup.receipt is not None
            bank.commit(warmup.receipt, targets.signals)
        get_info = collector._env.get_last_info
        failed = False

        def fail_third_start_once() -> list[dict[str, Any]]:
            nonlocal failed
            info = get_info()
            episode = collector._episodes[2]
            if not failed and episode.branch is not None and episode.choices == 1:
                assert episode.branch.branch_index == 0
                info[2] = dict(info[2], game_error="retry the first funded branch")
                failed = True
            return info

        with (
            bank.begin_batch() as plan,
            patch.object(
                collector._env, "get_last_info", side_effect=fail_third_start_once
            ),
        ):
            batch = collect(collector, agent, 1, snapshot_plan=plan)
            assert failed and plan.receipt is not None
            assert len(plan.receipt.visited_groups) == 2
            targets = torch_targets(learner, batch, Learning())
            bank.commit(plan.receipt, targets.signals)
        saved = SnapshotBankState.model_validate_json(bank.state().model_dump_json())
        assert saved.scheduler.allowance == -1
        cold = SnapshotBank.from_state(config, saved)
        with cold.begin_batch() as plan:
            assert [plan.next_episode() for _ in range(3)] == [None, None, None]
            assert plan.next_episode() is not None
        bank.close()


def test_coverage_cycle_and_random_draws_resume_with_the_saved_rng() -> None:
    agent = scalar_backend_agent()
    recipe, _ = replay_root(agent.observation_space)
    config = SnapshotConfig(
        capacity=3,
        replay_fraction=1,
        fresh_floor=0,
        horizon=1,
        signal="staleness",
        capture=CapturePolicy(every_n_decisions=2, max_captures_per_batch=3),
    )
    bank = SnapshotBank(config)
    learner = Learner(agent, "cpu", minibatch_rows=8)
    with replay_collector(agent.observation_space, recipe) as collector:
        for floor in (1, 8):
            with bank.begin_batch() as plan:
                batch = collect(collector, agent, floor, snapshot_plan=plan)
                targets = torch_targets(learner, batch, Learning())
                assert plan.receipt is not None
                visits = plan.receipt.visited_groups
                if floor == 8:
                    ids = [v.snapshot_id for v in visits]
                    assert len(set(ids[:3])) == 3 and len(ids) == 8
                    assert len(plan.receipt.next_scheduler_state.coverage) == 2
                bank.commit(plan.receipt, targets.signals)
        saved = bank.state()
        restored = SnapshotBank.from_state(
            config, SnapshotBankState.model_validate_json(saved.model_dump_json())
        )
        with bank.begin_batch() as hot, restored.begin_batch() as cold:
            starts = [hot.next_episode() for _ in range(15)]
            assert starts == [cold.next_episode() for _ in starts]
            # Draw five used the first coverage entry. Draws ten and fifteen
            # continue that queue across the saved update boundary.
            assert [starts[1].snapshot_id, starts[6].snapshot_id] == list(
                saved.scheduler.coverage
            )
            for e in saved.entries:
                assert e.priority(config, saved.generation + 3) == 3
        bank.close()


def test_mixed_batch_retries_the_same_sibling_after_the_row_floor() -> None:
    agent = scalar_backend_agent()
    recipe, _ = replay_root(agent.observation_space)
    config = SnapshotConfig(
        capacity=4,
        replay_fraction=0.875,
        fresh_floor=0.125,
        group_size=3,
        horizon=1,
        capture=CapturePolicy(probability=1),
    )
    bank = SnapshotBank(config)
    policy = GameErrorPolicy()
    attempts: list[GameAttempt] = []

    def record(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
        attempts.append(attempt)
        return settle(state, attempt, policy)

    with replay_collector(
        agent.observation_space,
        recipe,
        num_envs=2,
        game_errors=policy,
        record_attempt=record,
    ) as collector:
        collector.capture_predicate = lambda d: d.decision == 3
        learner = Learner(agent, "cpu", minibatch_rows=16)
        with bank.begin_batch() as plan:
            batch = collect(collector, agent, 1, snapshot_plan=plan)
            targets = torch_targets(learner, batch, Learning())
            assert plan.receipt is not None and len(plan.receipt.captures) == 2
            assert all(
                len(c.snapshot.recipe.actions) == 2 for c in plan.receipt.captures
            )
            bank.commit(plan.receipt, targets.signals)

        original = collector._env.get_last_info
        last_sibling = []

        def fail_once() -> list[dict[str, Any]]:
            info = original()
            for slot, episode in enumerate(collector._episodes):
                branch = episode.branch
                if (
                    branch
                    and branch.branch_index == 2
                    and collector._pending[2 * slot + branch.recipe.root_chooser]
                    is not None
                ):
                    last_sibling.append((branch.snapshot_id, branch.branch_group_id))
                    if len(last_sibling) == 1:
                        info[slot] = dict(info[slot], game_error="one failed sibling")
            return info

        with (
            bank.begin_batch() as plan,
            patch.object(collector._env, "get_last_info", side_effect=fail_once),
        ):
            batch = collect(collector, agent, 1, snapshot_plan=plan)
            receipt = plan.receipt
            assert receipt is not None and len(receipt.captures) == 1
            [group] = receipt.visited_groups
            assert len(group.roots) == 3 and len(group.rows) == 3
            assert last_sibling == [(group.snapshot_id, group.group_id)] * 2
            assert collector.stats.snapshot.failed_branches == 1 and not plan.discarded
            assert np.count_nonzero(batch.episode_source == 0) > 3
            assert np.count_nonzero(batch.episode_source == 1) == 3
            assert receipt.next_scheduler_state.starts == 2 + 5
            targets = torch_targets(learner, batch, Learning())
            bank.commit(receipt, targets.signals)
        bank.close()


@pytest.mark.parametrize(("replay_fraction", "horizon"), [(1.0, 3), (0.75, None)])
def test_native_snapshot_self_play(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_fraction: float,
    horizon: int | None,
) -> None:
    """Resume the saved warmup and exercise replay-only cuts or mixed complete games."""
    base = gray_ogre_self_play(horizon=2)
    assert (
        "snapshot" not in base.model_dump()
        and "snapshot_bank" not in Counts().model_dump()
    )
    group_size = 4 if horizon is not None else 2
    settings = SelfPlay.model_validate(
        base.model_dump()
        | {
            "learning": Learning(horizon=2, epochs=1, minibatches=1, ema=0.9)
            if horizon is None
            else DampedPolicyLearning(gradient="damped_policy", filter_scope="actor"),
            "workers": 2,
            "streams": group_size,
            "model": base.model.model_copy(
                update={
                    "decision_memory": "gru",
                    "policy_film": True,
                    "world": managym.WORLD_SETUP,
                }
            )
            if horizon
            else base.model,
            "observation": ObservationConfig(
                max_decision_records=128,
                max_decision_references=1024,
                max_decision_reference_details=8192,
                max_decision_members=4096,
            )
            if horizon
            else base.observation,
            "collection_rows": 4,
            "bootstrap_rows": 4,
            "learner_rows": 4,
            "behavior": "ema-self",
            "transitions": 1,
            "games_per_stream": 1,
            "snapshot": SnapshotConfig(
                # Hold warmup entries alongside update 2's fresh captures so
                # untrained critic-residual noise cannot evict the visited root.
                capacity=4,
                group_size=group_size,
                horizon=horizon,
                replay_fraction=replay_fraction,
                fresh_floor=0,
                fresh_weight=0,
                capture=CapturePolicy(every_n_decisions=1, max_captures_per_batch=2),
            ),
        }
    )
    track = tmp_path / "track"
    out = tmp_path / "out"
    monkeypatch.setenv("ETUDE_OUT", str(out))

    def capture(decision: CaptureDecision) -> bool:
        return decision.decision == 2

    def score(entry: EntrySignals) -> float:
        return float(entry.root_signals.critic_residual.mean()) + 1

    counts = []
    for update in (1, 2):
        with recording(
            experiment="snapshot",
            run="resume",
            clocks=CLOCKS,
            directory=track,
            seed=42,
            resume=update == 2,
            strict=True,
        ) as run:
            self_play(
                settings,
                run,
                updates=update,
                snapshot_scorer=score,
                capture_predicate=capture,
            )
            pointer = json.loads((out / "state/current.json").read_bytes())
            saved = json.loads(
                (out / "state" / pointer["save"] / "state.json").read_bytes()
            )
            current = Counts.model_validate(saved["extra"])
            bank = current.snapshot_bank
            assert bank is not None and bank.generation == update
            assert all(
                e.score >= 1 and len(e.recipe.actions) == 1 for e in bank.entries
            )
            assert "snapshot_bank" not in current.coordinates()
            counts.append(current)
            if update == 1:
                assert current.optimizer_exposures == current.optimizer_updates == 0
                assert len(bank.entries) == 2 and all(
                    e.visits == 0 for e in bank.entries
                )
                initial, _ = create(settings, seed=run.seed)
                restored, averaged = create(settings, seed=run.seed + 1)
                assert averaged is not None
                resume_training(settings, run, backend, restored, averaged)
                assert not restored.optimizer.state
                for policy in (restored.agent, averaged):
                    for name, value in policy.state_dict().items():
                        torch.testing.assert_close(
                            value, initial.agent.state_dict()[name], atol=0, rtol=0
                        )
            else:
                assert current.optimizer_exposures > 0 and current.optimizer_updates > 0
                assert any(e.visits for e in bank.entries)

    warm, resumed = counts
    data = read(track, "snapshot", "resume")

    def latest(metric: str) -> float:
        points = data.points(metric, "updates")
        assert points.x[-1] == 2
        return float(points.y[-1])

    assert (
        latest("snapshot/starts")
        == latest("snapshot/groups_completed") * group_size
        > 0
    )
    assert latest("snapshot/choices") == latest("snapshot/rows")
    assert (
        latest("snapshot/groups_discarded") == latest("snapshot/failed_branches") == 0
    )
    assert list(data.points("snapshot/captures", "updates").y)[0] == 2
    rows = resumed.learner_transitions - warm.learner_transitions
    games = resumed.games - warm.games
    outcomes = data.read_table("self_play_outcomes", "updates")
    for update_outcomes, fresh_games in zip(outcomes, (warm.games, games), strict=True):
        assert (
            sum(r["wins"] + r["losses"] + r["draws"] for r in update_outcomes.rows)
            == 2 * fresh_games
        )
    _, game_settings = data.read_table("training_game_settings", "updates")
    assert (
        sum(r["games"] for r in game_settings.rows if r["setting"] == "decklists")
        == games
    )
    assert latest("rows_per_second") > 0 and latest("updates_per_hour") > 0
    if horizon is None:
        assert games > 0 and latest("snapshot/terminals") == latest("snapshot/starts")
        assert latest("snapshot/horizon_cuts") == 0
        assert latest("decisions_per_game") == (rows - latest("snapshot/rows")) / games
    else:
        assert games == 0 and latest("snapshot/terminals") == 0
        assert latest("snapshot/horizon_cuts") == latest("snapshot/starts")
        assert rows == horizon * latest("snapshot/starts")
        for metric in ("decisions_per_game", "skipped_share", "games_per_hour"):
            assert list(data.points(metric, "updates").x) == [1]


def test_snapshot_terminal_leaves_fresh_deck_exposure_and_skip_counts_intact() -> None:
    """Play the same sampled fresh deal with and without a preceding snapshot."""
    space = ObservationSpace()
    agent = ActorCritic(
        space,
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=managym.WORLD_SETUP),
    )
    terminal, _ = replay_root(space, terminal=True)
    decks = UniformDecks(
        decks=tuple(
            managym.Deck({land: 10}) for land in ("Forest", "Mountain", "Island")
        )
    )
    fresh_stats = []
    for with_replay in (False, True):
        collector = open_collector(
            space,
            Match(decks.match_config()),
            Reward(RewardConfig()),
            num_envs=1,
            seed=42,
            deck_source=decks,
            opponent_mode="self",
            games_per_stream=2 if with_replay else 1,
            snapshots=True,
            numerics=CollectionSession(device="cpu", seed=0),
        )
        try:
            with patch.object(
                torch,
                "multinomial",
                side_effect=lambda probs, *args, **kwargs: torch.zeros(
                    (len(probs), 1), dtype=torch.long
                ),
            ):
                batch = collect(
                    collector,
                    agent,
                    1,
                    snapshot_starts=[SnapshotBranch(8, terminal, horizon=3)]
                    if with_replay
                    else [],
                )
            stats = collector.stats
            observed = stats.to_dict()
            assert stats.games == 1
            assert observed["fresh_learner_transitions"] == np.count_nonzero(
                batch.episode_source == 0
            )
            assert observed["snapshot"]["terminals"] == int(with_replay)
            assert observed["snapshot"]["rows"] == (3 if with_replay else 0)
            assert sum(c["game_roots"] for c in stats.exposure.values()) == 1
            assert (
                sum(c["learner_transitions"] for c in stats.exposure.values())
                == stats.fresh_learner_transitions
            )
            fresh_stats.append(
                (
                    stats.exposure,
                    stats.self_play.drain(),
                    stats.skipped_decisions,
                    stats.games_by_stream,
                )
            )
        finally:
            close_collector(collector)
    assert fresh_stats[0] == fresh_stats[1]


@pytest.mark.parametrize("failure", ["exception", "nan"])
def test_scorer_failure_preserves_live_learner_ema_and_bank(
    tmp_path: Path, failure: str
) -> None:
    config = SnapshotConfig(
        capacity=1,
        replay_fraction=0,
        fresh_floor=1,
        capture=CapturePolicy(every_n_decisions=1, max_captures_per_batch=1),
    )
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={
            "snapshot": config,
            "learning": Learning(horizon=2, epochs=1, minibatches=1, ema=0.9),
        }
    )
    counts = Counts()

    def score(signals: EntrySignals) -> float:
        if counts.updates == 0:
            return float(signals.root_signals.policy_entropy.mean())
        if failure == "exception":
            raise ValueError("scorer failed")
        return float("nan")

    bank = SnapshotBank(config, scorer=score)
    learner, ema = create(settings, seed=0)
    assert ema is not None
    rng = np.random.default_rng(0)
    collector = self_play_collector(
        settings,
        seed=0,
        start=0,
        snapshot_bank=bank,
    )
    try:
        step(settings, learner, ema, collector, counts, rng, out=tmp_path)
        before = bank.state().model_dump_json()
        params = deepcopy(learner.agent.state_dict())
        averaged = deepcopy(ema.state_dict())
        adam = deepcopy(learner.optimizer.state_dict())
        assert adam["state"] and counts.updates == bank.generation == 1
        with pytest.raises(ValueError, match="scorer failed|finite and nonnegative"):
            step(settings, learner, ema, collector, counts, rng, out=tmp_path)
        torch.testing.assert_close(learner.agent.state_dict(), params, atol=0, rtol=0)
        torch.testing.assert_close(learner.optimizer.state_dict(), adam, atol=0, rtol=0)
        torch.testing.assert_close(ema.state_dict(), averaged, atol=0, rtol=0)
        assert counts.updates == 1 and bank.state().model_dump_json() == before
        assert counts.snapshot_bank == bank.state()
    finally:
        close_collector(collector)


@pytest.mark.parametrize("signal", ["action_value_spread", "wdl_entropy"])
def test_self_play_rejects_unavailable_signals_without_custom_scorer(
    signal: Literal["action_value_spread", "wdl_entropy"],
) -> None:
    settings = gray_ogre_self_play(horizon=2).model_copy(
        update={
            "snapshot": SnapshotConfig(
                capacity=1, replay_fraction=0.5, fresh_floor=0.5, signal=signal
            )
        }
    )
    with pytest.raises(ValueError, match=signal):
        self_play_collector(settings, start=0, seed=0)
    collector = self_play_collector(
        settings,
        start=0,
        seed=0,
        snapshot_scorer=lambda signals: 1.0,
    )
    close_collector(collector)


@pytest.mark.parametrize("use_factory", [True, False])
def test_deprecated_snapshot_factory_uses_native_deferred_slots(
    use_factory: bool,
) -> None:
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        vector = (
            InMemorySnapshotVector.factory(ObservationSpace(), Match(), 1, 42)
            if use_factory
            else InMemorySnapshotVector(ObservationSpace(), 1, 42)
        )
    try:
        assert vector.next_fresh_seeds() == [42]
    finally:
        vector.close()


def test_snapshot_callbacks_require_settings() -> None:
    settings = gray_ogre_self_play(horizon=2)
    with pytest.raises(ValueError, match="require snapshot settings"):
        self_play_collector(
            settings, start=0, seed=0, snapshot_scorer=lambda signals: 1.0
        )
    with pytest.raises(ValueError, match="require snapshot settings"):
        self_play_collector(settings, start=0, seed=0, capture_predicate=lambda d: True)


def test_snapshot_settings_require_both_seats() -> None:
    settings = gray_ogre_self_play(horizon=2).model_dump() | {
        "snapshot": SnapshotConfig(capacity=1, replay_fraction=0.5, fresh_floor=0.5)
    }
    for changes in (
        {"learning_seats": "one"},
        {"behavior": "random"},
        {"batch_closure": "exact-quota"},
    ):
        with pytest.raises(ValueError, match="both learning seats"):
            SelfPlay.model_validate(settings | changes)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_inspector_labels_a_cut_seat_trajectory_as_censored(
    capsys: pytest.CaptureFixture[str], kind: Literal["scalar", "categorical_wdl"]
) -> None:
    agent = small_agent(kind=kind)
    batch, raw = three_choices(agent)
    _report(
        batch,
        Learner(agent, "cpu", minibatch_rows=3),
        DampedPolicyLearning(gradient="damped_policy"),
        rows=2,
    )
    report = capsys.readouterr().out
    assert "2 seat trajectories" in report
    assert "first trajectory, which was censored" in report
    expected = (
        backend.rl.expected_outcome(raw.softmax(-1))
        if kind == "categorical_wdl"
        else raw
    )
    assert float(report.splitlines()[-1].split()[4]) == pytest.approx(
        expected[0].item(), abs=0.0005
    )


def test_deferred_collector_cuts_beside_terminal_failed_and_parked_slots() -> None:
    space = ObservationSpace()
    agent = small_agent(space)
    _, source = replay_root(space)
    root = source.snapshot()
    chooser = root.acting_player
    assert chooser in (0, 1)
    recipe = SnapshotRecipe.from_replay_json(root.to_replay_json(), chooser)
    for _ in range(64):
        last = source.snapshot()
        if source.step(0)[2]:
            break
    else:
        pytest.fail("fixture did not reach a terminal decision")
    chooser = last.metadata().acting_player
    assert chooser in (0, 1)
    final_recipe = SnapshotRecipe.from_replay_json(last.to_replay_json(), chooser)
    early_branch = SnapshotBranch(
        1,
        recipe,
        horizon=2,
        restore_options=managym.RestoreOptions.shuffle_libraries(101),
    )
    end_branch = SnapshotBranch(2, final_recipe)
    del root, last, source
    attempts: list[GameAttempt] = []
    policy = GameErrorPolicy()

    def record(state: GameErrorState, attempt: GameAttempt) -> GameErrorState:
        attempts.append(attempt)
        return settle(state, attempt, policy)

    collector = open_collector(
        space,
        Match(recipe.match),
        Reward(RewardConfig()),
        num_envs=4,
        seed=70,
        opponent_mode="self",
        games_per_stream=1,
        game_errors=policy,
        record_attempt=record,
        snapshots=True,
        workers=2,
        numerics=CollectionSession(device="cpu", seed=0),
    )
    vector = collector._env
    step = vector.step_isolated_into_buffers
    calls = 0

    def fail_once(
        actions: list[int], active: list[bool], recover_capacity: bool
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            actions[2] = 999
        step(actions, active, recover_capacity)

    with (
        patch.object(vector, "install_slots", wraps=vector.install_slots) as installs,
        patch.object(vector, "step_isolated_into_buffers", side_effect=fail_once),
        patch.object(
            vector,
            "observations_for_players",
            wraps=vector.observations_for_players,
        ) as reads,
        patch.object(
            torch,
            "multinomial",
            side_effect=lambda probs, *args, **kwargs: torch.zeros(
                len(probs), 1, dtype=torch.long
            ),
        ),
    ):
        try:
            batch = collect(
                collector,
                agent,
                1,
                snapshot_starts=[
                    early_branch,
                    end_branch,
                    early_branch,
                    replace(
                        end_branch,
                        restore_options=managym.RestoreOptions.exact(
                            forced_first_action=0
                        ),
                    ),
                    None,
                ],
            )
            assert collector.stats.snapshot.terminals == 2
            assert collector.stats.snapshot.failed_branches == 1
            assert collector.stats.snapshot.horizon_cuts == 1
            assert collector.stats.games == 1
            assert batch.is_root_decision.sum() == 2  # cut and sampled terminal
            assert all(
                call.args[0][1:] == [None, None, None] for call in reads.call_args_list
            )
            assert reads.called
            assert [a.valid_samples for a in attempts if a.stream == 3] == [0]
            [failure] = [a for a in attempts if a.outcome == "engine_error"]
            assert (
                failure.actions[-1] == 999
                and failure.interventions[-1].op == "shuffle_libraries"
            )
            assert vector.next_fresh_seeds() == [70, 71, 76, 73]
            assert [
                (r.slot, r.recipe) for r in installs.call_args_list[-1].args[0]
            ] == [(2, None)]
            # The next update starts with one mixed allocation. Ended fresh
            # slot 2 spends its next seed; the opening deals parked by branches resume.
            collect(collector, agent, 1, snapshot_starts=[early_branch])
            mixed = installs.call_args_list[-1].args[0]
            assert [(r.slot, r.recipe is None) for r in mixed] == [
                (0, False),
                (1, True),
                (2, True),
                (3, True),
            ]
            assert vector.current_game_seeds() == [42, 71, 76, 73]
        finally:
            close_collector(collector)
    result = replay_attempt(GameAttempt.model_validate_json(failure.model_dump_json()))
    assert (
        result["at_retained_failure"]
        and result["matches_binding"]
        and result["matches_error"]
    )


def test_grouped_plan_rejects_forced_opening_before_install() -> None:
    space = ObservationSpace()
    recipe, _ = replay_root(space)
    bank = SnapshotBank(SnapshotConfig(capacity=1, replay_fraction=1, fresh_floor=0))
    with replay_collector(space, recipe) as collector, bank.begin_batch() as plan:
        generations = collector._env.current_game_generations()
        plan.pending.append(
            SnapshotBranch(
                0,
                recipe,
                restore_options=managym.RestoreOptions.exact(forced_first_action=0),
            )
        )
        with pytest.raises(ValueError, match="cannot force an opening"):
            collect(collector, small_agent(space), 1, snapshot_plan=plan)
        assert collector._env.current_game_generations() == generations
        assert collector.stats.micro_steps == 0
