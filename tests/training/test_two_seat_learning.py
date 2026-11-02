"""Both-seat terminal credit, behavior ownership and independent optimizer updates."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    RewardConfig,
)
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.training.execution import execute_regime
from manabot.training.learning.learner import Learner, observation_tensors
from manabot.training.learning.objectives import update_iteration
from manabot.training.learning.returns import transition_gae
from manabot.training.regime import (
    DampedPolicyLearning,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.collection import RolloutBatch, SeatRoutedCollector
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_recovery import tracked_updates


class ScriptedGames:
    """Unequal decision counts and consecutive moves, followed by a seat-1 win.

    Encodings are legal native initial observations, with life used as a visible
    marker for verifying whose observation was retained. Stream 1 ends earlier,
    exercising pauses while the other stream finishes its game.
    """

    def __init__(self, buffers: dict[str, np.ndarray]) -> None:
        self.buffers = buffers
        self.positions = [0, 0]
        self.seats = [(0, 0, 1, 0, 1), (1, 0, 1)]
        self.steps = [0, 0]
        self._encode()

    def _encode(self) -> None:
        for row, position in enumerate(self.positions):
            self.buffers["agent_player"][row, 0, 0] = 10 * row + position

    def current_agent_indices(self) -> list[int]:
        return [self.seats[row][pos] for row, pos in enumerate(self.positions)]

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.buffers["terminated"].fill(0)
        self.buffers["truncated"].fill(0)
        for row, enabled in enumerate(active):
            if not enabled:
                continue
            self.steps[row] += 1
            self.positions[row] += 1
            if self.positions[row] == len(self.seats[row]):
                self.positions[row] = 0
                self.buffers["terminated"][row] = 1
        self._encode()

    def get_last_info(self) -> list[dict[str, int]]:
        return [{"winner_index": 1}, {"winner_index": 1}]


def _agent(kind: Literal["scalar", "categorical_wdl"] = "scalar") -> ActorCritic:
    torch.set_num_threads(1)
    return ActorCritic(
        ObservationSpace(),
        ModelConfig(attention_on=False, hidden_dim=8, value_kind=kind),
    )


def _collector(
    agent: ActorCritic,
    *,
    second: ActorCritic | None = None,
    seats: Literal["one", "both"] = "both",
    games: int | None = None,
    scripted: bool = True,
) -> SeatRoutedCollector:
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="learning" if second is not None else "self",
        opponent_agent=second,
        learning_seats=seats,
        games_per_stream=games,
        recovery_max_microsteps=10_000 if second is None else None,
    )
    if scripted:
        collector._env = ScriptedGames(collector._buffers)
    return collector


def _behavior_matches(batch: RolloutBatch, agent: ActorCritic) -> None:
    obs = observation_tensors(
        {
            key: value.reshape((-1,) + value.shape[2:])
            for key, value in batch.obs.items()
        },
        "cpu",
    )
    with torch.no_grad():
        logits, values = agent(obs)
    np.testing.assert_allclose(batch.values.ravel(), values.numpy().ravel(), atol=1e-6)
    np.testing.assert_allclose(
        batch.probabilities.reshape(logits.shape), logits.softmax(-1).numpy(), atol=1e-6
    )
    selected = np.take_along_axis(batch.log_probabilities, batch.actions[..., None], -1)
    np.testing.assert_allclose(batch.logprobs, selected[..., 0])


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_both_seats_keep_their_own_trajectory_and_terminal_credit(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = _agent(kind)
    collector = _collector(agent)
    batch = collector.collect(agent, 1)
    assert batch.actions.shape == (8, 1)
    np.testing.assert_array_equal(
        batch.obs["agent_player"][:, 0, 0, 0], [0, 1, 3, 2, 4, 11, 10, 12]
    )
    np.testing.assert_array_equal(batch.rewards[:, 0], [0, 0, -1, 0, 1, -1, 0, 1])
    np.testing.assert_array_equal(batch.dones[:, 0], [0, 0, 1, 0, 1, 1, 0, 1])
    _, returns = transition_gae(
        torch.from_numpy(batch.rewards),
        torch.zeros_like(torch.from_numpy(batch.values)),
        torch.from_numpy(batch.dones),
        torch.tensor([100.0]),
        1.0,
        1.0,
    )
    np.testing.assert_array_equal(returns[:, 0], [-1, -1, -1, 1, 1, -1, 1, 1])
    _behavior_matches(batch, agent)
    if kind == "categorical_wdl":
        assert batch.outcome_probabilities is not None
        np.testing.assert_allclose(
            batch.values,
            batch.outcome_probabilities[..., 2] - batch.outcome_probabilities[..., 0],
            atol=1e-6,
        )
    assert collector.stats.learner_transitions == collector.stats.micro_steps == 8
    assert collector.stats.opponent_decisions == 0
    assert collector.stats.games == 2
    assert collector._env.steps == [5, 3]
    collector.snapshot()  # no pending or banked rows at the update boundary
    with torch.no_grad():
        agent.value_head[-1].bias.add_(4)
    second = collector.collect(agent, 7)
    _behavior_matches(second, agent)
    assert (
        second.actions.size == 19
    )  # 2 x five-decision games, 3 x three-decision games
    collector.snapshot()


def test_matched_games_control_keeps_only_designated_seats() -> None:
    agent = _agent()
    one = _collector(agent, seats="one", games=2)
    both = _collector(agent, games=2)
    a, b = one.collect(agent, 999), both.collect(agent, 999)
    assert one.stats.games == both.stats.games == 4
    assert one.stats.micro_steps == both.stats.micro_steps == 16
    assert a.actions.size == 10 and b.actions.size == 16
    assert one.stats.opponent_decisions == 6
    assert both.stats.opponent_decisions == 0


def test_pair_routes_own_behavior() -> None:
    first, second = _agent(), _agent()
    with torch.no_grad():
        first.value_head[-1].bias.fill_(0.3)
        second.value_head[-1].bias.fill_(-0.7)
    collector = _collector(first, second=second, games=1)
    a, b = collector.collect_pair(first, 1)
    _behavior_matches(a, first)
    _behavior_matches(b, second)
    np.testing.assert_array_equal(a.obs["agent_player"][:, 0, 0, 0], [0, 1, 3, 10, 12])
    np.testing.assert_array_equal(b.obs["agent_player"][:, 0, 0, 0], [2, 4, 11])


def test_two_seat_real_native_games_and_recovery() -> None:
    agent = _agent()
    collector = _collector(agent, games=1, scripted=False)
    batch = collector.collect(agent, 1)
    assert batch.actions.size == collector.stats.micro_steps
    assert collector.stats.games == 2
    assert int(batch.dones.sum()) == 4
    assert float(batch.rewards.sum()) == 0
    boundary = collector.snapshot()
    restored = _collector(agent, games=1, scripted=False)
    restored.restore(boundary, lambda: None)
    expected = collector.collect(agent, 1)
    actual = restored.collect(agent, 1)
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_array_equal(actual.rewards, expected.rewards)
    for key in actual.obs:
        np.testing.assert_array_equal(actual.obs[key], expected.obs[key])


@pytest.mark.parametrize("mode", ["random", "frozen"])
def test_off_policy_opponents_cannot_train_both_seats(mode: str) -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="both learning seats"):
        SeatRoutedCollector(
            agent.observation_space,
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode=mode,
            opponent_agent=agent,
            learning_seats="both",
        )


def test_self_play_defaults_to_both_and_rejects_off_policy_rows() -> None:
    stage = TrainSelfPlay(id="train", operation="train_self_play")
    assert stage.learning_seats == "both"
    assert stage.model_dump()["learning_seats"] == "both"
    assert (
        TrainSelfPlay(
            id="random", operation="train_self_play", behavior="random"
        ).learning_seats
        == "one"
    )
    assert "games_per_stream" not in stage.model_dump()
    with pytest.raises(ValueError, match="both learning seats"):
        TrainSelfPlay(
            id="train",
            operation="train_self_play",
            behavior="random",
            learning_seats="both",
        )


@pytest.mark.parametrize("behavior", ["current-self", "ema-self"])
def test_regime_trains_and_exports_both_seats(
    behavior: Literal["current-self", "ema-self"], tmp_path: Path
) -> None:
    agent = _agent()
    stage = TrainSelfPlay(
        id="learn",
        operation="train_self_play",
        updates=2,
        streams=2,
        transitions=1,
        learning_seats="both",
        games_per_stream=1,
        behavior=behavior,
        learning=Learning(epochs=1, minibatches=1, retained_fraction=1, ema=0.9),
    )
    regime = TrainingRegime(
        id="two-seat-smoke",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        agent=agent.config,
        stages=[stage],
        wall_seconds=120,
        schedule_clock="iteration_fraction",
    )
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(regime, 17, tmp_path / "run", store)
    assert run.status == "completed"
    record = run.stages[0]
    assert record.games == 4
    assert record.environment_decisions == record.learner_transitions
    rows = tracked_updates(run, tmp_path)
    assert sum(row["rows"] for row in rows) == record.learner_transitions
    assert record.optimizer_exposures == record.learner_transitions
    assert all(row["bootstrapped_tail_fraction"] == 0 for row in rows)
    for role in ("raw", "ema"):
        loaded, _ = load_checkpoint(record.artifacts[role]["path"])
        assert loaded.config == agent.config


def test_live_continuation_cannot_silently_change_seat_recipe() -> None:
    first = TrainSelfPlay(id="a", operation="train_self_play", learning_seats="one")
    second = TrainSelfPlay(
        id="b", operation="train_self_play", initial="a", learning_seats="both"
    )
    with pytest.raises(ValueError, match="continuation must preserve"):
        TrainingRegime(
            id="invalid",
            world=managym.WORLD_VERSION,
            match=MatchConfig(),
            stages=[first, second],
        )


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_move_learning_consumes_complete_seat_trajectories(
    kind: Literal["scalar", "categorical_wdl"],
) -> None:
    agent = _agent(kind)
    collector = _collector(agent)
    batch = collector.collect(agent, 1)
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    result = update_iteration(
        trainer,
        batch,
        DampedPolicyLearning(
            gradient="damped_policy", advantage_quantile=0, min_advantage=0
        ),
        0,
        np.random.default_rng(1),
        iteration=1,
    )
    assert result["rows"] == 8
    assert result["optimizer_exposures"] == result["retained"]
    assert result["bootstrapped_tail_fraction"] == 0
    assert trainer.optimizer.state
    # Adam advances at most once per two native streams, not once per packed row.
    steps = {int(state["step"]) for state in trainer.optimizer.state.values()}
    assert max(steps) <= 4


def test_collector_defaults_follow_policy_ownership() -> None:
    agent = _agent()
    for mode, seats in (
        ("self", "both"),
        ("learning", "both"),
        ("random", "one"),
        ("frozen", "one"),
    ):
        collector = SeatRoutedCollector(
            agent.observation_space,
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode=mode,
            opponent_agent=_agent(),
        )
        assert collector.learning_seats == seats
        collector.close()


def test_named_ladder_recipes_preserve_the_control_and_select_both_seats() -> None:
    from experiments.batched_baseline import scale_ladder, scale_ladder_both

    control, treatment = scale_ladder(), scale_ladder_both()
    first, second = control.stages[0], treatment.stages[0]
    assert isinstance(first, TrainSelfPlay) and isinstance(second, TrainSelfPlay)
    assert first.learning_seats == "one" and second.learning_seats == "both"
    assert first.learning == second.learning
    assert first.streams == second.streams == 256
    assert control.world == treatment.world == managym.WORLD_VERSION
    assert control.id != treatment.id
