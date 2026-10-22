"""Hand-calculated credit through production routing, packing and move losses.

Only the game sequence and policy are controlled. Collection, likelihood
recording, target construction, selection and optimizer loss are production code.
Each marker is a distinct acting-viewer observation, never a hidden-state input.
"""

from collections.abc import Iterator
from typing import Literal

import numpy as np
import pytest
import torch
from torch import Tensor

from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.config import ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.training.collection import RolloutBatch, SeatRoutedCollector
from manabot.training.damped_policy import _targets
from manabot.training.learner import Learner
from manabot.training.models import DampedPolicyLearning
from manabot.training.objectives import update_iteration

Kind = Literal["scalar", "categorical_wdl"]
Seats = Literal["one", "both"]

# Predictions are intentionally asymmetric: opposing players need not predict
# complementary outcomes. Credit must use the acting player's own prediction.
DISTRIBUTIONS = {
    0: [0.2, 0.4, 0.4],
    1: [0.5, 0.2, 0.3],
    2: [0.6, 0.2, 0.2],
    3: [0.1, 0.4, 0.5],
    4: [0.1, 0.2, 0.7],
    10: [0.2, 0.5, 0.3],
    11: [0.7, 0.2, 0.1],
    12: [0.2, 0.3, 0.5],
}
# lambda_pi=.5. E.g. A3=-1-.4=-1.4; A1=.4+.2-.7=-.1;
# A0=-.2-.2-.05=-.45. Opponent 2->4 instead has A2=.6+.4+.2=1.2.
ADVANTAGES = {0: -0.45, 1: -0.1, 3: -1.4, 2: 1.2, 4: 0.4, 11: -0.4, 10: 0.55, 12: 0.7}
# lambda_v=.8. These are expanded outcome mixtures, independently tabulated.
OUTCOMES = {
    0: [0.756, 0.104, 0.14],
    1: [0.82, 0.08, 0.1],
    3: [1, 0, 0],
    2: [0.02, 0.04, 0.94],
    4: [0, 0, 1],
    11: [1, 0, 0],
    10: [0.04, 0.06, 0.9],
    12: [0, 0, 1],
}
RETURNS = {0: -0.616, 1: -0.72, 3: -1, 2: 0.92, 4: 1, 11: -1, 10: 0.86, 12: 1}


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TableAgent(ActorCritic):
    """Independent logits per visible marker make per-row gradients inspectable."""

    def __init__(self, kind: Kind, *, opposite: bool = False) -> None:
        super().__init__(
            ObservationSpace(),
            ModelConfig(attention_on=False, hidden_dim=4, value_kind=kind),
        )
        p = torch.tensor(
            [DISTRIBUTIONS.get(i % 20, [0.3, 0.3, 0.4]) for i in range(80)]
        )
        if opposite:
            p = p.flip(-1)
        self.predictions = torch.nn.Parameter(
            p.log() if kind == "categorical_wdl" else (p[:, 2] - p[:, 0])[:, None]
        )
        # A large padded logit must get exactly zero probability and gradient.
        logits = torch.full(
            (80, self.observation_space.shapes["actions_valid"][0]), 900.0
        )
        logits[:, :2] = torch.tensor([0.25, 0.75]).log()
        self.offers = torch.nn.Parameter(logits)

    def forward_distribution(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        index = obs["agent_player"][:, 0, 0].long()
        return self.offers[index].masked_fill(
            obs["actions_valid"] == 0, -torch.inf
        ), self.predictions[index]


class CreditGames:
    """Two unequal games; native seat swaps leave the visible sequence fixed.

    Stream 0: A,A,B,A,B; stream 1: B,A,B. B wins. Reset can
    start a draw game. Buffers deliberately contain unrelated native rewards.
    """

    def __init__(
        self,
        buffers: dict[str, np.ndarray],
        *,
        swapped: bool = False,
        draw: bool = False,
        truncated: bool = False,
    ) -> None:
        self.buffers = buffers
        self.sequences = [(0, 0, 1, 0, 1), (1, 0, 1)]
        self.positions = [0, 0]
        self.episodes = [0, 0]
        self.swapped = int(swapped)
        self.draw = draw
        self.truncated = truncated
        self._encode()

    def _encode(self) -> None:
        self.buffers["actions_valid"].fill(0)
        self.buffers["actions_valid"][:, :2] = 1
        for row, position in enumerate(self.positions):
            self.buffers["agent_player"][row, 0, 0] = (
                20 * self.episodes[row] + 10 * row + position
            )
            if position == 1:
                self.buffers["actions_valid"][row, 1] = 0  # forced offer

    def current_agent_indices(self) -> list[int]:
        return [
            self.sequences[r][p] ^ self.swapped for r, p in enumerate(self.positions)
        ]

    def step_into_buffers(self, actions: list[int], active: list[bool]) -> None:
        self.buffers["terminated"].fill(0)
        self.buffers["truncated"].fill(0)
        self.buffers["rewards"].fill(37)  # never a source of learner credit
        for row, enabled in enumerate(active):
            if not enabled:
                continue
            assert self.buffers["actions_valid"][row, actions[row]]
            self.positions[row] += 1
            if self.positions[row] == len(self.sequences[row]):
                self.positions[row] = 0
                self.episodes[row] += 1
                self.buffers["truncated" if self.truncated else "terminated"][row] = 1
        self._encode()

    def get_last_info(self) -> list[dict[str, int]]:
        return [
            {} if self.draw or e > 1 else {"winner_index": 1 ^ self.swapped}
            for e in self.episodes
        ]


def collector_for(
    agent: TableAgent,
    seats: Seats,
    *,
    games: int | None = 1,
    swapped: bool = False,
    second: TableAgent | None = None,
    draw: bool = False,
    truncated: bool = False,
) -> SeatRoutedCollector:
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self" if second is None else "learning",
        opponent_agent=second,
        learning_seats=seats,
        games_per_stream=games,
    )
    collector._env = CreditGames(
        collector._buffers, swapped=swapped, draw=draw, truncated=truncated
    )
    return collector


def markers(batch: RolloutBatch) -> list[int]:
    return batch.obs["agent_player"][..., 0, 0].astype(int).ravel().tolist()


def learning(**changes: object) -> DampedPolicyLearning:
    return DampedPolicyLearning(
        gradient="damped_policy",
        policy_lambda=0.5,
        value_lambda=0.8,
        advantage_quantile=0,
        min_advantage=0,
        collection_kl=0,
        tau_scale=0,
    ).model_copy(update=changes)


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
@pytest.mark.parametrize("seats", ["one", "both"])
@pytest.mark.parametrize("swapped", [False, True])
def test_complete_game_targets_and_actual_loss_gradients(
    kind: Kind,
    seats: Seats,
    swapped: bool,
) -> None:
    agent = TableAgent(kind)
    collector = collector_for(agent, seats, swapped=swapped)
    batch = collector.collect(agent, 1)
    ids = markers(batch)
    expected_ids = (
        [0, 1, 3, 2, 4, 11, 10, 12] if not swapped else [2, 4, 0, 1, 3, 10, 12, 11]
    )
    if seats == "one":
        expected_ids = [0, 1, 3, 10, 12] if not swapped else [2, 4, 11]
    assert ids == expected_ids
    assert batch.actions.size == len(expected_ids)  # no trajectory padding
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    rule = learning()
    targets = _targets(trainer, batch, rule)
    torch.testing.assert_close(
        targets.advantages[:, 0], torch.tensor([ADVANTAGES[i] for i in ids])
    )
    expected = torch.tensor(
        [OUTCOMES[i] if kind == "categorical_wdl" else RETURNS[i] for i in ids],
        dtype=torch.float32,
    )
    torch.testing.assert_close(targets.values[:, 0], expected)
    assert not targets.advantages.requires_grad and not targets.values.requires_grad
    np.testing.assert_allclose(
        batch.values.ravel(),
        [DISTRIBUTIONS[i][2] - DISTRIBUTIONS[i][0] for i in ids],
        atol=1e-7,
    )
    selected_logs = np.take_along_axis(
        batch.log_probabilities, batch.actions[..., None], -1
    )
    np.testing.assert_array_equal(batch.logprobs, selected_logs[..., 0])

    # Hooks capture gradients before clipping/Adam. Rows have independent
    # parameters, so prior minibatches cannot change later predictions.
    policy_grad = torch.zeros_like(agent.offers)
    value_grad = torch.zeros_like(agent.predictions)

    def capture_policy(gradient: Tensor) -> None:
        policy_grad.add_(gradient)

    def capture_value(gradient: Tensor) -> None:
        value_grad.add_(gradient)

    hooks = [
        agent.offers.register_hook(capture_policy),
        agent.predictions.register_hook(capture_value),
    ]
    result = update_iteration(trainer, batch, rule, 0, np.random.default_rng(0))
    assert result["actor_exposures"] == result["critic_exposures"] == len(ids)
    for row, marker in enumerate(ids):
        n = min(2, len(ids) - (row // 2) * 2)
        p = torch.tensor([0.25, 0.75]) if marker % 10 != 1 else torch.tensor([1.0, 0.0])
        expected_pg = p.clone()
        expected_pg[int(batch.actions[row, 0])] -= 1
        expected_pg *= ADVANTAGES[marker] / n  # d(-A log pi)/dz
        torch.testing.assert_close(policy_grad[marker, :2], expected_pg)
        if kind == "categorical_wdl":
            expected_vg = (
                torch.tensor(DISTRIBUTIONS[marker]) - torch.tensor(OUTCOMES[marker])
            ) / n
        else:
            v = DISTRIBUTIONS[marker][2] - DISTRIBUTIONS[marker][0]
            expected_vg = torch.tensor([2 * (v - RETURNS[marker]) / n])
        torch.testing.assert_close(value_grad[marker], expected_vg)
    assert torch.count_nonzero(policy_grad[:, 2:]) == 0
    unused = sorted(set(range(80)) - set(ids))
    assert (
        torch.count_nonzero(policy_grad[unused])
        == torch.count_nonzero(value_grad[unused])
        == 0
    )
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_exact_quota_uses_same_player_paused_value_and_resamples(
    kind: Kind,
) -> None:
    agent = TableAgent(kind)
    collector = collector_for(agent, "one", games=None)
    batch = collector.collect(agent, 2)
    # Stream 0 has taken 0,1 and pauses before 3 (after opponent 2).
    # Stream 1 has taken 10,12 and terminated. Its reset value is irrelevant.
    assert markers(batch) == [0, 10, 1, 12]
    np.testing.assert_array_equal(batch.dones, [[False, False], [False, True]])
    np.testing.assert_array_equal(batch.next_obs["agent_player"][:, 0, 0], [3, 30])
    assert not any(collector._pending) and not any(collector._streams)
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    targets = _targets(trainer, batch, learning())
    torch.testing.assert_close(
        targets.advantages, torch.tensor([[-0.1, 0.55], [0.6, 0.7]])
    )
    expected = (
        [[[0.18, 0.36, 0.46], [0.04, 0.06, 0.9]], [[0.1, 0.4, 0.5], [0, 0, 1]]]
        if kind == "categorical_wdl"
        else [[0.28, 0.86], [0.4, 1.0]]
    )
    torch.testing.assert_close(targets.values, torch.tensor(expected))
    with torch.no_grad():
        agent.offers[:, :2] = torch.tensor([0.8, 0.2]).log()
        agent.predictions[3] = (
            torch.tensor([0.6, 0.3, 0.1]).log() if kind == "categorical_wdl" else -0.5
        )
    second = collector.collect(agent, 1)
    assert markers(second)[0] == 3
    assert second.values[0, 0] == pytest.approx(-0.5)
    np.testing.assert_allclose(second.probabilities[0, 0, :2], [0.8, 0.2])


@pytest.mark.parametrize("seats", ["one", "both"])
def test_draw_reset_has_no_previous_game_credit(seats: Seats) -> None:
    agent = TableAgent("categorical_wdl")
    collector = collector_for(agent, seats, games=2)
    batch = collector.collect(agent, 1)
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    targets = _targets(trainer, batch, learning(policy_lambda=1.0, value_lambda=1.0))
    for row, marker in enumerate(markers(batch)):
        outcome = 0 if marker >= 20 else -1 if marker in (0, 1, 3, 11) else 1
        torch.testing.assert_close(targets.values[row, 0], torch.eye(3)[outcome + 1])
        assert targets.advantages[row, 0] == pytest.approx(
            outcome - batch.values[row, 0]
        )


@pytest.mark.parametrize("seats", ["one", "both"])
def test_truncation_never_returns_a_draw_training_batch(seats: Seats) -> None:
    agent = TableAgent("scalar")
    collector = collector_for(agent, seats, truncated=True)
    with pytest.raises(RuntimeError, match="truncated; no terminal target"):
        collector.collect(agent, 20)
    assert collector.stats.truncations == 1


@pytest.mark.parametrize("kind", ["scalar", "categorical_wdl"])
def test_distinct_policies_use_their_own_predictions_and_never_each_others_gradients(
    kind: Kind,
) -> None:
    first, second = TableAgent(kind), TableAgent(kind, opposite=True)
    collector = collector_for(first, "both", second=second)
    a, b = collector.collect_pair(first, 1)
    assert markers(a) == [0, 1, 3, 10, 12]
    assert markers(b) == [2, 4, 11]
    trainers = [
        Learner(first, "cpu", minibatch_rows=collector.num_envs),
        Learner(second, "cpu", minibatch_rows=collector.num_envs),
    ]
    # Second model predicts the opposite signed values: .4,-.6,.6.
    # Its B win gives A4=1.6 and A2=-.6-.4+.5*1.6=-.2;
    # its stream-1 A loss gives A11=-1.6.
    target = _targets(trainers[1], b, learning())
    torch.testing.assert_close(target.advantages[:, 0], torch.tensor([-0.2, 1.6, -1.6]))
    expected = (
        [[0.14, 0.04, 0.82], [0, 0, 1], [1, 0, 0]]
        if kind == "categorical_wdl"
        else [0.68, 1, -1]
    )
    torch.testing.assert_close(
        target.values[:, 0], torch.tensor(expected, dtype=torch.float32)
    )
    before_second = second.predictions.detach().clone()
    update_iteration(trainers[0], a, learning(), 0, np.random.default_rng(0))
    assert second.predictions.grad is None
    torch.testing.assert_close(second.predictions, before_second)
    before_first = first.predictions.detach().clone()
    update_iteration(trainers[1], b, learning(), 0, np.random.default_rng(0))
    torch.testing.assert_close(first.predictions, before_first)
    assert not torch.equal(second.predictions, before_second)


@pytest.mark.parametrize("scope", ["actor", "actor_critic"])
def test_filter_retains_negative_credit_and_critic_coverage(
    scope: Literal["actor", "actor_critic"],
) -> None:
    agent = TableAgent("scalar")
    collector = collector_for(agent, "both")
    batch = collector.collect(agent, 1)
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    rule = learning(min_advantage=0.6, filter_scope=scope)
    targets = _targets(trainer, batch, rule)
    # Magnitude selection includes the losing player's -1.4, not just winners.
    np.testing.assert_array_equal(targets.selected[:, 0], [0, 0, 1, 1, 0, 0, 0, 1])
    old_policy, old_values = (
        agent.offers.detach().clone(),
        agent.predictions.detach().clone(),
    )
    result = update_iteration(trainer, batch, rule, 0, np.random.default_rng(0))
    assert result["actor_exposures"] == 3
    assert result["critic_exposures"] == (8 if scope == "actor" else 3)
    for marker in markers(batch):
        selected = marker in (3, 2, 12)
        assert bool((old_policy[marker] != agent.offers[marker]).any()) == selected
        assert bool((old_values[marker] != agent.predictions[marker]).any()) == (
            selected or scope == "actor"
        )


@pytest.mark.parametrize("mode", ["random", "frozen"])
def test_off_policy_actions_never_enter_batch_or_receive_gradients(
    mode: str,
) -> None:
    agent, opponent = TableAgent("scalar"), TableAgent("scalar", opposite=True)
    opponent.requires_grad_(False)
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode=mode,
        opponent_agent=opponent,
        games_per_stream=1,
    )
    collector._env = CreditGames(collector._buffers)
    batch = collector.collect(agent, 1)
    assert markers(batch) == [0, 1, 3, 10, 12]
    trainer = Learner(agent, "cpu", minibatch_rows=collector.num_envs)
    before = opponent.offers.detach().clone()
    update_iteration(trainer, batch, learning(), 0, np.random.default_rng(0))
    assert opponent.offers.grad is None
    torch.testing.assert_close(opponent.offers, before)
    assert collector.stats.opponent_decisions == 3
