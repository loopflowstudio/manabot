"""MTG-252 readout arms vary only the critic readout of the re-pinned 183 recipe."""

import pytest
import torch

from experiments import value_readout_252
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.recipes.builders import with_agent
from manabot.training.rollout.collection import SeatRoutedCollector


def test_arms_change_only_the_value_readout() -> None:
    baseline = value_readout_252.base()
    for aggregation in value_readout_252.ARMS:
        regime = value_readout_252.arm(aggregation)
        assert regime.agent.value_aggregation == aggregation
        assert regime.stages == baseline.stages
        assert regime.observation == baseline.observation
    wider = with_agent(
        baseline,
        id="wider",
        agent=baseline.agent.model_copy(
            update={"value_aggregation": "value_token", "hidden_dim": 128}
        ),
    )
    with pytest.raises(ValueError, match="more than agent.value_aggregation"):
        value_readout_252.require_single_axis(baseline, wider)


def test_static_costs_and_initialization_sharing() -> None:
    costs = {cost.aggregation: cost for cost in value_readout_252.costs()}
    masked, token, readout = (costs[arm] for arm in value_readout_252.ARMS)
    # 183: two player objects, 60 cards and 40 permanents per seat.
    assert masked.objects == 2 + 2 * 60 + 2 * 40
    assert masked.added_parameters == 0 and masked.shares_initialization
    assert token.added_parameters == 64
    assert token.attention_pairs == 2 * (masked.objects + 1) ** 2
    # Query 64 + in-projection 3*64*64+3*64 + out-projection 64*64+64.
    assert readout.added_parameters == 16_704
    assert readout.attention_pairs == masked.attention_pairs + masked.objects
    assert readout.shares_initialization
    # The token's draw precedes later input encoders, shifting their initial values.
    assert not token.shares_initialization


def test_every_arm_runs_under_the_same_seeds() -> None:
    grid = value_readout_252.runs([25201, 25202])
    for seed in (25201, 25202):
        assert sorted(r.aggregation for r in grid if r.seed == seed) == sorted(
            value_readout_252.ARMS
        )
    with pytest.raises(ValueError, match="distinct seeds"):
        value_readout_252.runs([25201, 25201])


def test_pairings_depend_on_the_run_seed_not_the_arm() -> None:
    """Different readouts play different games yet meet the same pairing per game.

    Records the native pair of each stream's `k`th game root under two arms'
    policies; every (stream, k) seen by both must hold the same pair. Under
    complete-games closure each collect here finishes one game per stream, so
    the arms also stay aligned game for game.
    """
    torch.set_num_threads(1)
    space = ObservationSpace()
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    decisions: list[int] = []
    for torch_seed, aggregation in ((1, "masked_mean"), (2, "attention_readout")):
        torch.manual_seed(torch_seed)
        agent = ActorCritic(
            space,
            ModelConfig(
                hidden_dim=8,
                num_attention_heads=2,
                world="elemental-world-v2",
                value_aggregation=aggregation,
            ),
        )
        collector = SeatRoutedCollector(
            space,
            Match(
                MatchConfig.authored(
                    "elemental-world-v2", "elemental_ug", "elemental_bg"
                )
            ),
            Reward(RewardConfig()),
            num_envs=4,
            seed=25201 + 10000,
            opponent_mode="self",
            learning_seats="both",
            matchup_curriculum="elemental-uniform",
        )
        for _ in range(4):
            games = collector.stats.games_by_stream
            for stream, pair in enumerate(collector._env.current_matchup_indices()):
                key = (stream, games.get(stream, 0))
                assert seen.setdefault(key, pair) == pair
            collector.collect(agent, 64)
        decisions.append(collector.stats.micro_steps)
    # The policies played different games over the same pairings.
    assert decisions[0] != decisions[1]
