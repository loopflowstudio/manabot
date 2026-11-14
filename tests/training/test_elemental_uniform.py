"""Native game-root uniform sampling and real CPU learning."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.inference import PolicyInference
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.training.rollout import collection
from manabot.training.rollout.collection import (
    Collector,
    close_collector,
    collect,
    open_collector,
)
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorPolicy,
    append_attempt,
    open_ledger,
    replay_attempt,
)
import managym

PACK = "elemental-world-v2"


def _match() -> Match:
    return Match(MatchConfig.authored(PACK, "elemental_ug", "elemental_bg"))


def _collector(
    streams: int = 4,
    learning_seats: Literal["one", "both"] = "one",
) -> Collector:
    return open_collector(
        ObservationSpace(),
        _match(),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=17402,
        opponent_mode="self",
        learning_seats=learning_seats,
        batch_closure="exact-quota" if learning_seats == "one" else "complete-games",
        matchup_curriculum="elemental-uniform",
    )


@pytest.mark.parametrize("learning_seats", ["one", "both"])
def test_uniform_collector_changes_terminal_roots(
    learning_seats: Literal["one", "both"],
) -> None:
    torch.set_num_threads(1)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=PACK),
    )
    first = _collector(learning_seats=learning_seats)
    initial_pairs = first._env.current_matchup_indices()
    batch = collect(first, agent, 128)
    assert first.stats.games > 0
    assert first._env.current_matchup_indices() != initial_pairs
    for match, players in zip(
        first.stream_matches, first._env.current_player_configs(), strict=True
    ):
        assert (match.hero_deck, match.villain_deck) == (
            players[0].decklist,
            players[1].decklist,
        )
    exposure = first.stats.exposure
    assert (
        sum(c["learner_transitions"] for c in exposure.values()) == batch.actions.size
    )
    if learning_seats == "one":
        assert batch.actions.size == 4 * 128
    else:
        assert batch.actions.size == first.stats.micro_steps
        assert int(batch.dones.sum()) == first.stats.games * 2
    assert sum(c["terminal_games"] for c in exposure.values()) == first.stats.games
    assert sum(c["game_roots"] for c in exposure.values()) == first.stats.games + 4
    assert (
        sum(c["environment_decisions"] for c in exposure.values())
        == first.stats.micro_steps
    )
    assert (
        sum(c.wins + c.losses + c.draws for c in first.stats.self_play.counts.values())
        == first.stats.games * 2
    )
    close_collector(first)


@pytest.mark.parametrize("workers", [2, 4])
def test_uniform_draws_agree_across_producer_counts(workers: int) -> None:
    torch.set_num_threads(1)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=PACK),
    )
    collectors = [
        open_collector(
            ObservationSpace(),
            _match(),
            Reward(RewardConfig()),
            num_envs=4,
            seed=17402,
            opponent_mode="self",
            learning_seats="both",
            matchup_curriculum="elemental-uniform",
            workers=count,
        )
        for count in (1, workers)
    ]
    one, many = collectors
    try:
        for _ in range(2):
            left, right = collect(one, agent, 64), collect(many, agent, 64)
            expected, actual = left.ordered_obs(), right.ordered_obs()
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])
            np.testing.assert_array_equal(left.actions, right.actions)
            assert one._env.current_matchup_indices() == (
                many._env.current_matchup_indices()
            )
            assert one._env.current_game_seeds() == many._env.current_game_seeds()
            assert one.stats.exposure == many.stats.exposure
        assert one.stats.games > 0
    finally:
        close_collector(one)
        close_collector(many)


def test_uniform_256_stream_roots_and_full_roster_admission() -> None:
    collector = _collector(256)
    np.testing.assert_array_equal(collector.learner_seat, np.arange(256) % 2)
    assert sum(c["game_roots"] for c in collector.stats.exposure.values()) == 256
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=PACK),
    )
    agent.world_binding = checkpoint_world(_match().to_rust(), ObservationSpace())
    assert len(agent.world_binding["setups"]) == 10
    roster = elemental_roster(_match())
    for a in roster:
        for b in roster:
            validate_agent_setup(agent, [a, b])
    with pytest.raises(ValueError, match="exact Elemental"):
        elemental_roster(Match())
    bad = _match().config.model_copy(deep=True)
    bad.hero_sideboard = {}
    with pytest.raises(ValueError, match="exact Elemental"):
        elemental_roster(Match(bad))
    for mode, root in [("random", None), ("self", "lethal-target-v1")]:
        with pytest.raises(ValueError, match="current-self and ordinary roots"):
            open_collector(
                ObservationSpace(),
                _match(),
                Reward(RewardConfig()),
                num_envs=4,
                matchup_curriculum="elemental-uniform",
                opponent_mode=mode,
                root=root,
            )
    native = managym.VectorEnv(num_envs=1, seed=174, player_policies=("agent", "agent"))
    with pytest.raises(RuntimeError, match="every exact Elemental"):
        native.set_uniform_roster(roster[:-1])
    with pytest.raises(RuntimeError, match="every exact Elemental"):
        native.set_uniform_roster([roster[0]] * 10)


def test_uniform_crash_refill_retains_sampled_setups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    policy = GameErrorPolicy()
    ledger, _ = open_ledger(tmp_path / "attempts.jsonl", policy)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, world=PACK),
    )
    collector = open_collector(
        ObservationSpace(),
        _match(),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=1,
        matchup_curriculum="elemental-uniform",
        game_errors=policy,
        record_attempt=lambda state, row: append_attempt(ledger, state, row, {}),
    )
    original = collection._sample_learning_rows
    calls = 0

    def sample(
        collector: Collector,
        agent: ActorCritic,
        rows: np.ndarray,
        acting: np.ndarray,
        inference: PolicyInference,
    ) -> np.ndarray:
        nonlocal calls
        actions = original(collector, agent, rows, acting, inference)
        calls += 1
        if calls == 4:
            actions[0] = -1
        return actions

    monkeypatch.setattr(collection, "_sample_learning_rows", sample)
    try:
        batch = collect(collector, agent, 1)
        attempts = [
            GameAttempt.model_validate(json.loads(line)["attempt"])
            for line in ledger.path.read_text().splitlines()
        ]
        errors = collector.game_error_state
        assert errors.crashed == 1 and errors.completed == 2
        failed = next(row for row in attempts if row.outcome == "engine_error")
        assert len(failed.actions) == 4
        replay = replay_attempt(failed)
        assert replay["status"] == "failed" and replay["at_retained_failure"]
        assert replay["frame"] == failed.frame_json
        # Independently reset native roots at every retained seed: receipts must
        # bind the sampled pair, including the replacement, rather than the recipe pair.
        for attempt in attempts:
            probe = managym.VectorEnv(num_envs=1, seed=attempt.seed)
            roster = elemental_roster(_match())
            probe.set_uniform_roster(roster)
            probe.reset_all(_match().to_rust())
            a, b = probe.current_matchup_indices()[0]
            assert attempt.matchup.hero_deck == roster[a].decklist
            assert attempt.matchup.villain_deck == roster[b].decklist
            assert attempt.matchup.hero_sideboard == roster[a].sideboard
            assert attempt.matchup.villain_sideboard == roster[b].sideboard
        replacement = next(
            row for row in attempts if row.stream == 0 and row.outcome == "completed"
        )
        assert replacement.seed == failed.seed + 2
        assert sum(c["game_roots"] for c in collector.stats.exposure.values()) == 5
        assert (
            sum(c["learner_transitions"] for c in collector.stats.exposure.values())
            == batch.actions.size
        )
        assert sum(c["terminal_games"] for c in collector.stats.exposure.values()) == 2
        assert collector.game_error_state.discarded_samples == 4
    finally:
        close_collector(collector)
