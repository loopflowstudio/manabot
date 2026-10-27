"""Native game-root uniform sampling, exact recovery and real CPU learning."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from experiments.ataraxos_move import ataraxos_move_scalar
from manabot.env import Match, ObservationSpace, Reward
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.checkpoint import load_checkpoint
from manabot.model.inference import PolicyInference
from manabot.model.world import checkpoint_world, validate_agent_setup
from manabot.training.execution import execute_regime, validate_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.rollout.matchups import elemental_roster
from manabot.training.rollout.self_play import deck_identity
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    replay_attempt,
)
from manabot.verify.store import VerifyStore
import managym
from tests.training.test_recovery import tracked_updates

PACK = "elemental-world-v2"


def _match() -> Match:
    return Match(MatchConfig.authored(PACK, "elemental_ug", "elemental_bg"))


def _collector(
    recovery: str,
    streams: int = 4,
    learning_seats: Literal["one", "both"] = "one",
) -> SeatRoutedCollector:
    return SeatRoutedCollector(
        ObservationSpace(),
        _match(),
        Reward(RewardConfig()),
        num_envs=streams,
        seed=17402,
        opponent_mode="self",
        learning_seats=learning_seats,
        matchup_curriculum="elemental-uniform",
        recovery_max_microsteps=200000 if recovery == "exact" else None,
        recovery_max_game_microsteps=200000 if recovery == "current" else None,
    )


@pytest.mark.parametrize("recovery", ["exact", "current"])
@pytest.mark.parametrize("learning_seats", ["one", "both"])
def test_uniform_collector_recovers_changed_terminal_roots(
    recovery: str, learning_seats: Literal["one", "both"]
) -> None:
    torch.set_num_threads(1)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )
    first = _collector(recovery, learning_seats=learning_seats)
    initial_pairs = first._env.current_matchup_indices()
    batch = first.collect(agent, 128)
    assert first.stats.games > 0
    assert first._env.current_matchup_indices() != initial_pairs
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
    if recovery == "exact":
        # Independent action-journal audit: count decisions against the actual
        # native pair BEFORE auto-reset, not the pair in the returned observation.
        replay = _collector(recovery, learning_seats=learning_seats)
        expected: dict[str, dict[str, int]] = {}
        identities = [
            deck_identity(p.decklist, p.sideboard) for p in elemental_roster(_match())
        ]

        def key(row: int, pair: tuple[int, int]) -> str:
            seat = "both" if learning_seats == "both" else str(row % 2)
            return f"{identities[pair[0]]}/{identities[pair[1]]}/learner-{seat}"

        def count(identity: str, kind: str) -> None:
            counts = expected.setdefault(
                identity,
                dict(
                    game_roots=0,
                    environment_decisions=0,
                    learner_transitions=0,
                    terminal_games=0,
                    truncated_games=0,
                ),
            )
            counts[kind] += 1

        for row, pair in enumerate(replay._env.current_matchup_indices()):
            count(key(row, pair), "game_roots")
        for actions, active in first.snapshot().journal:
            pairs = replay._env.current_matchup_indices()
            acting = replay._env.current_agent_indices()
            replay._env.step_into_buffers(actions, active)
            for row, enabled in enumerate(active):
                if not enabled:
                    continue
                identity = key(row, pairs[row])
                count(identity, "environment_decisions")
                if learning_seats == "both" or acting[row] == row % 2:
                    count(identity, "learner_transitions")
                if replay._buffers["terminated"][row]:
                    count(identity, "terminal_games")
                    count(
                        key(row, replay._env.current_matchup_indices()[row]),
                        "game_roots",
                    )
        assert expected == first.stats.exposure
    restored = _collector(recovery, learning_seats=learning_seats)
    restored.restore(first.snapshot(), lambda: None)
    assert first._self_play_decks == restored._self_play_decks
    left, right = first.collect(agent, 64), restored.collect(agent, 64)
    for key in left.obs:
        np.testing.assert_array_equal(left.obs[key], right.obs[key])
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    assert first.stats.exposure == restored.stats.exposure
    assert first.stats.self_play == restored.stats.self_play
    assert first._env.current_game_seeds() == restored._env.current_game_seeds()


def test_uniform_256_stream_roots_and_full_roster_admission() -> None:
    collector = _collector("current", 256)
    np.testing.assert_array_equal(collector.learner_seat, np.arange(256) % 2)
    assert sum(c["game_roots"] for c in collector.stats.exposure.values()) == 256
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
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
    for workers, mode, root in [
        (2, "self", None),
        (1, "random", None),
        (1, "self", "lethal-target-v1"),
    ]:
        with pytest.raises(ValueError, match="one producer"):
            SeatRoutedCollector(
                ObservationSpace(),
                _match(),
                Reward(RewardConfig()),
                num_envs=4,
                matchup_curriculum="elemental-uniform",
                workers=workers,
                opponent_mode=mode,
                root=root,
            )
    native = managym.VectorEnv(num_envs=1, seed=174, opponent_policy="none")
    with pytest.raises(RuntimeError, match="every exact Elemental"):
        native.set_uniform_roster(roster[:-1])
    with pytest.raises(RuntimeError, match="every exact Elemental"):
        native.set_uniform_roster([roster[0]] * 10)


def test_uniform_real_cpu_update_and_checkpoint_reload(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    value = ataraxos_move_scalar().model_dump(mode="json")
    value.update(
        world=managym.WORLD_VERSION,
        id="uniform-cpu-proof",
        match=_match().config.model_dump(),
    )
    value["agent"].update(hidden_dim=8, semantic_pack=PACK, policy_film=False)
    value["stages"] = value["stages"][:1]
    stage = value["stages"][0]
    stage.update(
        streams=4,
        transitions=64,
        updates=2,
        matchup_curriculum="elemental-uniform",
        learning_seats="one",
    )
    stage["execution"]["wall_seconds"] = 180
    stage["learning"].update(min_advantage=0, advantage_quantile=0)
    bad_input = deepcopy(value)
    bad_input["agent"]["semantic_pack"] = "ur-lessons-vs-gw-allies"
    with pytest.raises(ValueError, match="complete Elemental policy input"):
        validate_regime(bad_input)
    recipe = validate_regime(value)
    stage_model = recipe.stages[0]
    assert isinstance(stage_model, TrainSelfPlay)
    baseline = deepcopy(value)
    baseline["stages"][0].update(streams=256, transitions=64)
    assert validate_regime(baseline).stages[0].streams == 256
    for invalid in (2, 4):
        bad = deepcopy(value)
        bad["stages"][0]["execution"]["workers"] = invalid
        with pytest.raises(ValueError, match="one producer"):
            TrainingRegime.model_validate(bad)
    with VerifyStore(tmp_path / "store.sqlite") as store:
        run = execute_regime(recipe, 17402, tmp_path / "run", store)
    assert run.status == "completed"
    result = run.stages[0]
    assert result.learner_transitions == 4 * 64 * 2
    assert result.optimizer_exposures > 0
    assert result.updates == 2
    last = tracked_updates(run, tmp_path)[-1]

    def exposure(count: str) -> int:
        prefix = "cumulative_matchup_exposure/"
        return sum(
            int(value)
            for name, value in last.items()
            if name.startswith(prefix) and name.endswith(f"/{count}")
        )

    assert exposure("learner_transitions") == 512
    assert exposure("terminal_games") == result.games
    raw, _ = load_checkpoint(result.artifacts["raw"]["path"])
    ema, _ = load_checkpoint(result.artifacts["ema"]["path"])
    initial, _ = load_checkpoint(result.artifacts["initial_raw"]["path"])
    assert any(
        not torch.equal(a, b)
        for a, b in zip(raw.parameters(), initial.parameters(), strict=True)
    )
    assert len(raw.world_binding["setups"]) == 10
    assert len(ema.world_binding["setups"]) == 10
    assert all(torch.isfinite(p).all() for p in raw.parameters())


def test_uniform_crash_refill_retains_sampled_setups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    policy = GameErrorPolicy()
    ledger = GameErrorLedger(tmp_path / "attempts.jsonl", policy)
    agent = ActorCritic(
        ObservationSpace(),
        ModelConfig(hidden_dim=8, num_attention_heads=2, semantic_pack=PACK),
    )
    collector = SeatRoutedCollector(
        ObservationSpace(),
        _match(),
        Reward(RewardConfig()),
        num_envs=2,
        seed=17402,
        opponent_mode="self",
        games_per_stream=1,
        matchup_curriculum="elemental-uniform",
        game_errors=policy,
        record_attempt=lambda row: ledger.record(row, {}),
    )
    original = collector._sample_learning_rows
    calls = 0

    def sample(
        agent: ActorCritic,
        rows: np.ndarray,
        acting: np.ndarray,
        inference: PolicyInference,
    ) -> np.ndarray:
        nonlocal calls
        actions = original(agent, rows, acting, inference)
        calls += 1
        if calls == 4:
            actions[0] = -1
        return actions

    monkeypatch.setattr(collector, "_sample_learning_rows", sample)
    try:
        batch = collector.collect(agent, 1)
        attempts = [
            GameAttempt.model_validate(json.loads(line)["attempt"])
            for line in ledger.path.read_text().splitlines()
        ]
        assert ledger.state.crashed == 1 and ledger.state.completed == 2
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
        assert ledger.state.discarded_samples == 4
    finally:
        collector.close()
