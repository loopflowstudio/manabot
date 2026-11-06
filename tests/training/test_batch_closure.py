"""Batch closure is declared apart from learning seats; saved regimes keep theirs."""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError
import pytest

from experiments.batched_baseline import (
    closure_comparison,
    lambda_sweep,
    scale_ladder,
    seat_comparison,
)
from manabot.env.match import Match, Reward
from manabot.experiment.attacks import AttackPlan, AttackTarget
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.infra.identity import canonical_sha256
from manabot.training.regime import (
    BatchClosure,
    FrozenOpponent,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.state.game_errors import GameErrorPolicy
import managym
from tests.training.test_two_seat_learning import ScriptedGames, _agent

ROOT = Path(__file__).resolve().parents[2]
OPPONENT = FrozenOpponent(path="target.pt", sha256="a" * 64)


def _stage(**fields: object) -> TrainSelfPlay:
    return TrainSelfPlay.model_validate(
        {"id": "self-play", "operation": "train_self_play", **fields}
    )


def _regime(stage: TrainSelfPlay) -> TrainingRegime:
    return TrainingRegime(
        id="closure",
        world=managym.WORLD_VERSION,
        match=MatchConfig(),
        stages=[stage],
    )


def _saved(**fields: object) -> dict[str, object]:
    """A regime as old saves spell it: the stage has no `batch_closure` key."""
    dumped = _regime(_stage()).model_dump(mode="json")
    stage = {"id": "self-play", "operation": "train_self_play", **fields}
    return {**dumped, "stages": [stage]}


def _closure(regime: TrainingRegime) -> BatchClosure:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    return stage.batch_closure


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"learning_seats": "one"},
        {"behavior": "random"},
        {"behavior": "frozen", "opponent": OPPONENT},
    ],
)
def test_new_regimes_default_to_complete_games(fields: dict[str, object]) -> None:
    regime = _regime(_stage(**fields))
    assert _closure(regime) == "complete-games"
    assert _closure(TrainingRegime.model_validate(regime.model_dump())) == (
        "complete-games"
    )


@pytest.mark.parametrize(
    ("fields", "closure"),
    [
        ({"learning_seats": "one"}, "exact-quota"),
        ({"behavior": "random"}, "exact-quota"),
        (
            {"behavior": "frozen", "opponent": OPPONENT.model_dump()},
            "exact-quota",
        ),
        ({"learning_seats": "one", "games_per_stream": 2}, "complete-games"),
        (
            {"learning_seats": "one", "game_errors": GameErrorPolicy().model_dump()},
            "complete-games",
        ),
        ({}, "complete-games"),
        ({"learning_seats": "both", "games_per_stream": 1}, "complete-games"),
    ],
)
def test_saved_stages_resolve_as_they_ran_and_keep_their_bytes(
    fields: dict[str, object], closure: BatchClosure
) -> None:
    regime = TrainingRegime.model_validate(_saved(**fields))
    assert _closure(regime) == closure
    (stage,) = regime.model_dump(mode="json")["stages"]
    assert "batch_closure" not in stage


def test_schema_one_stage_is_a_saved_one_seat_exact_quota() -> None:
    regime = TrainingRegime.model_validate({**_saved(), "schema_version": 1})
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    assert (stage.learning_seats, stage.batch_closure) == ("one", "exact-quota")
    (dumped,) = regime.model_dump(mode="json")["stages"]
    assert "learning_seats" not in dumped and "batch_closure" not in dumped


def test_explicit_closure_survives_a_round_trip() -> None:
    exact = _regime(_stage(learning_seats="one", batch_closure="exact-quota"))
    complete = _regime(_stage(learning_seats="one", batch_closure="complete-games"))
    (exact_stage,) = exact.model_dump(mode="json")["stages"]
    (complete_stage,) = complete.model_dump(mode="json")["stages"]
    assert "batch_closure" not in exact_stage
    assert complete_stage["batch_closure"] == "complete-games"
    for regime in (exact, complete):
        assert _closure(TrainingRegime.model_validate(regime.model_dump())) == (
            _closure(regime)
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"learning_seats": "both"},
        {"learning_seats": "one", "games_per_stream": 2},
        {"learning_seats": "one", "game_errors": GameErrorPolicy()},
    ],
)
def test_exact_quota_rejects_settings_that_need_whole_games(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="complete games"):
        _stage(batch_closure="exact-quota", **fields)


def test_exact_quota_accepts_capacity_recovery() -> None:
    stage = _stage(
        learning_seats="one",
        batch_closure="exact-quota",
        game_errors=GameErrorPolicy(recover_capacity=True),
    )
    regime = _regime(stage)
    assert _closure(TrainingRegime.model_validate(regime.model_dump())) == "exact-quota"


def test_a_game_error_policy_does_not_choose_the_closure() -> None:
    without = _stage(learning_seats="one")
    with_policy = _stage(learning_seats="one", game_errors=GameErrorPolicy())
    assert without.batch_closure == with_policy.batch_closure == "complete-games"


@pytest.mark.parametrize(
    "fields",
    [
        {"learning_seats": "both"},
        {"learning_seats": "one", "games_per_stream": 2},
        {
            "learning_seats": "one",
            "game_errors": GameErrorPolicy(),
            "record_attempt": lambda _: None,
        },
    ],
)
def test_collector_rejects_exact_quota_with_whole_game_settings(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="exact quota"):
        SeatRoutedCollector(
            _agent().observation_space,
            Match(),
            Reward(RewardConfig()),
            num_envs=2,
            opponent_mode="self",
            batch_closure="exact-quota",
            **fields,
        )


def test_continuation_preserves_closure() -> None:
    first = _stage(learning_seats="one", batch_closure="exact-quota")
    second = _stage(id="next", initial="self-play", learning_seats="one")
    with pytest.raises(ValidationError, match="batch closure"):
        TrainingRegime(
            id="closure",
            world=managym.WORLD_VERSION,
            match=MatchConfig(),
            stages=[first, second],
        )


def _one_seat(closure: BatchClosure) -> SeatRoutedCollector:
    agent = _agent()
    collector = SeatRoutedCollector(
        agent.observation_space,
        Match(),
        Reward(RewardConfig()),
        num_envs=2,
        opponent_mode="self",
        learning_seats="one",
        batch_closure=closure,
    )
    collector._env = ScriptedGames(collector._buffers)
    return collector


def test_exact_quota_pauses_mid_game_at_the_quota() -> None:
    agent = _agent()
    batch = _one_seat("exact-quota").collect(agent, 2)
    assert batch.dones.shape == (2, 2)
    assert not batch.next_done.all()


def test_one_seat_complete_games_end_every_trajectory_at_a_result() -> None:
    agent = _agent()
    collector = _one_seat("complete-games")
    batch = collector.collect(agent, 2)
    assert batch.obs["agent_player"].shape[1] == 1
    assert bool(batch.next_done.all())
    # Every packed trajectory closes on a terminal row, so nothing bootstraps.
    assert bool(batch.dones[-1].all())
    assert np.count_nonzero(batch.dones) >= 1
    assert collector.stats.games >= 2
    # Stream 1's shorter game leaves it idle while stream 0 finishes.
    assert collector.stats.idle_stream_steps > 0


def _attack(schema_version: Literal[1, 2]) -> AttackPlan:
    return AttackPlan(
        id="attack-proof",
        schema_version=schema_version,
        targets=(AttackTarget(id="target-a", policy=OPPONENT, producer_seeds=(17,)),),
        attacker_seeds=(101, 201),
        cumulative_updates=(2, 5),
        final_deal_seeds=(950001,),
        template=TrainingRegime(
            id="attacker",
            world=managym.WORLD_VERSION,
            match=MatchConfig(),
            wall_seconds=30,
            stages=[_stage()],
        ),
        total_seconds=1000,
        evaluation_seconds=10,
        prediction="closure",
    )


@pytest.mark.parametrize(
    ("schema_version", "closure"), [(1, "exact-quota"), (2, "complete-games")]
)
def test_attack_plans_keep_saved_closure_and_default_new_ones(
    schema_version: Literal[1, 2], closure: BatchClosure
) -> None:
    plan = _attack(schema_version)
    regime = plan.regime_for(plan.targets[0])
    assert {
        stage.batch_closure
        for stage in regime.stages
        if isinstance(stage, TrainSelfPlay)
    } == {closure}


@pytest.mark.parametrize(
    ("name", "digest"),
    [
        (
            "elemental-two-seat-181-v1",
            "c797953518dd0349b473ea451b4d6d3dcaee4243a23b044e6fca68d27befe899",
        ),
        (
            "elemental-long-183-v1",
            "48f56059fa2ed072637e573a0a72a6260f5cab56225e877559206f6348c2d62d",
        ),
        (
            "tla-trophy-213-v9",
            "04d344024aa8428a125936a2d709588d5fdf9c26af1e2b653d38550b9f47048c",
        ),
    ],
)
def test_saved_regime_digests_do_not_move(name: str, digest: str) -> None:
    payload = json.loads((ROOT / f"experiments/regimes/{name}.json").read_text())
    regime = TrainingRegime.model_validate(payload)
    assert canonical_sha256(regime.model_dump(mode="json")) == digest


def test_frozen_one_seat_recipes_keep_their_closure_and_digest() -> None:
    ladder = scale_ladder()
    assert _closure(ladder) == "exact-quota"
    assert canonical_sha256(ladder.model_dump(mode="json")) == (
        "ea87e37fee96bf5f64854a4ed27819a359a854174526adb5d45d350294c6946c"
    )
    one = seat_comparison("one", 10)
    assert _closure(one) == "complete-games"
    assert canonical_sha256(one.model_dump(mode="json")) == (
        "91be75cc279047c86590b5d8c8ca52d23f63f20a5d4930b89a1c9dca1f2108f8"
    )


def _changed(first: dict[str, object], second: dict[str, object]) -> set[str]:
    return {
        key for key in first.keys() | second.keys() if first.get(key) != second.get(key)
    }


def test_closure_study_arms_differ_only_in_closure_and_floor() -> None:
    exact = closure_comparison("exact-quota", 10).model_dump(mode="json")
    complete = closure_comparison("complete-games", 10).model_dump(mode="json")
    assert _changed(exact, complete) == {"id", "stages"}
    assert _changed(exact["stages"][0], complete["stages"][0]) == {
        "batch_closure",
        "transitions",
    }


def test_lambda_study_arms_differ_only_in_their_traces() -> None:
    first = lambda_sweep(0.5, 0.8, 10).model_dump(mode="json")
    second = lambda_sweep(0.95, 1.0, 10).model_dump(mode="json")
    assert _changed(first, second) == {"id", "stages"}
    assert _changed(first["stages"][0], second["stages"][0]) == {"learning"}
    assert _changed(
        first["stages"][0]["learning"], second["stages"][0]["learning"]
    ) == {
        "policy_lambda",
        "value_lambda",
    }
