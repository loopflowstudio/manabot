"""The explicit policy prefix survives collection and evaluation encodings."""

from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.backends.torch.collection import (
    CollectionSession as TorchCollectionSession,
)
from manabot.backends.torch.policy import ActorCritic
from manabot.env.env import Env
from manabot.env.failures import CaptureScope, RecordedVectorEnv, capture_scope
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace, step_buffers
from manabot.infra.config import (
    MatchConfig,
    ModelConfig,
    ObservationConfig,
    RewardConfig,
)
from manabot.sim.heuristic import HeuristicPlayer
from manabot.training.rollout.collection import (
    close_collector,
    collect,
    open_collector,
)
import managym
from managym.decision import Command, DecisionFrame
from tests.fixtures.observation_rows import observation_arrays
from tests.fixtures.rollout_arrays import trajectory_rows


def _distinct_filler_creatures(count: int) -> list[str]:
    defs = json.loads(managym.card_definitions_json())
    disallowed = (
        "foretell",
        "adventure",
        "omen",
        "cycling",
        "channel",
        "warp",
        "plot",
        "evoke",
        "prototype",
        "spree",
        "offspring",
        "kicker",
        "exhaust",
        "bestow",
        "additional cost",
        "double strike",
        "flying",
        "deathtouch",
        "lifelink",
        "hexproof",
        "indestructible",
        "trample",
        "first strike",
        "vigilance",
        "menace",
        "reach",
        "flash",
        "haste",
    )
    fillers = [
        card["name"]
        for card in defs
        if not card["is_token"]
        and card["types"]["types"] == ["Creature"]
        and card["name"] not in ("Hill Giant", "Shivan Dragon")
        and card["mana_cost"] is not None
        and 1 <= card["mana_cost"]["mana_value"] <= 3
        and card["mana_cost"]["cost"][5] == 0
        and all(v <= 2 for v in card["mana_cost"]["cost"][:5])
        and not card["targeting"]
        and not card["activated_abilities"]
        and (card.get("power") or 0) <= 2
        and (card.get("toughness") or 0) <= 2
        and not any(word in card.get("text_box", "").lower() for word in disallowed)
    ]
    assert len(fillers) >= count
    return fillers[:count]


def test_heuristic_scores_only_retained_legal_offers(tmp_path: Path) -> None:
    basics = ("Plains", "Island", "Swamp", "Mountain", "Forest")
    fillers = _distinct_filler_creatures(64)
    deck = {
        **{land: 4 for land in basics},
        "Hill Giant": 1,
        "Shivan Dragon": 1,
        **{name: 1 for name in fillers},
    }
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128, max_actions=64, action_overflow="first-legal"
        )
    )
    with capture_scope(CaptureScope(tmp_path)):
        env = Env(
            Match(MatchConfig(hero_deck=deck, villain_deck=deck)),
            space,
            Reward(RewardConfig()),
            seed=213,
        )
        try:
            env.reset()
            env.step(0)
            env.step(0)
            actor = env.last_raw_obs.agent.player_index
            engine = env._engine
            engine.scenario_clear_hand(actor)
            engine.scenario_force_card_in_hand(actor, "Hill Giant")
            for name in fillers:
                engine.scenario_force_card_in_hand(actor, name)
            engine.scenario_force_card_in_hand(actor, "Shivan Dragon")
            for land in basics:
                for _ in range(2):
                    engine.scenario_force_battlefield(actor, land)
            raw = env._last_obs = engine.scenario_refresh()
            obs = observation_arrays(space, raw)
            total_actions = len(raw.action_space.actions)
            assert total_actions >= 67
            assert obs["actions_valid"].sum() == 64
            frame = json.loads(engine.semantic_decision_frame_json())
            before = engine.state_digest()
            unrestricted = HeuristicPlayer().act(env, raw)
            retained = HeuristicPlayer(max_actions=64).act(env, raw)
            assert unrestricted == total_actions - 2 and retained == 0
            cards = {card.id: card.characteristics.name for card in raw.agent_cards}
            assert (
                cards[raw.action_space.actions[unrestricted].focus[0]]
                == "Shivan Dragon"
            )
            assert cards[raw.action_space.actions[retained].focus[0]] == "Hill Giant"
            assert engine.state_digest() == before
            assert json.loads(engine.semantic_decision_frame_json()) == frame
            assert len(env.last_raw_obs.action_space.actions) == total_actions
            # Execute the exact retained offer, without clamping or remapping.
            decision = DecisionFrame.from_json(engine.semantic_decision_frame_json())
            command = Command(
                "matched-prefix",
                decision.revision,
                int(decision.offers[retained]["id"]),
            )
            env.step_semantic(command)
            assert engine.state_digest() != before
        finally:
            env.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["counts_by_kind"] == {
        "Modal": {"decisions": 2, "overflows": 0, "dropped_actions": 0},
        "Priority": {
            "decisions": 1,
            "overflows": 1,
            "dropped_actions": total_actions - 64,
        },
    }
    assert report["samples"][0]["game"]["actor"] == actor


@pytest.mark.parametrize("workers", [1, 2])
def test_collector_uses_configured_prefix_and_one_masked_distribution(
    workers: int, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    agent = ActorCritic(space, ModelConfig(hidden_dim=8, attention_on=False))
    distinct_lands = {
        "Mountain": 6,
        "Forest": 6,
        "Plains": 6,
        "Island": 6,
        "Swamp": 6,
    }
    match = Match(MatchConfig(hero_deck=distinct_lands, villain_deck=distinct_lands))
    with capture_scope(CaptureScope(tmp_path)):
        collector = open_collector(
            space,
            match,
            Reward(RewardConfig()),
            num_envs=2,
            seed=213,
            workers=workers,
            opponent_mode="self",
            learning_seats="both",
            numerics=TorchCollectionSession(
                device="cpu", seed=213, precision="float32"
            ),
        )
        try:
            batch = collect(collector, agent, 2)
        finally:
            close_collector(collector)
    rows = trajectory_rows(batch)
    valid = rows.observation["actions_valid"].astype(bool)
    assert rows.log_probabilities.shape == valid.shape
    assert np.all(np.isneginf(rows.log_probabilities[~valid]))
    assert np.all(np.isfinite(rows.log_probabilities[valid]))
    np.testing.assert_allclose(np.exp(rows.log_probabilities).sum(-1), 1, atol=1e-6)
    assert np.all(np.take_along_axis(valid, batch.actions[..., None], -1))
    reports = [
        json.loads(path.read_bytes()) for path in tmp_path.glob("action-cutoff-*.json")
    ]
    assert len(reports) == workers
    assert (
        sum(
            sum(counts["overflows"] for counts in report["counts_by_kind"].values())
            for report in reports
        )
        > 0
    )


def test_scalar_evaluation_cutoff_is_diagnostic_not_game_truncation(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    distinct_lands = {
        "Mountain": 6,
        "Forest": 6,
        "Plains": 6,
        "Island": 6,
        "Swamp": 6,
    }
    match = Match(MatchConfig(hero_deck=distinct_lands, villain_deck=distinct_lands))
    with capture_scope(CaptureScope(tmp_path)):
        env = Env(match, space, Reward(RewardConfig()), seed=213)
        try:
            env.reset()
            env.step(0)
            obs, *_ = env.step(0)
            assert managym.encode_observation_batch(
                [obs], space.layout, "first-legal"
            ).action_counts == [2]
            assert len(env.last_raw_obs.action_space.actions) > 2
            actor = env.last_raw_obs.agent.player_index
            _, _, _, truncated, info = env.step(0)
            assert not truncated and not info["action_space_truncated"]
        finally:
            env.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["samples"][0]["game"]["seed"] == 213
    assert report["samples"][0]["retained_count"] == 2
    sample = report["samples"][0]
    assert sample["decision_kind"] == "Priority"
    assert sample["game"]["actor"] == actor
    assert json.loads(sample["game"]["frame"])["actor"] == actor
    assert sample["game"]["actions"] == [0, 0]


def test_vector_reseed_retains_consumed_decision_counts(tmp_path: Path) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=64, action_overflow="first-legal")
    )
    match = Match(
        MatchConfig(hero_deck={"Mountain": 30}, villain_deck={"Mountain": 30})
    )
    with (
        capture_scope(CaptureScope(tmp_path)),
        closing(RecordedVectorEnv(num_envs=1)) as env,
    ):
        env.set_observation_layout(
            step_buffers(1), space.config.action_overflow, layout=space.layout
        )
        for seed in (213, 214):
            env.reset_each_into_buffers([match.to_rust()], seeds=[seed])
            env.step_into_buffers([0])
    reports = [
        json.loads(path.read_bytes()) for path in tmp_path.glob("action-cutoff-*.json")
    ]
    (report,) = reports
    assert report["counts_by_kind"] == {
        "Modal": {"decisions": 2, "overflows": 0, "dropped_actions": 0}
    }
