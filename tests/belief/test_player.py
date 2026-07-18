from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.belief.likelihood import (
    FrozenPolicyLikelihood,
    LikelihoodResult,
    _matching_offer_indexes,
    file_sha256,
)
from manabot.belief.player import ExactRangePlayer, UniformRangePlayer
from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentHypers, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.sim.flat_mc import play_games
from manabot.sim.teacher1_evidence import _fresh_env
from managym.decision import SEMANTIC_DECISION_VERSION, DecisionFrame


class NeutralLikelihood:
    def evaluate(self, root_engine, *, viewer: int, commitment, belief):
        del root_engine, viewer, commitment
        return LikelihoodResult(
            likelihoods=np.ones(belief.support_size, dtype=np.float64),
            legal_action_counts=np.full(belief.support_size, 2, dtype=np.int64),
            matching_action_counts=np.ones(belief.support_size, dtype=np.int64),
            seconds=0.0,
        )


def test_python_authority_exposes_only_canonical_exact_world_materialization() -> None:
    engine = _fresh_env(79)._engine

    assert hasattr(engine, "possible_world_space_json")
    assert hasattr(engine, "materialize_possible_world")
    assert hasattr(engine, "flat_mc_scores_for_worlds")
    assert hasattr(engine, "step_semantic_command")
    assert hasattr(engine, "semantic_event_cursor")
    assert not hasattr(engine, "hidden_pool_summary")
    assert not hasattr(engine, "determinize_to_hand")
    assert not hasattr(engine, "flat_mc_scores_for_hands")


def test_provider_commitment_groups_duplicate_semantic_offers() -> None:
    frame = DecisionFrame(
        schema_version=SEMANTIC_DECISION_VERSION,
        revision=4,
        actor=1,
        fingerprint="frame",
        offers=(
            {"id": 0, "public_commitment": {"kind": "cast", "card": "Bolt"}},
            {"id": 1, "public_commitment": {"kind": "cast", "card": "Bolt"}},
            {
                "id": 2,
                "public_commitment": {"kind": "pass_priority"},
            },
            {"id": 3},
        ),
        object_candidates=(),
    )

    matching, legal_count = _matching_offer_indexes(
        frame, {"kind": "cast", "card": "Bolt"}
    )

    assert matching == [0, 1]
    assert legal_count == 4


def test_provider_commitment_groups_discard_family_without_physical_identity() -> None:
    frame = DecisionFrame(
        schema_version=SEMANTIC_DECISION_VERSION,
        revision=29,
        actor=0,
        fingerprint="discard-frame",
        offers=(
            {"id": 0, "public_commitment": {"kind": "discard", "card": "Island"}},
            {"id": 1, "public_commitment": {"kind": "discard", "card": "Island"}},
            {"id": 2, "public_commitment": {"kind": "decline_discard"}},
        ),
        object_candidates=(),
    )

    discard, legal_count = _matching_offer_indexes(
        frame, {"kind": "discard", "card": "Island"}
    )
    decline, _ = _matching_offer_indexes(frame, {"kind": "decline_discard"})

    assert discard == [0, 1]
    assert decline == [2]
    assert legal_count == 3


def test_frozen_likelihood_fails_closed_on_checkpoint_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "likelihood.pt"
    checkpoint.write_bytes(b"not the registered model")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenPolicyLikelihood(checkpoint, expected_sha256="0" * 64)


def test_belief_and_prior_use_identical_canonical_search_path() -> None:
    env = _fresh_env(83)
    viewer = int(env._engine.current_agent_index())
    likelihood = NeutralLikelihood()
    belief = ExactRangePlayer(1, likelihood=likelihood, seed=109)
    prior = UniformRangePlayer(1, likelihood=likelihood, seed=109)
    belief.start_game(env, viewer)
    prior.start_game(env, viewer)

    belief_action = belief.act(env, {})
    prior_action = prior.act(env, {})

    assert belief_action == prior_action
    assert np.array_equal(belief.last_scores, prior.last_scores)
    assert belief.stats.search.simulations == prior.stats.search.simulations
    assert belief.stats.materialization_failures == 0
    assert prior.stats.materialization_failures == 0


def test_player_emits_authoritative_revision_bound_command() -> None:
    env = _fresh_env(89)
    viewer = int(env._engine.current_agent_index())
    player = ExactRangePlayer(1, likelihood=NeutralLikelihood(), seed=127)
    player.start_game(env, viewer)
    action = player.act(env, {})
    frame = DecisionFrame.from_json(env._engine.semantic_decision_frame_json())

    command = player.command_for_action(env._engine, action, command_id="int-9-test")

    assert command.command_id == "int-9-test"
    assert command.expected_revision == frame.revision
    assert command.offer_id == frame.offers[action]["id"]
    assert player.stats.commands_emitted == 1
    assert player.evidence_stats()["p50_end_to_end_latency_ms"] is not None


def test_player_archives_canonical_history_at_game_boundary() -> None:
    env = _fresh_env(95)
    viewer = int(env._engine.current_agent_index())
    player = ExactRangePlayer(1, likelihood=NeutralLikelihood(), seed=130)
    player.start_game(env, viewer)

    player.finish_game(game_index=4, seed=95)

    replay = player.replay_receipts()[0]
    assert replay["game_index"] == 4
    assert replay["seed"] == 95
    assert replay["viewer"] == viewer
    assert replay["initial_space_id"]


def test_player_tracks_opponent_pass_from_semantic_receipt() -> None:
    env = _fresh_env(97)
    viewer = int(env._engine.current_agent_index())
    player = ExactRangePlayer(1, likelihood=NeutralLikelihood(), seed=131)
    player.start_game(env, viewer)

    for step in range(6):
        acting = int(env._engine.current_agent_index())
        frame = DecisionFrame.from_json(env._engine.semantic_decision_frame_json())
        pass_index = next(
            index
            for index, offer in enumerate(frame.offers)
            if offer.get("public_commitment") == {"kind": "pass_priority"}
        )
        player.prepare_step(env, acting, pass_index)
        command = player.command_for_action(
            env._engine, pass_index, command_id=f"pass-{step}"
        )
        _, _, _, _, _, transition = env.step_semantic(command)
        player.observe_step(env, acting, transition)
        if acting != viewer:
            break

    assert acting != viewer
    assert player.stats.action_updates == 1
    assert player.tracker is not None
    assert (
        player.tracker.posterior.space.identity == player.tracker.prior.space.identity
    )
    assert player.stats.peak_range_bytes > 0
    assert player.tracker.records[-1].public_commitment == {"kind": "pass_priority"}


def test_tiny_matchup_records_calibration_replay_and_system_cost(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tiny-policy.pt"
    obs_hypers = ObservationSpaceHypers()
    agent_hypers = AgentHypers()
    obs_space = ObservationSpace(obs_hypers)
    agent = Agent(obs_space, agent_hypers)
    torch.save(
        {
            "hypers": {
                "observation_hypers": obs_hypers.model_dump(),
                "agent_hypers": agent_hypers.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        checkpoint,
    )
    digest = file_sha256(checkpoint)
    common = {
        "sims": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "rollouts_per_world": 1,
        "max_steps": 200,
        "likelihood_batch_size": 16,
    }
    tiny_deck = {"Mountain": 4, "Raging Goblin": 4}

    result = play_games(
        {"kind": "exact_range", **common},
        {"kind": "uniform_range", **common},
        num_games=1,
        seed=157,
        hero_deck=tiny_deck,
        villain_deck=tiny_deck,
    )

    assert len(result.records) == 1
    assert result.hero_evidence is not None
    assert result.villain_evidence is not None
    assert result.hero_evidence["materialization_failures"] == 0
    assert result.hero_evidence["calibration"]["points"] > 0
    assert result.hero_evidence["p95_end_to_end_latency_ms"] is not None
    assert result.hero_evidence["peak_rss_bytes"] > 0
    assert result.hero_replays[0]["game_index"] == 0
    assert result.hero_known_truth
