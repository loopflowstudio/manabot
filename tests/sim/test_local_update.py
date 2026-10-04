"""Analytic mixing, exact branch replay and current-world collection proofs."""

from pathlib import Path

import numpy as np
import pytest
import torch

from etude.advice import BeliefNormalizationReceipt
from etude.local_advice import local_update_scenario
from manabot.arena.match import _execute_game
from manabot.arena.models import (
    PlayerRegistration,
    SearchSemantics,
    canonical_sha256,
)
from manabot.arena.replay import replay_games
from manabot.belief.likelihood import file_sha256
from manabot.belief.tracker import BeliefTracker
from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim import local_update
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.local_update import (
    LocalSearchConfig,
    LocalUpdateTeacher,
    regularized_update,
)
from managym.possible_worlds import WorldQuery


def small_match() -> MatchHypers:
    return MatchHypers(
        hero_deck={"Mountain": 8, "Gray Ogre": 8},
        villain_deck={"Forest": 8, "Llanowar Elves": 8},
    )


def make_teacher(tmp_path: Path) -> tuple[LocalUpdateTeacher, Env]:
    torch.set_num_threads(1)
    torch.manual_seed(15)
    space = ObservationSpace()
    agent = Agent(space, AgentHypers(hidden_dim=8, num_attention_heads=2))
    match = Match(small_match())
    path = tmp_path / "policy.pt"
    save_bc_checkpoint(
        agent,
        space,
        path,
        player_configs=match.to_rust(),
        extra={"value_semantic": "signed_outcome"},
    )
    teacher = LocalUpdateTeacher(
        path,
        file_sha256(path),
        LocalSearchConfig(worlds=2, depth=3, decision_seconds=10),
    )
    env = Env(
        match, space, Reward(RewardHypers()), auto_reset=False, enable_profiler=False
    )
    env.reset(seed=42)
    return teacher, env


def test_two_kl_optimum_and_shift_invariance() -> None:
    base = np.array([0.2, 0.8])
    reference = np.array([0.5, 0.5])
    values = np.array([-1.0, 1.0])
    result = regularized_update(base, values, reference, alpha=0.3, beta=2)
    # First-order simplex stationarity independently checks the optimizer.
    gradient = (
        values
        - 0.3 * (np.log(result / reference) + 1)
        - 2 * (np.log(result / base) + 1)
    )
    assert gradient[0] == pytest.approx(gradient[1])
    assert np.all(result > 0)
    np.testing.assert_allclose(
        result, regularized_update(base, values + 8, reference, alpha=0.3, beta=2)
    )
    with pytest.raises(ValueError, match="positive support"):
        regularized_update(np.array([0.0, 1.0]), values, reference, alpha=0.3, beta=2)


def test_real_root_replay_and_hidden_swap(tmp_path: Path) -> None:
    teacher, env = make_teacher(tmp_path)
    viewer = env._engine.current_agent_index()
    tracker = BeliefTracker.from_engine(
        env._engine, viewer=viewer, likelihood=teacher.likelihood, epsilon=0
    )
    before = env._engine.search_witness_json()
    receipt = teacher.search(env._engine, tracker.posterior, seed=7)
    replay = teacher.search(env._engine, tracker.posterior, seed=7)
    assert receipt.replay_identity() == replay.replay_identity()
    teacher.verify_replay(env._engine, tracker.posterior, receipt)
    assert before == env._engine.search_witness_json()
    assert receipt.rollouts and all(row.branch_audit_json for row in receipt.rollouts)
    assert len(receipt.offer_ids) == len(receipt.target)
    assert sum(receipt.target) == pytest.approx(1)
    for index in (0, tracker.space.support_size - 1):
        swapped = tracker.space.materialize(index, seed=95)
        swapped_tracker = BeliefTracker.from_engine(
            swapped, viewer=viewer, likelihood=teacher.likelihood, epsilon=0
        )
        result = teacher.search(swapped, swapped_tracker.posterior, seed=7)
        np.testing.assert_array_equal(receipt.target, result.target)
    assert receipt.mixing_diagnostics()["target_entropy"] >= 0
    with pytest.raises(TimeoutError):
        teacher.search(env._engine, tracker.posterior, seed=7, deadline=1)
    with pytest.raises(ValueError):
        teacher.search(
            env._engine,
            tracker.posterior,
            seed=7,
            query=WorldQuery.has("Lightning Bolt"),
        )


def test_advice_projection_is_schema_compatible_and_private(tmp_path: Path) -> None:

    teacher, env = make_teacher(tmp_path)
    tracker = BeliefTracker.from_engine(
        env._engine,
        viewer=env._engine.current_agent_index(),
        likelihood=teacher.likelihood,
        epsilon=0,
    )
    receipt = teacher.search(env._engine, tracker.posterior, seed=21)
    belief = BeliefNormalizationReceipt(
        scenario_id="baseline",
        space_identity=receipt.world_identity,
        belief_model_id=receipt.belief_model,
        distribution_sha256=receipt.belief_digest,
        normalized_belief_sha256=receipt.belief_digest,
        positive_support=tracker.posterior.positive_support_size,
        normalization_error=0,
        provenance_kind="model_inferred",
        provenance_identity=receipt.policy_sha256,
    )
    scenario = local_update_scenario(
        receipt, belief, labels=["offer"] * len(receipt.offer_ids)
    )
    assert sum(action.probability for action in scenario.actions) == pytest.approx(1)
    public = scenario.model_dump_json()
    for private_field in (
        "rollouts",
        "sampling_probabilities",
        "branch_audit_json",
        "world_seed",
    ):
        assert private_field not in public
    assert scenario.root_uncertainty.status == "unavailable"


def test_arena_uses_exact_range_lifecycle_and_replays(tmp_path: Path) -> None:

    teacher, _ = make_teacher(tmp_path)
    teacher.config = teacher.config.model_copy(
        update={"sampling": "compatible_prior", "worlds": 1, "depth": 1}
    )
    identity = "0" * 64
    common = dict(
        role="challenger",
        compute_class_id="bounded-fixture",
        information_boundary="acting-viewer-history-only-v1",
        world="w4",
        content_suite="tiny-fixture",
        observation_abi_sha256=identity,
        action_abi_sha256=identity,
        matchup_sha256=identity,
        player_seed_derivation_id="fixture-seed",
    )
    candidate = PlayerRegistration(
        **common,
        player_id="local-fixture",
        display_name="Local fixture",
        runner_kind="checkpoint",
        checkpoint_sha256=teacher.likelihood.checkpoint_sha256,
        checkpoint_bytes=teacher.likelihood.checkpoint.stat().st_size,
        parameter_count=sum(p.numel() for p in teacher.agent.parameters()),
        training_seed=15,
        artifact_id="untrained-fixture-15",
        evidence_class="fixture",
        player_spec={
            "kind": "local_update",
            "config": teacher.config.model_dump(),
            "implementation_source_sha256": file_sha256(Path(local_update.__file__)),
        },
        search_call_seed_derivation_id="local-policy-seed-times-1000003-plus-call-mod-2pow63/v1",
        search_semantics=SearchSemantics(
            branch_audit=True,
            root_prior="two-kl-local-update/v1",
            leaf_evaluator="frozen-viewer-policy-signed-value/v1",
        ),
    )
    opponent = PlayerRegistration(
        **common,
        player_id="random-fixture",
        display_name="Random",
        runner_kind="code",
        source_sha256=identity,
        player_spec={"kind": "random"},
    )
    games = []
    for leg in range(2):
        registrations = [candidate, opponent] if leg == 0 else [opponent, candidate]
        game = dict(
            match_id=f"fixture-{leg}",
            deal_seed=93,
            leg=leg,
            match_hypers=small_match().model_dump(),
            seat_players=[p.player_id for p in registrations],
            player_seeds={p.player_id: 7 for p in registrations},
            decisions=[],
            integrity=dict(
                private_exposures=0, root_mutations=0, illegal_actions=0, truncations=0
            ),
        )
        messages: list[tuple[str, object]] = []
        _execute_game(
            game,
            registrations,
            {candidate.player_id: str(teacher.likelihood.checkpoint)},
            1000,
            messages.append,
        )
        assert game.get("terminated"), game.get("failure")
        assert any("local_update_receipt" in decision for decision in game["decisions"])
        game["game_trace_sha256"] = canonical_sha256(game)
        games.append(game)
    assert replay_games(games).passed
