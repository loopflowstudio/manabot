"""Behavioral proof for supervised exact-world belief learning."""

from dataclasses import replace
import json

import numpy as np
import pytest

from manabot.belief.demo import _runtime_env
from manabot.belief.encoding import belief_schema_from_engine
from manabot.belief.learning import (
    BehaviorPopulationProvenance,
    BeliefPopulationDataset,
    BeliefPopulationEpisode,
    BeliefTrainingExample,
    capture_materialized_world_supervision,
    pack_belief_examples,
    split_belief_population,
    train_bounded_population,
)
from manabot.belief.learning_demo import run_demo
from manabot.belief.range import BeliefError
from manabot.belief.state import ViewerHistory, ViewerHistoryEvent
from managym.decision import Observation, PublicCommitment
from managym.possible_worlds import PossibleWorldSpace
from tests.belief.support import fixture_history, fixture_schema, fixture_space


def test_ragged_training_batch_uses_exact_combinatorial_p0() -> None:
    history = fixture_history()
    space = fixture_space(history)
    schema = fixture_schema(space)
    examples = tuple(
        BeliefTrainingExample(
            world_space=space,
            viewer_history=history,
            target_world=target,
            supervision_receipt=f"test-supervision-{target}",
        )
        for target in (1, 3)
    )

    packed = pack_belief_examples(examples, schema)

    support = space.support_size
    expected = np.asarray([world.weight / space.total_weight for world in space.worlds])
    assert packed.world_counts.shape == (2 * support, len(space.pool))
    assert packed.offsets.tolist() == [0, support, 2 * support]
    assert packed.world_batch.tolist() == [0] * support + [1] * support
    assert packed.target_world.tolist() == [1, 3]
    assert packed.history_actor_role_ids.numel() == 0
    assert packed.history_kind_ids.numel() == 0
    assert packed.history_card_indexes.numel() == 0
    assert np.allclose(
        packed.log_p0[:support].exp().numpy(), expected, atol=1e-15, rtol=0.0
    )


def test_semantic_history_pooling_is_explicitly_order_invariant() -> None:
    history = fixture_history()
    space = fixture_space(history)
    schema = fixture_schema(space)
    events = (
        ViewerHistoryEvent(1, PublicCommitment("play_land", "Mountain")),
        ViewerHistoryEvent(1, PublicCommitment("pass_priority")),
        ViewerHistoryEvent(0, PublicCommitment("cast", "Lightning Bolt")),
    )
    forward = BeliefTrainingExample(
        world_space=space,
        viewer_history=replace(history, semantic_events=events),
        target_world=0,
        supervision_receipt="forward",
    )
    reverse = replace(
        forward,
        viewer_history=replace(history, semantic_events=tuple(reversed(events))),
        supervision_receipt="reverse",
    )
    artifact_variant = replace(
        forward,
        viewer_history=replace(
            forward.viewer_history,
            initial_observation_identity="f" * 64,
            events=("different-receipt-identity",),
        ),
        supervision_receipt="different-supervision-receipt",
    )

    forward_batch = pack_belief_examples((forward,), schema)
    reverse_batch = pack_belief_examples((reverse,), schema)
    artifact_batch = pack_belief_examples((artifact_variant,), schema)

    assert forward.viewer_history.identity != reverse.viewer_history.identity
    assert forward_batch.history_actor_role_ids.equal(
        reverse_batch.history_actor_role_ids
    )
    assert forward_batch.history_kind_ids.equal(reverse_batch.history_kind_ids)
    assert forward_batch.history_card_indexes.equal(reverse_batch.history_card_indexes)
    assert artifact_variant.viewer_history.identity != forward.viewer_history.identity
    assert artifact_variant.supervision_receipt != forward.supervision_receipt
    assert forward_batch.world_counts.equal(artifact_batch.world_counts)
    assert forward_batch.log_p0.equal(artifact_batch.log_p0)
    assert forward_batch.history_actor_role_ids.equal(
        artifact_batch.history_actor_role_ids
    )
    assert forward_batch.history_kind_ids.equal(artifact_batch.history_kind_ids)
    assert forward_batch.history_card_indexes.equal(artifact_batch.history_card_indexes)
    first = train_bounded_population((forward, reverse), schema, steps=2, seed=197)
    second = train_bounded_population((forward, reverse), schema, steps=2, seed=197)
    assert first.model.identity == second.model.identity
    assert first.final_nll == second.final_nll


def test_population_split_is_by_episode_and_provenance_is_not_packed() -> None:
    history = fixture_history()
    space = fixture_space(history)
    schema = fixture_schema(space)
    episodes = tuple(
        BeliefPopulationEpisode(
            episode_id=f"deal-{episode}",
            examples=(
                BeliefTrainingExample(
                    world_space=space,
                    viewer_history=replace(
                        history,
                        initial_observation_identity=f"{episode + 1:064x}",
                        events=(f"artifact-{episode}",),
                    ),
                    target_world=episode % space.support_size,
                    supervision_receipt=f"private-label-{episode}",
                ),
            ),
        )
        for episode in range(6)
    )
    dataset = BeliefPopulationDataset(
        episodes=episodes,
        source_distribution="compatible-deal-belief/v1",
        behavior_population=BehaviorPopulationProvenance("frozen-policy", "1"),
        sampling_seed=197,
    )

    split = split_belief_population(dataset, held_out_episodes=2, seed=991)
    training_ids = {episode.episode_id for episode in split.training_episodes}
    held_out_ids = {episode.episode_id for episode in split.held_out_episodes}
    packed = pack_belief_examples(split.training_examples, schema)

    assert training_ids.isdisjoint(held_out_ids)
    assert len(split.training_examples) == 4
    assert len(split.held_out_examples) == 2
    assert not hasattr(packed, "episode_id")
    assert not hasattr(packed, "behavior_population")
    assert not hasattr(packed, "supervision_receipt")


def test_supervision_rejects_an_authority_engine_from_another_revision() -> None:
    env, _ = _runtime_env()
    engine = env._engine
    viewer = int(engine.current_agent_index())
    observation = Observation.from_json(engine.semantic_observation_json(viewer))
    history = ViewerHistory.from_observation(observation)
    space = PossibleWorldSpace.from_engine(engine, viewer)
    schema = belief_schema_from_engine(
        engine,
        space,
        count_buckets=max(2, max(count for _, count in space.pool) + 1),
    )

    class MismatchedAuthority:
        def semantic_observation_json(self, requested_viewer: int) -> str:
            payload = json.loads(engine.semantic_observation_json(requested_viewer))
            payload["identity"]["revision"] += 1
            return json.dumps(payload)

        def observation_for_player(self, player: int):
            return engine.observation_for_player(player)

    with pytest.raises(BeliefError, match="authority revision"):
        capture_materialized_world_supervision(
            MismatchedAuthority(),
            world_space=space,
            viewer_history=history,
            schema=schema,
        )


def test_real_population_learning_is_held_out_and_beats_p0() -> None:
    evidence = run_demo(episodes=160, held_out_episodes=32, steps=16, seed=197)

    assert evidence["proof"] == "frozen-population-supervised-exact-world-belief"
    assert evidence["population"]["episodes"] == 160
    assert evidence["population"]["distinct_sampled_worlds"] > 100
    assert evidence["population"]["real_managym_transitions"] == 160
    assert sum(evidence["population"]["public_commitments"].values()) == 160
    assert evidence["split"] == {
        "unit": "deal-episode",
        "training_episodes": 128,
        "held_out_episodes": 32,
        "episode_identity_overlap": 0,
        "semantic_history_groups_in_both_arms": 2,
    }
    assert evidence["supervision"]["semantic_history_groups"] == 3
    assert (
        evidence["supervision"]["repeated_history_groups_with_multiple_world_labels"]
        == 3
    )
    assert evidence["model_boundary"]["policy_version_is_provenance_only"] is True
    assert "acting_policy_identity" in evidence["model_boundary"]["excluded_inputs"]
    assert "supervision_receipt" in evidence["model_boundary"]["excluded_inputs"]
    assert evidence["training"]["fresh_model_started_at_p0"] is True
    assert (
        evidence["training"]["final_joint_nll"] < evidence["training"]["p0_joint_nll"]
    )
    assert evidence["held_out"]["joint_nll_improvement_nats"] > 0.1
    assert evidence["held_out"]["inclusion_brier_improvement"] > 0.0
    assert evidence["held_out"]["inclusion_ece_improvement"] > 0.0
    assert evidence["held_out"]["learned"]["credible_90_coverage"] >= 0.8
    assert evidence["cost"]["training_support_sizes"] == {4_865: 125, 10_832: 3}
    assert evidence["cost"]["held_out_support_sizes"] == {4_865: 32}
    assert evidence["cost"]["training_candidate_world_rows_per_step"] == (
        125 * 4_865 + 3 * 10_832
    )
    assert evidence["cost"]["held_out_candidate_world_rows"] == 32 * 4_865
    assert evidence["cost"]["total_seconds"] > 0.0
