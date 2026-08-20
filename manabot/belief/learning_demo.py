"""Bounded population-supervision experiment for exact-world belief learning."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import time
from typing import Any, Mapping

import numpy as np

from manabot.belief.demo import _runtime_env
from manabot.belief.encoding import BeliefEncodingSchema, belief_schema_from_engine
from manabot.belief.learning import (
    BehaviorPopulationProvenance,
    BeliefPopulationDataset,
    BeliefPopulationEpisode,
    BeliefPopulationEpisodeProvenance,
    BeliefTrainingExample,
    capture_materialized_world_supervision,
    compare_population_to_p0,
    split_belief_population,
    train_bounded_population,
)
from manabot.belief.range import BeliefState
from manabot.belief.state import CompatibleDealBeliefModel, ViewerHistory
from managym.decision import (
    Command,
    DecisionFrame,
    Observation,
    apply_semantic_command,
)
from managym.possible_worlds import PossibleWorldSpace

SCRIPTED_POPULATION = BehaviorPopulationProvenance(
    policy="scripted-most-held-land-then-pass",
    version="1",
)


def _scripted_offer(frame: DecisionFrame) -> Mapping[str, Any]:
    """Choose the most-held land, breaking ties by canonical card name."""

    land_offers = tuple(offer for offer in frame.offers if offer["verb"] == "play_land")
    if not land_offers:
        return next(offer for offer in frame.offers if offer["verb"] == "pass_priority")
    land_counts = Counter(
        str(offer["public_commitment"]["card"]) for offer in land_offers
    )
    selected_name = min(land_counts, key=lambda name: (-land_counts[name], name))
    return next(
        offer
        for offer in land_offers
        if offer["public_commitment"]["card"] == selected_name
    )


def _history_key(example: BeliefTrainingExample) -> str:
    return json.dumps(
        [event.to_payload() for event in example.viewer_history.semantic_events],
        sort_keys=True,
        separators=(",", ":"),
    )


def _capture_population(
    *,
    episodes: int,
    seed: int,
) -> tuple[BeliefPopulationDataset, BeliefEncodingSchema, dict[str, int], int]:
    env, _ = _runtime_env()
    root = env._engine
    root_frame = DecisionFrame.from_json(root.semantic_decision_frame_json())
    acting = root_frame.actor
    viewer = (acting + 1) % 2
    root_space = PossibleWorldSpace.from_engine(root, viewer)
    schema = belief_schema_from_engine(
        root,
        root_space,
        count_buckets=max(2, max(count for _, count in root_space.pool) + 1),
    )
    source_distribution = CompatibleDealBeliefModel().identity
    sampled_worlds = BeliefState.compatible_prior(
        root_space,
        model_id=source_distribution,
    ).sample_indexes(episodes, seed=seed)
    rows: list[BeliefPopulationEpisode] = []
    commitment_counts: Counter[str] = Counter()
    for episode_index, world_index in enumerate(sampled_worlds):
        materialization_seed = seed * 1_000_003 + episode_index
        branch = root_space.materialize(
            world_index,
            seed=materialization_seed,
            refresh_opponent_commitment=True,
        )
        initial = Observation.from_json(branch.semantic_observation_json(viewer))
        history = ViewerHistory.from_observation(initial)
        frame = DecisionFrame.from_json(branch.semantic_decision_frame_json())
        if frame.actor != acting:
            raise RuntimeError("materialized population changed the acting player")
        offer = _scripted_offer(frame)
        transition = apply_semantic_command(
            branch,
            Command(
                command_id=f"belief-population-{episode_index}",
                expected_revision=frame.revision,
                offer_id=int(offer["id"]),
            ),
        )
        current = Observation.from_json(branch.semantic_observation_json(viewer))
        history = history.advance(transition.receipt, current, acting=acting)
        if len(history.semantic_events) != 1:
            raise RuntimeError("population rollout did not emit one public commitment")
        commitment = history.semantic_events[0].commitment
        commitment_counts[
            commitment.kind
            if commitment.card is None
            else f"{commitment.kind}:{commitment.card}"
        ] += 1
        space = PossibleWorldSpace.from_engine(branch, viewer)
        example = capture_materialized_world_supervision(
            branch,
            world_space=space,
            viewer_history=history,
            schema=schema,
        )
        rows.append(
            BeliefPopulationEpisode(
                episode_id=f"deal-{episode_index:04d}",
                provenance=BeliefPopulationEpisodeProvenance.from_transition(
                    source_world_space_identity=root_space.identity,
                    sampled_world_index=world_index,
                    materialization_seed=materialization_seed,
                    transition=transition.receipt,
                ),
                examples=(example,),
            )
        )
    return (
        BeliefPopulationDataset(
            episodes=tuple(rows),
            source_distribution=source_distribution,
            behavior_population=SCRIPTED_POPULATION,
            sampling_seed=seed,
        ),
        schema,
        dict(sorted(commitment_counts.items())),
        len(set(sampled_worlds)),
    )


def run_demo(
    *,
    episodes: int = 160,
    held_out_episodes: int = 32,
    steps: int = 16,
    seed: int = 197,
) -> dict[str, Any]:
    """Train once on sampled deals and score untouched deals against exact p0."""

    total_started = time.perf_counter()
    dataset_started = time.perf_counter()
    dataset, schema, commitment_counts, distinct_sampled_worlds = _capture_population(
        episodes=episodes,
        seed=seed,
    )
    dataset_seconds = time.perf_counter() - dataset_started
    split = split_belief_population(
        dataset,
        held_out_episodes=held_out_episodes,
        seed=seed ^ 0x5EED,
    )

    training_started = time.perf_counter()
    trained = train_bounded_population(
        split.training_examples,
        schema,
        steps=steps,
        seed=seed,
    )
    training_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    comparison = compare_population_to_p0(
        split.held_out_examples,
        schema,
        trained.model,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started

    histories: dict[str, list[BeliefTrainingExample]] = defaultdict(list)
    for episode in dataset.episodes:
        for example in episode.examples:
            histories[_history_key(example)].append(example)
    repeated_truth_groups = sum(
        len(examples) > 1
        and len(
            {
                example.world_space.world(example.target_world).hand
                for example in examples
            }
        )
        > 1
        for examples in histories.values()
    )
    training_history_keys = {
        _history_key(example) for example in split.training_examples
    }
    held_out_history_keys = {
        _history_key(example) for example in split.held_out_examples
    }
    training_episode_ids = {episode.episode_id for episode in split.training_episodes}
    held_out_episode_ids = {episode.episode_id for episode in split.held_out_episodes}
    p0_metrics = asdict(comparison.p0)
    learned_metrics = asdict(comparison.learned)
    replay_model_digest = hashlib.sha256(
        json.dumps(
            {
                "schema": "manabot.belief-population-replay-model/v1",
                "population_identity": dataset.identity,
                "split_identity": split.identity,
                "model_identity": trained.model.identity,
                "training_steps": trained.steps,
                "training_p0_nll": trained.p0_nll,
                "training_final_nll": trained.final_nll,
                "held_out_p0": p0_metrics,
                "held_out_learned": learned_metrics,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "proof": "frozen-population-supervised-exact-world-belief",
        "claim_boundary": (
            "one deterministic scripted population; no strength or broader "
            "opponent generalization claim"
        ),
        "population": {
            "episodes": len(dataset.episodes),
            "source_distribution": dataset.source_distribution,
            "sampling": "authoritative-exact-p0-with-replacement",
            "sampling_seed": dataset.sampling_seed,
            "behavior_policy": asdict(dataset.behavior_population),
            "public_commitments": commitment_counts,
            "distinct_sampled_worlds": distinct_sampled_worlds,
            "real_managym_transitions": len(dataset.episodes),
            "distinct_episode_provenance_identities": len(
                {episode.provenance.identity for episode in dataset.episodes}
            ),
        },
        "split": {
            "unit": "deal-episode",
            "training_episodes": len(split.training_episodes),
            "held_out_episodes": len(split.held_out_episodes),
            "episode_identity_overlap": len(
                training_episode_ids & held_out_episode_ids
            ),
            "semantic_history_groups_in_both_arms": len(
                training_history_keys & held_out_history_keys
            ),
        },
        "supervision": {
            "label": "authority-only-materialized-whole-world",
            "loss": "whole-world-negative-log-likelihood",
            "semantic_history_groups": len(histories),
            "repeated_history_groups_with_multiple_world_labels": repeated_truth_groups,
        },
        "model_boundary": {
            "inference_inputs": [
                "candidate_world_counts",
                "exact_log_p0",
                "typed_public_commitment_history",
            ],
            "excluded_inputs": [
                "acting_policy_identity",
                "episode_identity",
                "episode_provenance_identity",
                "supervision_receipt",
                "event_receipt_identity",
                "materialized_hidden_world",
            ],
            "policy_version_is_provenance_only": True,
            "semantic_history_schema_identity": (trained.model.history_schema_identity),
        },
        "training": {
            "fresh_model": True,
            "steps": trained.steps,
            "parameters": sum(
                parameter.numel() for parameter in trained.model.parameters()
            ),
            "initial_joint_nll": trained.initial_nll,
            "p0_joint_nll": trained.p0_nll,
            "final_joint_nll": trained.final_nll,
            "fresh_model_started_at_p0": bool(
                np.isclose(
                    trained.initial_nll,
                    trained.p0_nll,
                    atol=1e-10,
                    rtol=0.0,
                )
            ),
        },
        "held_out": {
            "p0": p0_metrics,
            "learned": learned_metrics,
            "joint_nll_improvement_nats": (
                comparison.p0.joint_nll - comparison.learned.joint_nll
            ),
            "inclusion_brier_improvement": (
                comparison.p0.inclusion_brier - comparison.learned.inclusion_brier
            ),
            "inclusion_ece_improvement": (
                comparison.p0.inclusion_ece - comparison.learned.inclusion_ece
            ),
        },
        "cost": {
            "training_support_sizes": dict(
                sorted(
                    Counter(
                        example.world_space.support_size
                        for example in split.training_examples
                    ).items()
                )
            ),
            "held_out_support_sizes": dict(
                sorted(
                    Counter(
                        example.world_space.support_size
                        for example in split.held_out_examples
                    ).items()
                )
            ),
            "training_candidate_world_rows_per_step": sum(
                example.world_space.support_size for example in split.training_examples
            ),
            "held_out_candidate_world_rows": comparison.candidate_worlds,
            "dataset_seconds": dataset_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
        "model_identity": trained.model.identity,
        "population_identity": dataset.identity,
        "split_identity": split.identity,
        "replay_model_digest": replay_model_digest,
    }


def main(
    *,
    episodes: int = 160,
    held_out_episodes: int = 32,
    steps: int = 16,
    seed: int = 197,
) -> None:
    print(
        json.dumps(
            run_demo(
                episodes=episodes,
                held_out_episodes=held_out_episodes,
                steps=steps,
                seed=seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
