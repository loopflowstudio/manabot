"""Build viewer-safe Study v1 evidence from replayed PUCT decisions.

The replay audit owns authority-only seeds and sampled worlds. This module
projects one evidence-complete decision into the closed Study contract without
copying any of that private authority material into the player-facing artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime
import math
from typing import Any, Mapping

import numpy as np

from etude.experience_protocol import Command, ExperienceFrame, InteractionOffer
from etude.replay_index import ReplayDecisionAddress, decision_payload_sha256
from etude.study_protocol import StudyArtifact
from manabot.sim.teacher1_evidence import canonical_sha256


def select_evidence_complete_decision(
    audit: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Select the first multi-offer root with sampled support for every offer."""

    for game in sorted(audit["games"], key=lambda item: int(item["game_index"])):
        game_index = int(game["game_index"])
        for decision_index, decision in enumerate(game["decisions"]):
            visits = np.asarray(decision["search"]["visit_counts"], dtype=np.int64)
            if len(decision["frame"]["offers"]) >= 2 and np.all(visits > 0):
                return game_index, decision_index, decision
    raise ValueError("trajectory audit has no evidence-complete Study decision")


def build_study_artifact(
    audit: dict[str, Any],
    *,
    policy_mass_by_offer: Mapping[int, float],
    source_replay_sha256: str,
    checkpoint_sha256: str,
    engine_build_sha256: str,
    content_pack_id: str,
    content_pack_version: str,
    model_id: str,
    producer_version: str,
    generated_at: str | None = None,
) -> StudyArtifact:
    """Project one historical audit root into the validated Study v1 schema."""

    game_index, decision_index, decision = select_evidence_complete_decision(audit)
    frame = decision["frame"]
    search = decision["search"]
    offers = frame["offers"]
    offer_ids = [int(offer["id"]) for offer in offers]
    if set(policy_mass_by_offer) != set(offer_ids):
        raise ValueError("policy mass must cover every legal offer exactly")
    probabilities = np.asarray(
        [policy_mass_by_offer[offer_id] for offer_id in offer_ids],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or float(probabilities.sum()) <= 0.0
    ):
        raise ValueError("policy mass must be finite, non-negative, and nonzero")
    probabilities /= probabilities.sum()

    visits = np.asarray(search["visit_counts"], dtype=np.int64)
    q_values = np.asarray(search["q_values"], dtype=np.float64)
    world_visits = np.asarray(search["world_visit_counts"], dtype=np.int64)
    world_q = np.asarray(search["world_q_values"], dtype=np.float64)
    expected_shape = (int(search["worlds"]), len(offers))
    if (
        visits.shape != (len(offers),)
        or q_values.shape != (len(offers),)
        or world_visits.shape != expected_shape
        or world_q.shape != expected_shape
    ):
        raise ValueError("search evidence shapes do not match the legal offers")

    alternative_ids = [f"offer-{offer_id}" for offer_id in offer_ids]
    alternatives = []
    for alternative_id, offer_id in zip(alternative_ids, offer_ids):
        alternatives.append(
            {
                "id": alternative_id,
                "command": {
                    **decision["command"],
                    "command_id": (
                        f"{frame['match_id']}:{frame['revision']}:analysis:{offer_id}"
                    ),
                    "offer_id": offer_id,
                },
            }
        )

    visited_q = np.where(world_visits > 0, world_q, -np.inf)
    world_best = visited_q.max(axis=1)
    robustness = []
    uncertainty = []
    for action_index, alternative_id in enumerate(alternative_ids):
        covered = world_visits[:, action_index] > 0
        covered_count = int(np.count_nonzero(covered))
        if covered_count == 0:
            raise ValueError("selected Study root has an offer with no world coverage")
        favorable = int(
            np.count_nonzero(
                covered
                & np.isclose(world_q[:, action_index], world_best, rtol=0.0, atol=1e-12)
            )
        )
        samples = world_q[covered, action_index]
        standard_error = (
            float(np.std(samples, ddof=1) / math.sqrt(covered_count))
            if covered_count > 1
            else 0.0
        )
        robustness.append(
            {
                "alternative": alternative_id,
                "favorable_worlds": favorable,
                "sampled_worlds": covered_count,
            }
        )
        uncertainty.append(
            {
                "alternative": alternative_id,
                "standard_error": standard_error,
                "method": "between-determinized-world-q-standard-error/v1",
            }
        )

    evidence_core = {
        "policy_mass": [
            {"alternative": alternative_id, "probability": float(probability)}
            for alternative_id, probability in zip(alternative_ids, probabilities)
        ],
        "search_value": [
            {
                "alternative": alternative_id,
                "perspective": int(decision["actor"]),
                "expected_match_points": float(value),
            }
            for alternative_id, value in zip(alternative_ids, q_values)
        ],
        "visits": [
            {"alternative": alternative_id, "visits": int(value)}
            for alternative_id, value in zip(alternative_ids, visits)
        ],
        "sampled_world_robustness": robustness,
        "uncertainty": uncertainty,
    }
    generated = generated_at or datetime.now(UTC).isoformat()
    evidence = {
        **evidence_core,
        "provenance": {
            "producer": "manabot.visit-iteration-study-export",
            "producer_version": producer_version,
            "generated_at": generated,
            "evidence_sha256": canonical_sha256(evidence_core),
        },
    }
    selected_offer_id = int(decision["command"]["offer_id"])
    selected_offer = next(
        offer for offer in offers if int(offer["id"]) == selected_offer_id
    )
    simulations = int(search["simulations"])
    worlds = int(search["worlds"])
    artifact_key = canonical_sha256(
        {
            "source_replay_sha256": source_replay_sha256,
            "game_index": game_index,
            "decision_index": decision_index,
            "checkpoint_sha256": checkpoint_sha256,
        }
    )[:16]
    source_replay_id = "int-4-trajectory-audit-v1"
    replay_ordinal = decision_index + sum(
        len(game["decisions"])
        for game in sorted(audit["games"], key=lambda item: int(item["game_index"]))
        if int(game["game_index"]) < game_index
    )
    presentation_cursor = 0
    decision_id = ReplayDecisionAddress(
        version=1,
        replay_id=source_replay_id,
        match_id=str(frame["match_id"]),
        ordinal=replay_ordinal,
        viewer=int(decision["actor"]),
        revision=int(frame["revision"]),
        prompt_id=int(frame["prompt"]["id"]),
        offer_id=selected_offer_id,
        command_id=str(decision["command"]["command_id"]),
        presentation_cursor=presentation_cursor,
        decision_sha256=decision_payload_sha256(
            ExperienceFrame.model_validate(frame),
            InteractionOffer.model_validate(selected_offer),
            Command.model_validate(decision["command"]),
            presentation_cursor,
        ),
    ).serialize()
    payload = {
        "version": 1,
        "identity": {
            "artifact_id": f"int-4-study-{artifact_key}",
            "source_replay_id": source_replay_id,
            "source_replay_sha256": source_replay_sha256,
            "match_id": frame["match_id"],
            "content_pack": {
                "id": content_pack_id,
                "version": content_pack_version,
                "content_hash": frame["content_hash"],
                "asset_manifest_sha256": frame["asset_manifest_hash"],
            },
            "engine": {
                "version": "managym-python-adapter",
                "build_sha256": engine_build_sha256,
            },
            "model": {"id": model_id, "checkpoint_sha256": checkpoint_sha256},
            "analysis_budget": {
                "id": f"t1-{simulations}-w{worlds}",
                "max_nodes": int(search["tree_nodes"]),
                "sampled_worlds": worlds,
                "rollouts_per_world": simulations // worlds,
            },
            "knowledge_scope": "historical_viewer",
        },
        "landmarks": [
            {
                "id": f"game-{game_index}-decision-{decision_index}",
                "decision_id": decision_id,
                "match_state_hash": frame["frame_hash"],
                "viewer": int(decision["actor"]),
                "prompt_id": int(frame["prompt"]["id"]),
                "offer_id": selected_offer_id,
                "frame": frame,
                "offer": selected_offer,
                "played": decision["command"],
                "alternatives": alternatives,
                "evidence": evidence,
            }
        ],
    }
    return StudyArtifact.model_validate(payload)
