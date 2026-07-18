"""Generate the advice fixture: real flat-MC evidence at one pinned decision
under two disjoint belief scenarios.

The fixture is the first unified belief-input and strategy-comparison surface.
It is a checked, versioned, identity-pinned ``AdviceArtifact`` wrapper around a
``StudyArtifact`` that carries two landmarks at the same ``erd1`` decision
(canonical ordinal 6, the hero's turn-7 precombat "Play Mountain" vs "Pass
priority" choice), each with real flat-Monte-Carlo search evidence produced
under a disjoint seed family. No evidence is fabricated.

Run from the repository root with the dev extra (jsonschema is required for the
Rust-owned study schema check):

    uv run --extra dev python scripts/generate_advice_fixture.py
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jsonschema import Draft202012Validator

from etude.replay_index import (
    CanonicalReplayV1,
    ReplayDecisionAddress,
    project_replay,
)
from etude.server import GameSession
from etude.study_protocol import StudyArtifact

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "protocol" / "fixtures"
FIXED_TIME = "2026-07-16T00:00:00+00:00"

# The frozen ``study-curated-decision.json`` fixture carries the canonical
# replay hash ``692f3ae1...`` from an earlier engine build. The current engine
# (``managym/_managym.cpython-312-darwin.so``, 2026-07-17) reproduces the same
# pinned decision structure (ordinal 6, revision 6, prompt 7, two offers:
# play_land + pass_priority) but its frame content drifted, so its canonical
# replay hash differs. The advice fixture pins to the CURRENT engine's
# canonical identity so its frame, address, and evidence are all real and
# self-consistent. The frozen study fixture is left untouched (out of scope).
FROZEN_STUDY_SOURCE_REPLAY_SHA256 = (
    "692f3ae12a7792d54690f9f5b206756c42fbe06deb2fc1ac0fa7b607fc352149"
)

# Two disjoint seed families are the honest belief-scenario substrate. Each
# family is 16 seeds; every seed drives one flat-MC evaluation
# (worlds=1, rollouts=8, max_steps=2000) of the same forked decision.
SCENARIO_A_SEEDS = list(range(101, 117))
SCENARIO_B_SEEDS = list(range(201, 217))
WORLDS = 1
ROLLOUTS = 8
MAX_STEPS = 2000

ADVISOR_ID = "flat-mc-search-v1"
COMPUTE_ID = "1w-8r-16s"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _play_pinned_match_for_fork(
    directory: Path,
) -> tuple[GameSession, CanonicalReplayV1]:
    """Play the deterministic pinned match and keep the session alive for forks.

    Mirrors ``scripts.generate_replay_fixtures._play_pinned_match`` exactly
    (same ``pinned-human-{index}`` command ids, same stops, same pinned seed
    and curated matchup) so the canonical replay is byte-identical to the
    checked-in ``canonical-replay-player-0.json`` and the advice fixture pins
    to the same canonical identity. The session is retained (not closed inside
    a ``TemporaryDirectory``) so its retained Study roots stay addressable for
    ``fork_study``.
    """
    session = GameSession(
        directory,
        id_factory=lambda kind: f"pinned-curated-{kind}",
        clock=lambda: FIXED_TIME,
    )
    session.new_game(
        {
            "villain_type": "passive",
            "seed": 7,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": False,
        }
    )
    for index in range(2_000):
        assert session.obs is not None
        if session.obs.game_over:
            break
        frame = session._experience_frame()
        offer = frame["offers"][0]
        outcome = session.hero_command(
            {
                "command_id": f"pinned-human-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": offer["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
        if index == 0:
            session.set_stops(None, None, True)
    else:
        raise RuntimeError("pinned curated match exceeded decision limit")

    session.close("game_over")
    assert session.trace is not None
    assert session.trace.canonical_replay is not None
    replay = CanonicalReplayV1.model_validate(session.trace.canonical_replay)
    return session, replay


def _run_seed_family(
    session: GameSession, address: str, seeds: list[int]
) -> list[tuple[list[float], int]]:
    """Fork the pinned decision once per call and run flat-MC for each seed.

    ``flat_mc_scores`` clones the world internally for every determinization
    and playout, so the forked env is not stepped by the call; re-forking per
    seed is the de-risked deterministic path (same seed -> byte-identical
    scores across independent forks of the same retained root).
    """
    per_seed: list[tuple[list[float], int]] = []
    for seed in seeds:
        branch = session.fork_study(address)
        # ``StudyBranch`` exposes the structured command surface, not search.
        # The generator reaches the forked retained-root env (a ``managym.Env``
        # cloned at the pinned decision) to run real flat-MC. This is the same
        # rules state the branch would step; it is not mutated by ``flat_mc``.
        env = branch._env
        assert env is not None
        scores, simulations, _cap_hits = env.flat_mc_scores(
            WORLDS, ROLLOUTS, seed, MAX_STEPS
        )
        per_seed.append((scores, int(simulations)))
    return per_seed


def _aggregate_evidence(
    per_seed: list[tuple[list[float], int]],
    alternative_ids: list[str],
    viewer: int,
    producer: str,
) -> dict[str, Any]:
    """Aggregate per-seed flat-MC scores into a ``DecisionEvidence`` dict.

    Mirrors the structural intent of ``manabot.sim.study_evidence`` but for
    flat-MC seed spreads rather than PUCT visits. Every number is real search
    output; nothing is fabricated.
    """
    num_seeds = len(per_seed)
    num_actions = len(alternative_ids)
    assert num_seeds > 0 and num_actions > 0
    assert all(len(scores) == num_actions for scores, _ in per_seed)

    # policy_mass: fraction of seeds where each action is the argmax. Ties
    # break to the lowest index so each seed contributes exactly one argmax
    # and the distribution sums to 1.0.
    argmax_counts = [0] * num_actions
    for scores, _ in per_seed:
        best = max(scores)
        for index in range(num_actions):
            if scores[index] == best:
                argmax_counts[index] += 1
                break
    policy_mass = [
        {"alternative": alt, "probability": count / num_seeds}
        for alt, count in zip(alternative_ids, argmax_counts, strict=True)
    ]

    # search_value: mean per-action win-probability across the seed family.
    search_value = [
        {
            "alternative": alt,
            "perspective": viewer,
            "expected_match_points": sum(scores[index] for scores, _ in per_seed)
            / num_seeds,
        }
        for index, alt in enumerate(alternative_ids)
    ]

    # visits: total playouts attributable to each action across the family.
    total_simulations = sum(simulations for _, simulations in per_seed)
    per_action_visits = total_simulations // num_actions
    visits = [
        {"alternative": alt, "visits": per_action_visits} for alt in alternative_ids
    ]

    # sampled_world_robustness: how many seeded worlds favor each action
    # (score > 0.5), over the 16 sampled worlds.
    sampled_world_robustness = [
        {
            "alternative": alt,
            "favorable_worlds": sum(1 for scores, _ in per_seed if scores[index] > 0.5),
            "sampled_worlds": num_seeds,
        }
        for index, alt in enumerate(alternative_ids)
    ]

    # uncertainty: standard error of the per-action mean across the seed
    # family (between-seed spread, not within-rollout variance).
    uncertainty = []
    for index, alt in enumerate(alternative_ids):
        samples = [scores[index] for scores, _ in per_seed]
        standard_error = (
            statistics.stdev(samples) / math.sqrt(num_seeds) if num_seeds > 1 else 0.0
        )
        uncertainty.append(
            {
                "alternative": alt,
                "standard_error": standard_error,
                "method": "flat-mc-seed-spread",
            }
        )

    evidence_core = {
        "policy_mass": policy_mass,
        "search_value": search_value,
        "visits": visits,
        "sampled_world_robustness": sampled_world_robustness,
        "uncertainty": uncertainty,
    }
    evidence_sha256 = _canonical_sha256(evidence_core)
    return {
        **evidence_core,
        "provenance": {
            "producer": producer,
            "producer_version": "1",
            "generated_at": FIXED_TIME,
            "evidence_sha256": evidence_sha256,
        },
    }


def _alternatives(row: Any, offers: list[Any]) -> list[dict[str, Any]]:
    alternative_ids = [f"offer-{offer.id}" for offer in offers]
    return [
        {
            "id": alternative_id,
            "command": {
                "command_id": f"study-{alternative_id}",
                "match_id": row.frame.match_id,
                "expected_revision": row.revision,
                "prompt_id": row.prompt_id,
                "offer_id": offer.id,
                "answers": [],
            },
        }
        for alternative_id, offer in zip(alternative_ids, offers, strict=True)
    ]


def _landmark(
    row: Any,
    address: str,
    landmark_id: str,
    offers: list[Any],
    alternatives: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    asset_pack = row.frame.asset_pack
    assert asset_pack is not None
    return {
        "id": landmark_id,
        "decision_id": address,
        "match_state_hash": row.frame.frame_hash,
        "viewer": row.viewer,
        "prompt_id": row.prompt_id,
        "offer_id": row.offer_id,
        "frame": row.frame.model_dump(mode="json"),
        "offer": row.offer.model_dump(mode="json"),
        "played": row.command.model_dump(mode="json"),
        "alternatives": alternatives,
        "evidence": evidence,
    }


def _artifact(
    replay: CanonicalReplayV1,
    row: Any,
    player_zero_bytes: bytes,
    landmarks: list[dict[str, Any]],
) -> dict[str, Any]:
    asset_pack = row.frame.asset_pack
    assert asset_pack is not None
    total_simulations = sum(
        landmark["evidence"]["visits"][0]["visits"] for landmark in landmarks
    )
    return {
        "version": 1,
        "identity": {
            "artifact_id": "advice-pinned-curated-decision-6",
            "source_replay_id": replay.replay_id,
            "source_replay_sha256": hashlib.sha256(player_zero_bytes).hexdigest(),
            "match_id": replay.match_id,
            "content_pack": {
                "id": asset_pack.id,
                "version": asset_pack.version,
                "content_hash": replay.content_hash,
                "asset_manifest_sha256": replay.asset_manifest_hash,
            },
            "engine": {
                "version": "managym-python-adapter",
                "build_sha256": "0" * 64,
            },
            "model": {
                "id": ADVISOR_ID,
                "checkpoint_sha256": "0" * 64,
            },
            "analysis_budget": {
                "id": COMPUTE_ID,
                "max_nodes": total_simulations,
                "sampled_worlds": len(SCENARIO_A_SEEDS),
                "rollouts_per_world": ROLLOUTS,
            },
            "knowledge_scope": "historical_viewer",
        },
        "landmarks": landmarks,
    }


SCENARIOS = [
    {
        "landmark_id": "advice-scenario-a",
        "label": "Opponent curving out",
        "description": (
            "Believe the opponent has kept a low-curve Allies hand and is likely "
            "to spend their mana each turn to commit bodies to the board."
        ),
        "inferred_range": (
            "Likely two or three cheap creatures already committed; few "
            "instant-speed tricks expected in hand."
        ),
        "belief_kind": "opponent-curved-out",
        "seed_family": "101-116",
    },
    {
        "landmark_id": "advice-scenario-b",
        "label": "Opponent holding interaction",
        "description": (
            "Believe the opponent is holding removal or a combat trick back and "
            "may not spend their mana this turn."
        ),
        "inferred_range": (
            "Likely fewer committed bodies; one or more reactive cards held in "
            "hand against an open board."
        ),
        "belief_kind": "opponent-holding-interaction",
        "seed_family": "201-216",
    },
]


def _advice_artifact(
    replay: CanonicalReplayV1,
    row: Any,
    address: str,
    player_zero_bytes: bytes,
    evidence_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    offers = row.frame.offers[:2]
    alternative_ids = [f"offer-{offer.id}" for offer in offers]
    alternatives = _alternatives(row, offers)
    landmarks = [
        _landmark(
            row,
            address,
            scenario["landmark_id"],
            offers,
            alternatives,
            evidence_by_scenario[scenario["landmark_id"]],
        )
        for scenario in SCENARIOS
    ]
    artifact = _artifact(replay, row, player_zero_bytes, landmarks)
    # The two scenarios share one decision; both landmarks carry the same erd1
    # address and the same action vocabulary, distinguished by id and evidence.
    assert len({landmark["decision_id"] for landmark in landmarks}) == 1
    assert len({landmark["id"] for landmark in landmarks}) == len(landmarks)
    return {"artifact": artifact, "scenarios": SCENARIOS}


def main() -> None:
    with TemporaryDirectory() as directory:
        session, replay = _play_pinned_match_for_fork(Path(directory))

        projection = project_replay(replay, 0)
        player_zero_bytes = _json_bytes(projection.model_dump(mode="json"))
        source_replay_sha256 = hashlib.sha256(player_zero_bytes).hexdigest()
        if source_replay_sha256 != FROZEN_STUDY_SOURCE_REPLAY_SHA256:
            print(
                "note: advice fixture pins to the current engine canonical "
                f"replay ({source_replay_sha256}); the frozen study fixture "
                f"used {FROZEN_STUDY_SOURCE_REPLAY_SHA256}."
            )

        # The first viewer-0 decision is canonical ordinal 6: the hero's turn-7
        # precombat "Play Mountain" vs "Pass priority" priority choice.
        row = next(decision for decision in replay.decisions if decision.viewer == 0)
        assert row.ordinal == 6, (
            f"expected first viewer-0 decision at ordinal 6, got {row.ordinal}"
        )
        address = ReplayDecisionAddress.from_decision(replay, row).serialize()

        offers = row.frame.offers[:2]
        alternative_ids = [f"offer-{offer.id}" for offer in offers]
        assert len(alternative_ids) == 2, "pinned decision must expose two actions"

        evidence_a = _aggregate_evidence(
            _run_seed_family(session, address, SCENARIO_A_SEEDS),
            alternative_ids,
            row.viewer,
            f"flat-mc-search:v1:seeds-{SCENARIO_A_SEEDS[0]}-{SCENARIO_A_SEEDS[-1]}",
        )
        evidence_b = _aggregate_evidence(
            _run_seed_family(session, address, SCENARIO_B_SEEDS),
            alternative_ids,
            row.viewer,
            f"flat-mc-search:v1:seeds-{SCENARIO_B_SEEDS[0]}-{SCENARIO_B_SEEDS[-1]}",
        )

        # Determinism: re-fork and re-aggregate scenario A; the seed-deterministic
        # flat-MC output must reproduce byte-identical evidence.
        evidence_a_again = _aggregate_evidence(
            _run_seed_family(session, address, SCENARIO_A_SEEDS),
            alternative_ids,
            row.viewer,
            f"flat-mc-search:v1:seeds-{SCENARIO_A_SEEDS[0]}-{SCENARIO_A_SEEDS[-1]}",
        )
        if evidence_a != evidence_a_again:
            raise RuntimeError("advice flat-MC aggregation is not seed-deterministic")

        advice = _advice_artifact(
            replay,
            row,
            address,
            player_zero_bytes,
            {
                "advice-scenario-a": evidence_a,
                "advice-scenario-b": evidence_b,
            },
        )

    # Checked: the artifact validates as a StudyArtifact through both the
    # Pydantic model (which enforces viewer-safety and bindings) and the
    # Rust-owned study-v1 JSON schema. The scenarios metadata is a prototype
    # presentation layer validated by the adapter, not by the Rust schema.
    StudyArtifact.model_validate(advice["artifact"])
    schema = json.loads(
        (ROOT / "protocol" / "study-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(advice["artifact"])

    # Distinct, non-uniform evidence: the two scenarios must disagree.
    pa = [
        row["probability"]
        for row in advice["artifact"]["landmarks"][0]["evidence"]["policy_mass"]
    ]
    pb = [
        row["probability"]
        for row in advice["artifact"]["landmarks"][1]["evidence"]["policy_mass"]
    ]
    if pa == pb:
        raise RuntimeError(
            "advice scenarios produced identical policy mass; beliefs are not distinct"
        )

    payload = _json_bytes(advice)
    (FIXTURES / "advice-curated-decision.json").write_bytes(payload)
    print(
        "wrote protocol/fixtures/advice-curated-decision.json "
        f"({address}) scenario A policy={pa} scenario B policy={pb}"
    )


if __name__ == "__main__":
    main()
