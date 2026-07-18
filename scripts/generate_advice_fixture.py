"""Generate conditional strategy advice at one pinned replay decision.

The fixture is the first unified belief-input and strategy-comparison surface.
It is a checked, versioned, identity-pinned ``AdviceArtifact`` wrapper around a
``StudyArtifact`` that carries two landmarks at the same ``erd1`` decision
(canonical ordinal 6, the hero's turn-7 precombat "Play Mountain" vs "Pass
priority" choice). One conditional determinized-PUCT run holds the root,
advisor, compute budget, action vocabulary, and paired seed plan fixed while it
searches two complementary viewer-safe beliefs over exact compatible hidden
hands. The displayed delta therefore comes from belief conditioning, not from
random-seed drift. No evidence is fabricated.

Run from the repository root with the dev extra (jsonschema is required for the
Rust-owned study schema check):

    uv run --extra dev python scripts/generate_advice_fixture.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jsonschema import Draft202012Validator

from etude.advice import AdviceArtifact, int13_condition_to_decision_evidence
from etude.replay_index import (
    CanonicalReplayV1,
    ReplayDecisionAddress,
    project_replay,
)
from etude.server import GameSession
from etude.study_protocol import StudyArtifact
from manabot.sim.conditional_search import (
    ConditionalStrategyResult,
    HasCard,
    NotQuery,
    ScenarioWorldSpace,
    WorldSpec,
    canonical_result_json,
    conditional_determinized_puct,
    make_prior,
    make_query_plan,
)
from manabot.sim.search_branch import REFERENCE_BRANCH_DRIVER_ID

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

# One paired conditional-search plan is shared by both belief scenarios.
# The world space contains four exact ten-card hands, preserving the public
# opponent hand count at the pinned decision. The two worlds with Allies at
# Last are interaction-heavy; its complement is creature-dense. PUCT receives
# the same root, seed, simulations, and action vocabulary for every condition.
SEED = 197
SIMULATIONS = 16
WORLDS = 4
MAX_STEPS = 200
C_PUCT = 1.5
INTERACTION_QUERY = HasCard("Allies at Last")
SECONDARY_QUERY = HasCard("Fancy Footwork")
INTERACTION_CONDITION_ID = INTERACTION_QUERY.query_id
CURVE_CONDITION_ID = NotQuery(INTERACTION_QUERY).query_id
SEED_PLAN = f"paired-seed-{SEED}"

ADVISOR_ID = "conditional-determinized-puct-v1"
COMPUTE_ID = f"2w-{SIMULATIONS}s-{SEED_PLAN}"
PRODUCER = f"conditional-determinized-puct:v1:{SEED_PLAN}"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


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


def _world_space() -> ScenarioWorldSpace:
    """Four exact viewer-compatible hands for two complementary beliefs."""
    worlds = (
        WorldSpec(
            0,
            "curve-a",
            0.25,
            (
                "Forest",
                "Forest",
                "Plains",
                "Plains",
                "Water Tribe Rallier",
                "Invasion Reinforcements",
                "Compassionate Healer",
                "Earth Kingdom Jailer",
                "South Pole Voyager",
                "Badgermole Cub",
            ),
        ),
        WorldSpec(
            1,
            "curve-b",
            0.25,
            (
                "Forest",
                "Forest",
                "Forest",
                "Plains",
                "Plains",
                "Water Tribe Rallier",
                "White Lotus Reinforcements",
                "Earth King's Lieutenant",
                "Kyoshi Warriors",
                "Badgermole Cub",
            ),
        ),
        WorldSpec(
            2,
            "interaction-a",
            0.25,
            (
                "Forest",
                "Forest",
                "Plains",
                "Plains",
                "Allies at Last",
                "Fancy Footwork",
                "Fancy Footwork",
                "Yip Yip!",
                "Earth Kingdom Jailer",
                "Suki, Kyoshi Warrior",
            ),
        ),
        WorldSpec(
            3,
            "interaction-b",
            0.25,
            (
                "Forest",
                "Forest",
                "Forest",
                "Plains",
                "Plains",
                "Allies at Last",
                "Allies at Last",
                "Yip Yip!",
                "South Pole Voyager",
                "Kyoshi Warriors",
            ),
        ),
    )
    return ScenarioWorldSpace(
        space_id="gam-6-curated-beliefs-v1",
        viewer=0,
        worlds=worlds,
        opponent_seat=1,
    )


def _run_conditional_search(
    session: GameSession,
    address: str,
    alternative_ids: list[str],
    expected_action_vocabulary: list[tuple[int, int, str]],
) -> ConditionalStrategyResult:
    """Search both beliefs once with one root, budget, and paired seed plan."""
    branch = session.fork_study(address)
    root = branch._env
    assert root is not None
    world_space = _world_space()

    # Every materialized world must preserve the viewer's complete semantic
    # observation and root action vocabulary. The exact ten-card hands differ
    # only behind the opponent-private boundary.
    root_observation = root.observation_for_player(world_space.viewer).toJSON()
    root_action_count = int(root.action_count())
    assert root_action_count == len(alternative_ids)
    search_offers = json.loads(root.search_context_json(False))["offers"]["offers"]
    search_vocabulary = [
        (int(offer["id"]), int(offer["actor"]), str(offer["verb"]))
        for offer in search_offers
    ]
    if search_vocabulary != expected_action_vocabulary:
        raise RuntimeError(
            "conditional search action order does not match the shared offers"
        )
    for world_index in range(len(world_space.worlds)):
        world = root.clone_env()
        world_space.configure(world, world_index=world_index)
        if (
            world.observation_for_player(world_space.viewer).toJSON()
            != root_observation
        ):
            raise RuntimeError(
                f"belief world {world_index} changed the viewer observation"
            )
        if int(world.action_count()) != root_action_count:
            raise RuntimeError(f"belief world {world_index} changed the action space")

    result = conditional_determinized_puct(
        root,
        prior=make_prior(world_space, viewer=world_space.viewer),
        query_plan=make_query_plan(has=INTERACTION_QUERY, q=SECONDARY_QUERY),
        simulations=SIMULATIONS,
        worlds=WORLDS,
        seed=SEED,
        c_puct=C_PUCT,
        max_steps=MAX_STEPS,
        branch_driver_id=REFERENCE_BRANCH_DRIVER_ID,
        branch_audit=True,
        branch_match_id=f"gam-6-advice-{SEED}",
    )
    if result.action_count != len(alternative_ids):
        raise RuntimeError("conditional search action vocabulary drifted")
    if result.action_labels != tuple(str(offer["label"]) for offer in search_offers):
        raise RuntimeError(
            "conditional search result labels drifted from its root action order"
        )
    return result


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
                "max_nodes": SIMULATIONS,
                "sampled_worlds": 2,
                "rollouts_per_world": SIMULATIONS // 2,
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
            "Believe the opponent kept a creature-dense Allies hand and is likely "
            "to spend its mana developing the board."
        ),
        "inferred_range": (
            "Weighted toward cheap Allies and follow-up creatures, with no "
            "Allies at Last in the conditioned worlds."
        ),
        "belief_kind": "opponent-curved-out",
        "condition_id": CURVE_CONDITION_ID,
        "seed_plan": SEED_PLAN,
    },
    {
        "landmark_id": "advice-scenario-b",
        "label": "Opponent holding interaction",
        "description": (
            "Believe the opponent kept an interaction-heavy hand and may hold "
            "mana for tricks instead of developing every turn."
        ),
        "inferred_range": (
            "Weighted toward Allies at Last and other instant-speed tricks, "
            "with fewer creature slots in the conditioned worlds."
        ),
        "belief_kind": "opponent-holding-interaction",
        "condition_id": INTERACTION_CONDITION_ID,
        "seed_plan": SEED_PLAN,
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
        action_vocabulary = [(offer.id, offer.actor, offer.verb) for offer in offers]
        assert len(alternative_ids) == 2, "pinned decision must expose two actions"

        result = _run_conditional_search(
            session, address, alternative_ids, action_vocabulary
        )
        repeated = _run_conditional_search(
            session, address, alternative_ids, action_vocabulary
        )
        if canonical_result_json(result) != canonical_result_json(repeated):
            raise RuntimeError("advice conditional search is not deterministic")

        evidence_a = int13_condition_to_decision_evidence(
            result.condition_by_id[CURVE_CONDITION_ID],
            alternative_ids,
            row.viewer,
            PRODUCER,
            FIXED_TIME,
        )
        evidence_b = int13_condition_to_decision_evidence(
            result.condition_by_id[INTERACTION_CONDITION_ID],
            alternative_ids,
            row.viewer,
            PRODUCER,
            FIXED_TIME,
        )

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
    AdviceArtifact.model_validate(advice)
    StudyArtifact.model_validate(advice["artifact"])
    schema = json.loads(
        (ROOT / "protocol" / "study-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(advice["artifact"])

    # Distinct, non-uniform evidence: with every search identity held fixed,
    # only the conditioned world subset differs between the two landmarks.
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
    if all(len(set(policy)) == 1 for policy in (pa, pb)):
        raise RuntimeError("advice policies must include a non-uniform scenario")
    producers = {
        landmark["evidence"]["provenance"]["producer"]
        for landmark in advice["artifact"]["landmarks"]
    }
    if producers != {PRODUCER}:
        raise RuntimeError("advice scenarios did not share one producer identity")

    payload = _json_bytes(advice)
    (FIXTURES / "advice-curated-decision.json").write_bytes(payload)
    print(
        "wrote protocol/fixtures/advice-curated-decision.json "
        f"({address}) scenario A policy={pa} scenario B policy={pb}"
    )


if __name__ == "__main__":
    main()
