"""Generate deterministic single-view canonical replay and Study fixtures.

Run from the repository root with:

    uv run --extra dev python scripts/generate_replay_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from etude.replay_index import (
    CanonicalReplayV1,
    ReplayDecisionAddress,
    canonical_projection_sha256,
    project_replay,
)
from etude.server import GameSession

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "protocol" / "fixtures"
FIXED_TIME = "2026-07-16T00:00:00+00:00"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _play_pinned_match() -> CanonicalReplayV1:
    with TemporaryDirectory() as directory:
        session = GameSession(
            Path(directory),
            id_factory=lambda kind: f"pinned-curated-{kind}",
            clock=lambda: FIXED_TIME,
        )
        session.new_game(
            {
                "villain_type": "passive",
                "seed": 7,
                "hero_deck": "ur_lessons",
                "villain_deck": "gw_allies",
                # Surface the first exact decision at presentation cursor zero,
                # then enable authored auto-pass so later automatic semantics
                # are proven between indexed rows rather than hidden in a
                # pre-index prologue.
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
        deliberate = [event for event in session.trace.events if not event.auto]
        assert len(deliberate) == len(session.canonical_decisions)
        assert any(event.auto for event in session.trace.events)
        replay = CanonicalReplayV1.model_validate(session.trace.canonical_replay)
        assert {row.viewer for row in replay.decisions} == {0, 1}
        return replay


def _study_fixture(
    replay: CanonicalReplayV1,
) -> dict:
    projection = project_replay(replay, 0)
    row = projection.decisions[0]
    address = ReplayDecisionAddress.from_decision(projection, row).serialize()
    offers = row.frame.offers[:2]
    alternative_ids = [f"offer-{offer.id}" for offer in offers]
    alternatives = [
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
    probability = 1 / len(alternative_ids)
    evidence = {
        "policy_mass": [
            {"alternative": alternative, "probability": probability}
            for alternative in alternative_ids
        ],
        "search_value": [
            {
                "alternative": alternative,
                "perspective": row.viewer,
                "expected_match_points": 0.0,
            }
            for alternative in alternative_ids
        ],
        "visits": [
            {"alternative": alternative, "visits": 1} for alternative in alternative_ids
        ],
        "sampled_world_robustness": [
            {
                "alternative": alternative,
                "favorable_worlds": 1,
                "sampled_worlds": 1,
            }
            for alternative in alternative_ids
        ],
        "uncertainty": [
            {"alternative": alternative, "standard_error": 0.0, "method": "fixture"}
            for alternative in alternative_ids
        ],
        "provenance": {
            "producer": "canonical-replay-fixture",
            "producer_version": "1",
            "generated_at": FIXED_TIME,
            "evidence_sha256": "0" * 64,
        },
    }
    asset_pack = row.frame.asset_pack
    assert asset_pack is not None
    return {
        "version": 1,
        "identity": {
            "artifact_id": "study-pinned-curated-decision-1",
            "source_replay_id": replay.replay_id,
            "source_replay_sha256": canonical_projection_sha256(projection),
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
                "id": "passive-policy",
                "checkpoint_sha256": "0" * 64,
            },
            "analysis_budget": {
                "id": "fixture-only",
                "max_nodes": 1,
                "sampled_worlds": 1,
                "rollouts_per_world": 1,
            },
            "knowledge_scope": "historical_viewer",
        },
        "landmarks": [
            {
                "id": f"pinned-decision-{row.ordinal}",
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
        ],
    }


def main() -> None:
    first = _play_pinned_match()
    second = _play_pinned_match()
    first_bytes = _json_bytes(first.model_dump(mode="json"))
    second_bytes = _json_bytes(second.model_dump(mode="json"))
    if first_bytes != second_bytes:
        raise RuntimeError("pinned canonical authority generation is not deterministic")

    projections: dict[int, bytes] = {}
    for viewer in (0, 1):
        projection = project_replay(first, viewer)
        payload = _json_bytes(projection.model_dump(mode="json"))
        projections[viewer] = payload
        (FIXTURES / f"canonical-replay-player-{viewer}.json").write_bytes(payload)

    metadata = {
        "version": 1,
        "replay_id": first.replay_id,
        "match_id": first.match_id,
        "decisions": first.metadata(),
    }
    (FIXTURES / "canonical-replay-authority-metadata.json").write_bytes(
        _json_bytes(metadata)
    )
    (FIXTURES / "study-curated-decision.json").write_bytes(
        _json_bytes(_study_fixture(first))
    )


if __name__ == "__main__":
    main()
