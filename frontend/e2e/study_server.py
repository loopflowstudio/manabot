"""Constructor-injected Study evidence for the real-stack browser proof only."""

from __future__ import annotations

from etude import server
from etude.server import GameSession
from etude.study_protocol import StudyArtifact
from etude.study_runtime import HistoricalStudyEvidenceRequest


class ExactFixtureEvidenceProvider:
    """Bind fixture-labelled evidence to the exact selected Game decision."""

    def artifact_for(self, request: HistoricalStudyEvidenceRequest) -> StudyArtifact:
        restored = request.restored
        pass_offer = next(
            offer
            for offer in restored.frame.offers
            if offer.verb.value == "pass_priority"
        )
        alternative_id = f"offer-{pass_offer.id}"
        asset_pack = restored.frame.asset_pack
        assert asset_pack is not None
        assert restored.frame.prompt is not None
        return StudyArtifact.model_validate(
            {
                "version": 1,
                "identity": {
                    "artifact_id": "study-browser-fixture",
                    "source_replay_id": request.projection.replay_id,
                    "source_replay_sha256": request.source_replay_sha256,
                    "match_id": request.projection.match_id,
                    "content_pack": {
                        "id": asset_pack.id,
                        "version": asset_pack.version,
                        "content_hash": request.projection.content_hash,
                        "asset_manifest_sha256": (
                            request.projection.asset_manifest_hash
                        ),
                    },
                    "engine": {
                        "version": "managym-browser-fixture",
                        "build_sha256": "0" * 64,
                    },
                    "model": {
                        "id": "browser-fixture-policy",
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
                        "id": f"browser-fixture-{restored.ordinal}",
                        "decision_id": request.address,
                        "match_state_hash": restored.frame.frame_hash,
                        "viewer": restored.viewer,
                        "prompt_id": restored.frame.prompt.id,
                        "offer_id": restored.offer.id,
                        "frame": restored.frame.model_dump(mode="json"),
                        "offer": restored.offer.model_dump(mode="json"),
                        "played": restored.command.model_dump(mode="json"),
                        "alternatives": [
                            {
                                "id": alternative_id,
                                "command": {
                                    "command_id": f"browser-{alternative_id}",
                                    "match_id": restored.frame.match_id,
                                    "expected_revision": restored.revision,
                                    "prompt_id": restored.frame.prompt.id,
                                    "offer_id": pass_offer.id,
                                    "answers": [],
                                },
                            }
                        ],
                        "evidence": {
                            "policy_mass": [
                                {
                                    "alternative": alternative_id,
                                    "probability": 1.0,
                                }
                            ],
                            "search_value": [
                                {
                                    "alternative": alternative_id,
                                    "perspective": restored.viewer,
                                    "expected_match_points": 0.25,
                                }
                            ],
                            "visits": [{"alternative": alternative_id, "visits": 64}],
                            "sampled_world_robustness": [
                                {
                                    "alternative": alternative_id,
                                    "favorable_worlds": 5,
                                    "sampled_worlds": 8,
                                }
                            ],
                            "uncertainty": [
                                {
                                    "alternative": alternative_id,
                                    "standard_error": 0.125,
                                    "method": "browser-fixture",
                                }
                            ],
                            "provenance": {
                                "producer": "canonical-replay-fixture",
                                "producer_version": "1",
                                "generated_at": "2026-07-17T00:00:00+00:00",
                                "evidence_sha256": "0" * 64,
                            },
                        },
                    }
                ],
            }
        )


def _new_fixture_session() -> GameSession:
    return GameSession(
        historical_evidence_provider=ExactFixtureEvidenceProvider(),
        allow_fixture_study_evidence=True,
    )


server._new_game_session = _new_fixture_session
app = server.app
