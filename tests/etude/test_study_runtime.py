"""Player-visible Retry, sealed reveal, preview, and exact return proof."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from etude import server, trace as trace_store
from etude.replay_index import CanonicalReplayV1
from etude.server import GameSession, SessionRecord, app
from etude.study_protocol import StudyArtifact
from etude.study_runtime import HistoricalStudyEvidenceRequest


class ExactFixtureEvidenceProvider:
    """Test-only evidence bound dynamically to the selected Game replay."""

    def __init__(self, *, drift_digest: bool = False) -> None:
        self.drift_digest = drift_digest

    def artifact_for(self, request: HistoricalStudyEvidenceRequest) -> StudyArtifact:
        restored = request.restored
        pass_offer = next(
            offer
            for offer in restored.frame.offers
            if offer.verb.value == "pass_priority"
        )
        alternative_id = f"offer-{pass_offer.id}"
        alternative_command = {
            "command_id": f"fixture-{alternative_id}",
            "match_id": restored.frame.match_id,
            "expected_revision": restored.revision,
            "prompt_id": restored.frame.prompt.id,
            "offer_id": pass_offer.id,
            "answers": [],
        }
        asset_pack = restored.frame.asset_pack
        assert asset_pack is not None
        return StudyArtifact.model_validate(
            {
                "version": 1,
                "identity": {
                    "artifact_id": "study-runtime-fixture",
                    "source_replay_id": request.projection.replay_id,
                    "source_replay_sha256": (
                        "0" * 64 if self.drift_digest else request.source_replay_sha256
                    ),
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
                        "version": "managym-test-fixture",
                        "build_sha256": "0" * 64,
                    },
                    "model": {
                        "id": "fixture-policy",
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
                        "id": f"fixture-{restored.ordinal}",
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
                                "command": alternative_command,
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
                                    "expected_match_points": 0.0,
                                }
                            ],
                            "visits": [{"alternative": alternative_id, "visits": 1}],
                            "sampled_world_robustness": [
                                {
                                    "alternative": alternative_id,
                                    "favorable_worlds": 1,
                                    "sampled_worlds": 1,
                                }
                            ],
                            "uncertainty": [
                                {
                                    "alternative": alternative_id,
                                    "standard_error": 0.0,
                                    "method": "fixture",
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


def _completed_session(
    trace_dir: Path,
    *,
    evidence_provider: ExactFixtureEvidenceProvider | None = None,
    allow_fixture: bool = False,
) -> tuple[GameSession, CanonicalReplayV1]:
    session = GameSession(
        trace_dir,
        id_factory=lambda kind: f"study-runtime-{kind}",
        clock=lambda: "2026-07-17T00:00:00+00:00",
        historical_evidence_provider=evidence_provider,
        allow_fixture_study_evidence=allow_fixture,
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
        outcome = session.hero_command(
            {
                "command_id": f"study-source-{index}",
                "match_id": frame["match_id"],
                "expected_revision": frame["revision"],
                "prompt_id": frame["prompt"]["id"],
                "offer_id": frame["offers"][0]["id"],
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
    else:
        pytest.fail("deterministic Study source match did not terminate")
    session.close("game_over")
    assert session.trace is not None
    assert session.trace.canonical_replay is not None
    return session, CanonicalReplayV1.model_validate(session.trace.canonical_replay)


def _install_live_session(
    monkeypatch: pytest.MonkeyPatch,
    trace_dir: Path,
    session: GameSession,
) -> None:
    monkeypatch.setattr(trace_store, "TRACES_DIR", trace_dir)
    server.SESSION_REGISTRY.clear()
    server.SESSION_REGISTRY["study-session"] = SessionRecord(
        session_id="study-session",
        resume_token="study-resume",
        game=session,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
    )


def _first_pass_retry(
    client: TestClient,
    trace_id: str,
) -> tuple[dict, dict, dict]:
    projection = client.get(f"/api/traces/{trace_id}/decisions").json()
    row = next(
        decision
        for decision in projection["decisions"]
        if any(
            offer["verb"] == "pass_priority" for offer in decision["frame"]["offers"]
        )
    )
    restored = client.get(f"/api/traces/{trace_id}/decisions/{row['address']}").json()
    pass_offer = next(
        offer
        for offer in restored["frame"]["offers"]
        if offer["verb"] == "pass_priority"
    )
    command = {
        "command_id": "player-retry",
        "match_id": restored["frame"]["match_id"],
        "expected_revision": restored["revision"],
        "prompt_id": restored["frame"]["prompt"]["id"],
        "offer_id": pass_offer["id"],
        "answers": [],
    }
    response = client.post(
        f"/api/traces/{trace_id}/decisions/{row['address']}/retry",
        json={"command": command},
    )
    assert response.status_code == 200, response.text
    return projection, restored, response.json()


def test_normal_runtime_seals_evidence_and_returns_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session, _ = _completed_session(tmp_path)
    _install_live_session(monkeypatch, tmp_path, session)
    trace_path = tmp_path / f"{session.trace_id}.json"
    before = trace_path.read_bytes()

    with TestClient(app) as client:
        projection, restored, retry = _first_pass_retry(client, session.trace_id)
        sealed = json.dumps({"projection": projection, "retry": retry})
        for forbidden in (
            "policy_mass",
            "search_value",
            "visits",
            "uncertainty",
            "analysis_budget",
            "provenance",
        ):
            assert forbidden not in sealed

        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 409
        assert reveal.json()["detail"] == "study_evidence_unavailable"

        returned = client.post(f"/api/study-attempts/{retry['attempt_id']}/return")
        assert returned.status_code == 200
        assert returned.json() == restored
        assert (
            client.post(f"/api/study-attempts/{retry['attempt_id']}/return").status_code
            == 404
        )

    assert trace_path.read_bytes() == before


def test_injected_exact_fixture_reveals_and_previews_on_fresh_forks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session, _ = _completed_session(
        tmp_path,
        evidence_provider=ExactFixtureEvidenceProvider(),
        allow_fixture=True,
    )
    _install_live_session(monkeypatch, tmp_path, session)
    trace_path = tmp_path / f"{session.trace_id}.json"
    before = trace_path.read_bytes()

    with TestClient(app) as client:
        _, restored, retry = _first_pass_retry(client, session.trace_id)
        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 200, reveal.text
        artifact = reveal.json()["artifact"]
        assert artifact["landmarks"][0]["decision_id"] == restored["address"]

        active = client.post(
            f"/api/traces/{session.trace_id}/decisions/{restored['address']}/retry",
            json={"command": retry["retry"]["command"]},
        )
        assert active.status_code == 409
        assert active.json()["detail"] == "study_attempt_active"

        previews = {}
        for plan in ("played", "policy", "search"):
            response = client.post(
                f"/api/study-attempts/{retry['attempt_id']}/preview",
                json={"plan": plan},
            )
            assert response.status_code == 200, response.text
            previews[plan] = response.json()
            assert response.json()["plan"] == plan
            assert response.json()["return_to"] == retry["return_to"]

        repeated = client.post(
            f"/api/study-attempts/{retry['attempt_id']}/preview",
            json={"plan": "policy"},
        )
        assert repeated.status_code == 200
        assert repeated.json()["projection"] == previews["policy"]["projection"]
        assert repeated.json()["presentation"] == previews["policy"]["presentation"]
        assert previews["played"]["presentation"]

        returned = client.post(f"/api/study-attempts/{retry['attempt_id']}/return")
        assert returned.json() == restored

    assert trace_path.read_bytes() == before


def test_drifted_or_runtime_fixture_evidence_fails_closed_but_keeps_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session, _ = _completed_session(
        tmp_path,
        evidence_provider=ExactFixtureEvidenceProvider(drift_digest=True),
        allow_fixture=True,
    )
    _install_live_session(monkeypatch, tmp_path, session)

    with TestClient(app) as client:
        _, restored, retry = _first_pass_retry(client, session.trace_id)
        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 409
        assert reveal.json()["detail"] == "study_evidence_identity_mismatch"
        returned = client.post(f"/api/study-attempts/{retry['attempt_id']}/return")
        assert returned.json() == restored

    forbidden_dir = tmp_path / "forbidden"
    forbidden_dir.mkdir()
    forbidden, _ = _completed_session(
        forbidden_dir,
        evidence_provider=ExactFixtureEvidenceProvider(),
        allow_fixture=False,
    )
    _install_live_session(monkeypatch, forbidden_dir, forbidden)
    with TestClient(app) as client:
        _, restored, retry = _first_pass_retry(client, forbidden.trace_id)
        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 409
        assert reveal.json()["detail"] == "fixture_evidence_forbidden"
        assert (
            client.post(f"/api/study-attempts/{retry['attempt_id']}/return").json()
            == restored
        )


def test_invalid_or_unstructured_retry_never_creates_an_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session, _ = _completed_session(tmp_path)
    _install_live_session(monkeypatch, tmp_path, session)

    with TestClient(app) as client:
        projection = client.get(f"/api/traces/{session.trace_id}/decisions").json()
        row = projection["decisions"][0]
        restored = client.get(
            f"/api/traces/{session.trace_id}/decisions/{row['address']}"
        ).json()
        play_land = next(
            offer
            for offer in restored["frame"]["offers"]
            if offer["verb"] == "play_land"
        )
        command = {
            "command_id": "unsupported-retry",
            "match_id": restored["frame"]["match_id"],
            "expected_revision": restored["revision"],
            "prompt_id": restored["frame"]["prompt"]["id"],
            "offer_id": play_land["id"],
            "answers": [],
        }
        unsupported = client.post(
            f"/api/traces/{session.trace_id}/decisions/{row['address']}/retry",
            json={"command": command},
        )
        assert unsupported.status_code == 409
        assert unsupported.json()["detail"] == "study_command_not_structured"

        drifted = deepcopy(command)
        drifted["expected_revision"] += 1
        invalid = client.post(
            f"/api/traces/{session.trace_id}/decisions/{row['address']}/retry",
            json={"command": drifted},
        )
        assert invalid.status_code == 409
        assert invalid.json()["detail"] == "study_command_identity_mismatch"

        assert client.post("/api/study-attempts/missing/reveal").status_code == 404
