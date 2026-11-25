"""Player-visible Retry, sealed reveal, preview, and exact return proof."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from managym.academy import server, trace as trace_store
from managym.academy.replay_index import CanonicalReplayV1
from managym.academy.server import GameSession, SessionRecord, app
from managym.academy.study_protocol import StudyArtifact
from managym.academy.study_runtime import HistoricalStudyEvidenceRequest
from managym.experience import ExperienceFrame, InteractionOffer


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


def _source_offer(frame: ExperienceFrame) -> InteractionOffer:
    """A deterministic test controller finishes through ordinary legal offers."""
    if frame.action_space == "PRIORITY":
        creatures = {
            card.name for card in frame.projection.agent.hand if card.types.is_creature
        }
        for offer in frame.offers:
            if offer.action_type == "PRIORITY_PLAY_LAND":
                return offer
        for offer in frame.offers:
            if offer.action_type == "PRIORITY_CAST_SPELL" and any(
                subject.object.label in creatures for subject in offer.subjects
            ):
                return offer
        return next(
            offer for offer in frame.offers if offer.verb.value == "pass_priority"
        )
    return frame.offers[0]


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
    session.attempt_owner = server.participant_id("study-resume")
    # Seeded early creatures and configured priority automation keep complete
    # match setup economical; every non-priority decision still uses Commands.
    initial = session.new_game(
        {
            "villain_type": "passive",
            "seed": 9,
            "hero_deck": "ur_lessons",
            "villain_deck": "gw_allies",
            "auto_pass": True,
        }
    )
    frame = ExperienceFrame.model_validate(initial["frame"])
    for index in range(2_000):
        assert session.obs is not None
        if session.obs.game_over:
            break
        offer = _source_offer(frame)
        outcome = session.hero_command(
            {
                "command_id": f"study-source-{index}",
                "match_id": frame.match_id,
                "expected_revision": frame.revision,
                "prompt_id": frame.prompt.id,
                "offer_id": offer.id,
                "answers": [],
            }
        )
        assert outcome["status"] == "accepted"
        frame = ExperienceFrame.model_validate(outcome["update"]["frame"])
    else:
        pytest.fail("deterministic Study source match did not terminate")
    session.close("game_over")
    assert session.trace is not None
    assert session.trace.canonical_replay is not None
    return session, CanonicalReplayV1.model_validate(session.trace.canonical_replay)


@pytest.fixture(scope="module")
def completed_session(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, GameSession, CanonicalReplayV1]:
    trace_dir = tmp_path_factory.mktemp("study-runtime")
    session, replay = _completed_session(trace_dir)
    return trace_dir, session, replay


def _install_live_session(
    monkeypatch: pytest.MonkeyPatch,
    trace_dir: Path,
    session: GameSession,
    *,
    evidence_provider: ExactFixtureEvidenceProvider | None = None,
    allow_fixture: bool = False,
) -> None:
    from managym.academy.study_runtime import UnavailableHistoricalStudyEvidenceProvider

    monkeypatch.setattr(trace_store, "TRACES_DIR", trace_dir)
    monkeypatch.setattr(
        session,
        "_historical_evidence_provider",
        evidence_provider or UnavailableHistoricalStudyEvidenceProvider(),
    )
    monkeypatch.setattr(session, "_allow_fixture_study_evidence", allow_fixture)
    session._study_attempts.clear()
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
    completed_session: tuple[Path, GameSession, CanonicalReplayV1],
) -> None:
    trace_dir, session, _ = completed_session
    _install_live_session(monkeypatch, trace_dir, session)
    before = trace_store.load_trace(session.trace_id, trace_dir)

    with TestClient(
        app,
        headers={
            "x-etude-participant-tokens": "study-resume",
            "x-etude-participant-token": "study-resume",
        },
    ) as client:
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

    assert trace_store.load_trace(session.trace_id, trace_dir) == before


def test_injected_exact_fixture_reveals_and_previews_on_fresh_forks(
    monkeypatch: pytest.MonkeyPatch,
    completed_session: tuple[Path, GameSession, CanonicalReplayV1],
) -> None:
    trace_dir, session, _ = completed_session
    _install_live_session(
        monkeypatch,
        trace_dir,
        session,
        evidence_provider=ExactFixtureEvidenceProvider(),
        allow_fixture=True,
    )
    before = trace_store.load_trace(session.trace_id, trace_dir)

    with TestClient(
        app,
        headers={
            "x-etude-participant-tokens": "study-resume",
            "x-etude-participant-token": "study-resume",
        },
    ) as client:
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
        # A surfaced priority pass need not emit narrated facts. Preserve the
        # recorded continuation instead of inventing a beat for an empty step.
        assert previews["played"]["presentation"] == restored["continuation"]

        returned = client.post(f"/api/study-attempts/{retry['attempt_id']}/return")
        assert returned.json() == restored

    assert trace_store.load_trace(session.trace_id, trace_dir) == before


def test_drifted_or_runtime_fixture_evidence_fails_closed_but_keeps_return(
    monkeypatch: pytest.MonkeyPatch,
    completed_session: tuple[Path, GameSession, CanonicalReplayV1],
) -> None:
    trace_dir, session, _ = completed_session
    _install_live_session(
        monkeypatch,
        trace_dir,
        session,
        evidence_provider=ExactFixtureEvidenceProvider(drift_digest=True),
        allow_fixture=True,
    )

    with TestClient(
        app,
        headers={
            "x-etude-participant-tokens": "study-resume",
            "x-etude-participant-token": "study-resume",
        },
    ) as client:
        _, restored, retry = _first_pass_retry(client, session.trace_id)
        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 409
        assert reveal.json()["detail"] == "study_evidence_identity_mismatch"
        returned = client.post(f"/api/study-attempts/{retry['attempt_id']}/return")
        assert returned.json() == restored

    _install_live_session(
        monkeypatch,
        trace_dir,
        session,
        evidence_provider=ExactFixtureEvidenceProvider(),
        allow_fixture=False,
    )
    with TestClient(
        app,
        headers={
            "x-etude-participant-tokens": "study-resume",
            "x-etude-participant-token": "study-resume",
        },
    ) as client:
        _, restored, retry = _first_pass_retry(client, session.trace_id)
        reveal = client.post(f"/api/study-attempts/{retry['attempt_id']}/reveal")
        assert reveal.status_code == 409
        assert reveal.json()["detail"] == "fixture_evidence_forbidden"
        assert (
            client.post(f"/api/study-attempts/{retry['attempt_id']}/return").json()
            == restored
        )


def test_semantic_retry_supports_land_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    completed_session: tuple[Path, GameSession, CanonicalReplayV1],
) -> None:
    trace_dir, session, _ = completed_session
    _install_live_session(monkeypatch, trace_dir, session)

    with TestClient(
        app,
        headers={
            "x-etude-participant-tokens": "study-resume",
            "x-etude-participant-token": "study-resume",
        },
    ) as client:
        projection = client.get(f"/api/traces/{session.trace_id}/decisions").json()
        for row in projection["decisions"]:
            restored = client.get(
                f"/api/traces/{session.trace_id}/decisions/{row['address']}"
            ).json()
            play_land = next(
                (
                    offer
                    for offer in restored["frame"]["offers"]
                    if offer["verb"] == "play_land"
                ),
                None,
            )
            if play_land is not None:
                break
        else:
            pytest.fail("completed source has no play-land decision")
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
        assert unsupported.status_code == 200
        attempt = unsupported.json()["attempt_id"]
        returned = client.post(f"/api/study-attempts/{attempt}/return", json={})
        assert returned.status_code == 200
        assert returned.json()["frame"] == restored["frame"]

        drifted = deepcopy(command)
        drifted["expected_revision"] += 1
        invalid = client.post(
            f"/api/traces/{session.trace_id}/decisions/{row['address']}/retry",
            json={"command": drifted},
        )
        assert invalid.status_code == 409
        assert invalid.json()["detail"] == "study_command_identity_mismatch"

        assert client.post("/api/study-attempts/missing/reveal").status_code == 404
