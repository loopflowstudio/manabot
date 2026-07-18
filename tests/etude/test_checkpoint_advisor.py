"""Checkpoint-backed advice-v1 parity, attribution, and fail-closed behavior."""

from __future__ import annotations

import json
import shutil

from fastapi.testclient import TestClient
import pytest

from etude.advice import (
    ADVISOR_SOURCE_PATHS,
    CHECKPOINT_VERSIONED_FIXTURE_PATH,
    AdviceProvider,
    AdviceRequest,
    RegisteredAdvisor,
    StudyAdvisorDecisionResolver,
    VersionedAdviceFixture,
    parse_advice_response_bytes,
    serialize_advice_response,
)
from etude.advice_identity import AdvisorIdentity, CheckpointArtifact
from etude.server import app
from etude.study_branch import StudyForkProvider
from etude.testing_house_protocol import ViewerIdentity
from experiments.runners.run_checkpoint_strategy_advisor import (
    CheckpointAdvisorRuntime,
    build_checkpoint_runtime,
    checkpoint_fixture,
)


@pytest.fixture(scope="module")
def runtime() -> CheckpointAdvisorRuntime:
    return build_checkpoint_runtime()


def _request_with_identity(
    runtime: CheckpointAdvisorRuntime,
    identity: AdvisorIdentity,
    **updates,
) -> AdviceRequest:
    assert runtime.request.scenario is not None
    scenario = runtime.request.scenario.model_copy(
        update={
            "source": runtime.request.scenario.source.model_copy(
                update={"advice_identity": identity}
            )
        }
    )
    return runtime.request.model_copy(
        update={"scenario": scenario, "advisor_identity": identity, **updates}
    )


def _unavailable_for_identity(
    runtime: CheckpointAdvisorRuntime,
    identity: AdvisorIdentity,
) -> str:
    provider = AdviceProvider(
        registered=RegisteredAdvisor(identity, ADVISOR_SOURCE_PATHS),
        decision_resolver=runtime.provider.decision_resolver,
        belief_resolver=runtime.provider.belief_resolver,
    )
    response = parse_advice_response_bytes(
        provider.advise(_request_with_identity(runtime, identity), recompute=True)
    )
    assert response.status == "unavailable"
    assert response.reason is not None
    assert response.strategy is None
    return response.reason


def test_checked_checkpoint_fixture_is_served_by_the_existing_endpoint(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    fixture_bytes = CHECKPOINT_VERSIONED_FIXTURE_PATH.read_bytes()
    fixture = VersionedAdviceFixture.model_validate_json(fixture_bytes)
    assert fixture.request == runtime.request
    expected = serialize_advice_response(fixture.response)

    with TestClient(app) as client:
        response = client.post(
            "/api/advice",
            json=runtime.request.model_dump(mode="json"),
        )

    assert response.status_code == 200
    assert response.content == expected


def test_fresh_live_and_study_checkpoint_advice_is_byte_identical(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    fixture, payload, retained_receipt = checkpoint_fixture(runtime)

    assert payload == serialize_advice_response(fixture.response)
    assert fixture == VersionedAdviceFixture.model_validate_json(
        CHECKPOINT_VERSIONED_FIXTURE_PATH.read_bytes()
    )
    assert retained_receipt.no_write_or_generation_verified
    assert retained_receipt.before == retained_receipt.after
    assert retained_receipt.before.manifest_digest == (
        "3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf"
    )
    assert retained_receipt.before.selected_checkpoints[0].file.sha256 == (
        "1673a237ef2460d0e699667987c29fe6b42c28711bdb2041989f37692edbd1e6"
    )


@pytest.mark.parametrize("drift", ("changed", "generated"))
def test_checkpoint_fixture_rejects_retained_evidence_drift(
    runtime: CheckpointAdvisorRuntime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    drift: str,
) -> None:
    import experiments.runners.run_checkpoint_strategy_advisor as runner
    import manabot.sim.search_runtime as search_runtime

    retained_root = tmp_path / "result"
    checkpoints = retained_root / "checkpoints"
    checkpoints.mkdir(parents=True)
    shutil.copyfile(
        search_runtime.INT7_VALUE_TARGET_MANIFEST,
        retained_root / "manifest.json",
    )
    shutil.copyfile(
        search_runtime.INT7_VALUE_TARGET_RESULT
        / "checkpoints"
        / "visit_policy_only-seed-197.pt",
        checkpoints / "visit_policy_only-seed-197.pt",
    )
    retained_receipt = retained_root / "receipt.json"
    retained_receipt.write_text('{"status":"retained"}\n')
    monkeypatch.setattr(search_runtime, "INT7_VALUE_TARGET_RESULT", retained_root)
    monkeypatch.setattr(
        search_runtime,
        "INT7_VALUE_TARGET_MANIFEST",
        retained_root / "manifest.json",
    )

    snapshot_calls = 0

    def snapshot_with_drift(training_seeds):
        nonlocal snapshot_calls
        snapshot = search_runtime.retained_int7_evidence_snapshot(training_seeds)
        snapshot_calls += 1
        if snapshot_calls == 1:
            if drift == "changed":
                retained_receipt.write_text('{"status":"changed"}\n')
            else:
                (retained_root / "generated.json").write_text("{}\n")
        return snapshot

    monkeypatch.setattr(
        runner,
        "retained_int7_evidence_snapshot",
        snapshot_with_drift,
    )

    with pytest.raises(RuntimeError, match="changed retained INT-7 evidence"):
        checkpoint_fixture(runtime)
    assert snapshot_calls == 2


def test_available_checkpoint_evidence_has_complete_exact_identities(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    assert runtime.authority_receipt_sha256 == (
        "57f5e2d1a482831e4e1edeb69632c0b34030d2e8ee701eab9b13b484c6937147"
    )
    assert runtime.parity_receipt_sha256 == (
        "828d678f3c84f28f82412dfb6208ed9a1157bf41030a5a0d3c6e8445fac76ec8"
    )
    fixture = VersionedAdviceFixture.model_validate_json(
        CHECKPOINT_VERSIONED_FIXTURE_PATH.read_bytes()
    )
    response = fixture.response
    assert response.status == "ok"
    assert response.strategy is not None
    assert response.request_sha256 is not None
    assert response.response_sha256 is not None
    identity = response.strategy.advisor_identity
    assert identity == runtime.request.advisor_identity
    assert identity.world_id == "w2"
    assert identity.planner_id == "determinized_puct"
    assert identity.evaluator_id == "checkpoint_policy_neutral_value"
    assert identity.compute.id == "int7-policy-only-puct-s32-w4-v1"
    assert identity.compute.simulations_per_scenario == 32
    assert identity.compute.sampled_worlds == 4
    assert identity.seed.root_seed == 197
    assert isinstance(identity.artifact, CheckpointArtifact)
    assert identity.artifact.checkpoint_sha256 == (
        "1673a237ef2460d0e699667987c29fe6b42c28711bdb2041989f37692edbd1e6"
    )
    assert identity.artifact.checkpoint_bytes == 428_629
    assert identity.artifact.manifest_sha256 == (
        "3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf"
    )
    assert identity.artifact.value_mode == "neutral"

    strategy = response.strategy
    assert strategy.decision_sha256
    assert strategy.viewer_frame_sha256 == response.frame.frame_hash
    assert strategy.prior_sha256
    assert strategy.plan_sha256
    assert len(strategy.scenarios) == 1
    scenario = strategy.scenarios[0]
    assert scenario.belief.normalized_belief_sha256
    assert scenario.belief.distribution_sha256
    offer_ids = [offer.offer_id for offer in strategy.offers]
    assert [action.offer_id for action in scenario.actions] == offer_ids
    assert sum(action.probability for action in scenario.actions) == pytest.approx(1.0)
    for action in scenario.actions:
        assert action.visits > 0
        assert action.q.status == "available"
        assert action.robustness.status == "available"
        assert action.uncertainty.status == "available"
    assert scenario.root_value.status == "available"
    assert scenario.root_uncertainty.status == "available"

    public = json.dumps(response.model_dump(mode="json"), sort_keys=True)
    for forbidden in (
        "_study_roots",
        "root_state_digest",
        "world_q_values",
        "world_visit_counts",
        "world_root_values",
        "sampled_indexes",
        "opponent_hand",
        "actual_query_truth",
        "branch_receipt",
        "checkpoint_path",
        "visit_policy_only-seed-197.pt",
    ):
        assert forbidden not in public


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_sha256", "a" * 64),
        ("manifest_sha256", "b" * 64),
        ("checkpoint_bytes", 1),
        (
            "checkpoint_id",
            "int-7-visit_terminal-seed-197-c669e7e19a938258",
        ),
    ),
)
def test_checkpoint_registry_mismatches_fail_closed(
    runtime: CheckpointAdvisorRuntime,
    field: str,
    value,
) -> None:
    assert runtime.request.advisor_identity is not None
    artifact = runtime.request.advisor_identity.artifact
    assert isinstance(artifact, CheckpointArtifact)
    identity = runtime.request.advisor_identity.model_copy(
        update={"artifact": artifact.model_copy(update={field: value})}
    )

    assert _unavailable_for_identity(runtime, identity) == "advisor_artifact_mismatch"


def test_model_and_protocol_abi_mismatches_fail_closed(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    assert runtime.request.advisor_identity is not None
    artifact = runtime.request.advisor_identity.artifact
    assert isinstance(artifact, CheckpointArtifact)
    model_abi_identity = runtime.request.advisor_identity.model_copy(
        update={
            "artifact": artifact.model_copy(
                update={
                    "observation_abi": artifact.observation_abi.model_copy(
                        update={"sha256": "c" * 64}
                    )
                }
            )
        }
    )
    assert (
        _unavailable_for_identity(runtime, model_abi_identity)
        == "advisor_artifact_mismatch"
    )

    protocol_abi_identity = runtime.request.advisor_identity.model_copy(
        update={
            "observation_abi": runtime.request.advisor_identity.observation_abi.model_copy(
                update={"sha256": "d" * 64}
            )
        }
    )
    assert (
        _unavailable_for_identity(runtime, protocol_abi_identity)
        == "observation_abi_mismatch"
    )


def test_address_and_viewer_mismatches_fail_closed(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    assert runtime.request.advisor_identity is not None
    parsed = runtime.request.parsed_address.model_copy(
        update={"replay_id": "replay.other"}
    )
    address_request = _request_with_identity(
        runtime,
        runtime.request.advisor_identity,
        address=parsed.serialize(),
    )
    address_response = parse_advice_response_bytes(
        runtime.provider.advise(address_request, recompute=True)
    )
    assert address_response.status == "unavailable"
    assert address_response.reason == "decision_identity_mismatch"

    missing = runtime.request.parsed_address.model_copy(update={"ordinal": 10_000})
    missing_request = _request_with_identity(
        runtime,
        runtime.request.advisor_identity,
        address=missing.serialize(),
    )
    missing_response = parse_advice_response_bytes(
        runtime.provider.advise(missing_request, recompute=True)
    )
    assert missing_response.status == "unavailable"
    assert missing_response.reason == "decision_not_found"

    assert runtime.request.viewer is not None
    viewer_request = _request_with_identity(
        runtime,
        runtime.request.advisor_identity,
        viewer=ViewerIdentity(
            viewer_id="viewer.not-the-author",
            table_id=runtime.request.viewer.table_id,
            rules_viewer=runtime.request.viewer.rules_viewer,
        ),
    )
    viewer_response = parse_advice_response_bytes(
        runtime.provider.advise(viewer_request, recompute=True)
    )
    assert viewer_response.status == "unavailable"
    assert viewer_response.reason == "belief_viewer_mismatch"


def test_drifted_retained_root_fails_closed(
    runtime: CheckpointAdvisorRuntime,
) -> None:
    assert runtime.request.viewer is not None
    assert runtime.request.advisor_identity is not None
    resolved = runtime.session.study_fork_provider().resolve_advisor_decision(
        runtime.request.address,
        runtime.request.viewer.rules_viewer,
    )
    isolated_root = resolved.root
    drifted_forks = StudyForkProvider(
        runtime.session.canonical_replay(),
        {int(resolved.restored.ordinal): isolated_root},
    )
    isolated_root.step(0)
    provider = AdviceProvider(
        registered=RegisteredAdvisor(
            runtime.request.advisor_identity,
            ADVISOR_SOURCE_PATHS,
        ),
        decision_resolver=StudyAdvisorDecisionResolver(
            drifted_forks,
            runtime.request.advisor_identity.source_replay_sha256,
        ),
        belief_resolver=runtime.study_provider.belief_resolver,
    )

    response = parse_advice_response_bytes(
        provider.advise(runtime.request, recompute=True)
    )
    assert response.status == "unavailable"
    assert response.reason == "decision_root_unavailable"
    assert response.strategy is None
