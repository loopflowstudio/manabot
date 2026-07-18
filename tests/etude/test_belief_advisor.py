from __future__ import annotations

import json

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import numpy as np
import pytest

from etude.advice import (
    ADVISOR_SOURCE_PATHS,
    VERSIONED_FIXTURE_PATH,
    AdviceProvider,
    AdviceRequest,
    RegisteredAdvisor,
    ResolvedAdvisorDecision,
    VersionedAdviceFixture,
    advice_schema,
    load_versioned_advice_fixture,
    parse_advice_response_bytes,
    request_versioned_fixture_advice,
)
from etude.advice_identity import AbiIdentity, CheckpointArtifact
from etude.server import app
from etude.testing_house_protocol import ModelInferredBeliefProvenance
from experiments.runners.run_belief_strategy_advisor import (
    AdvisorRuntime,
    build_runtime,
)
from manabot.sim.conditional_search import (
    ConditionalStrategyResult,
    ConditionResult,
    project_viewer_safe_result,
)
from managym.possible_worlds import PossibleWorldSpace


@pytest.fixture(scope="module")
def runtime() -> AdvisorRuntime:
    return build_runtime()


def _unavailable(request: AdviceRequest) -> str:
    response = parse_advice_response_bytes(request_versioned_fixture_advice(request))
    assert response.status == "unavailable"
    assert response.strategy is None
    assert response.evidence is None
    assert response.deltas is None
    assert response.reason is not None
    return response.reason


def test_checked_advice_v1_fixture_round_trips_schema_and_models() -> None:
    payload = json.loads(VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8"))
    schema = advice_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    fixture = VersionedAdviceFixture.model_validate(payload)
    assert fixture == load_versioned_advice_fixture()
    assert fixture.response.strategy is not None
    strategy = fixture.response.strategy
    assert all(
        not hasattr(action, "alternative_id")
        for action in strategy.scenarios[0].actions
    )
    assert [offer.offer_id for offer in strategy.offers] == [
        action.offer_id for action in strategy.scenarios[0].actions
    ]
    assert all(
        abs(sum(action.probability for action in scenario.actions) - 1.0) < 1e-9
        for scenario in strategy.scenarios
    )
    assert strategy.comparison is not None
    assert any(
        abs(action.policy_probability) > 0.0 for action in strategy.comparison.actions
    )


def test_public_fixture_cannot_carry_authority_private_fields() -> None:
    payload = json.loads(VERSIONED_FIXTURE_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "authority_private",
        "world_q_values",
        "world_root_values",
        "world_visit_counts",
        "branch_receipt",
        "sampled_indexes",
        "opponent_hand",
        "actual_query_truth",
        "rng_tapes",
        '"weights"',
    ):
        assert forbidden not in serialized


def test_versioned_endpoint_returns_the_provider_canonical_bytes() -> None:
    fixture = load_versioned_advice_fixture()
    expected = request_versioned_fixture_advice(fixture.request)
    with TestClient(app) as client:
        response = client.post(
            "/api/advice", json=fixture.request.model_dump(mode="json")
        )
    assert response.status_code == 200
    assert response.content == expected


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("source_replay_id", "replay.other", "replay_mismatch"),
        ("source_replay_sha256", "0" * 64, "replay_mismatch"),
        ("match_id", "match.other", "match_mismatch"),
        ("world_id", "w3", "world_mismatch"),
        ("content_sha256", "1" * 64, "content_mismatch"),
    ],
)
def test_identity_perturbations_fail_closed(
    field: str,
    value: object,
    reason: str,
) -> None:
    fixture = load_versioned_advice_fixture()
    assert fixture.request.advisor_identity is not None
    changed_identity = fixture.request.advisor_identity.model_copy(
        update={field: value}
    )
    changed = fixture.request.model_copy(update={"advisor_identity": changed_identity})
    assert _unavailable(changed) == reason


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("observation_abi", "observation_abi_mismatch"),
        ("action_abi", "action_abi_mismatch"),
        ("possible_world_abi", "possible_world_abi_mismatch"),
    ],
)
def test_abi_perturbations_fail_closed(field: str, reason: str) -> None:
    fixture = load_versioned_advice_fixture()
    assert fixture.request.advisor_identity is not None
    current = getattr(fixture.request.advisor_identity, field)
    changed_abi = AbiIdentity(
        name=current.name,
        version=current.version,
        sha256="f" * 64,
    )
    changed_identity = fixture.request.advisor_identity.model_copy(
        update={field: changed_abi}
    )
    changed = fixture.request.model_copy(update={"advisor_identity": changed_identity})
    assert _unavailable(changed) == reason


def test_compute_seed_and_artifact_perturbations_fail_closed() -> None:
    fixture = load_versioned_advice_fixture()
    assert fixture.request.advisor_identity is not None
    identity = fixture.request.advisor_identity
    cases = (
        (
            identity.model_copy(
                update={
                    "compute": identity.compute.model_copy(
                        update={"simulations_per_scenario": 13}
                    )
                }
            ),
            "compute_mismatch",
        ),
        (
            identity.model_copy(
                update={"seed": identity.seed.model_copy(update={"root_seed": 198})}
            ),
            "seed_plan_mismatch",
        ),
        (
            identity.model_copy(
                update={
                    "artifact": identity.artifact.model_copy(
                        update={"source_bundle_sha256": "e" * 64}
                    )
                }
            ),
            "advisor_artifact_mismatch",
        ),
    )
    for changed_identity, reason in cases:
        changed = fixture.request.model_copy(
            update={"advisor_identity": changed_identity}
        )
        assert _unavailable(changed) == reason


def test_decision_and_viewer_perturbations_fail_closed() -> None:
    fixture = load_versioned_advice_fixture()
    parsed = fixture.request.parsed_address
    changed_address = parsed.__class__(
        version=parsed.version,
        replay_id=parsed.replay_id,
        match_id=parsed.match_id,
        ordinal=parsed.ordinal,
        viewer=parsed.viewer,
        revision=parsed.revision,
        prompt_id=parsed.prompt_id,
        offer_id=parsed.offer_id,
        command_id=parsed.command_id,
        presentation_cursor=parsed.presentation_cursor,
        decision_sha256="0" * 64,
    ).serialize()
    changed = fixture.request.model_copy(update={"address": changed_address})
    assert _unavailable(changed) == "decision_identity_mismatch"

    assert fixture.request.viewer is not None
    changed_viewer = fixture.request.viewer.model_copy(
        update={"table_id": "table.other"}
    )
    changed = fixture.request.model_copy(update={"viewer": changed_viewer})
    assert _unavailable(changed) == "belief_viewer_mismatch"


def test_unvisited_actions_project_as_unavailable_instead_of_half_value() -> None:
    condition = ConditionResult(
        condition_id="authored",
        condition_mass=1.0,
        support=1,
        sampled_worlds=1,
        visit_counts=np.array([4, 0], dtype=np.int64),
        q_values=np.array([0.75, 0.5], dtype=np.float32),
        root_value=0.75,
        world_q_values=np.array([[0.75, 0.5]], dtype=np.float32),
        world_root_values=np.array([0.75], dtype=np.float32),
        uncertainty=0.0,
        simulations=4,
        cap_hits=0,
        tree_nodes=5,
        max_depth=2,
        branch_driver_id="selected_branch_driver/v1",
        branch_receipt={},
        world_visit_counts=np.array([[4, 0]], dtype=np.int64),
    )
    result = ConditionalStrategyResult(
        conditions=(condition,),
        action_count=2,
        action_labels=("act", "pass"),
        root_state_digest="0" * 64,
        planner="determinized_puct",
        search_params={},
        prior_sha256="1" * 64,
        plan_sha256="2" * 64,
        identities={},
        realized_compute={},
        comparison_deltas={},
    )
    projection = project_viewer_safe_result(result, offer_ids=(10, 11))
    actions = projection["conditions"][0]["actions"]
    assert actions[0]["q"]["status"] == "available"
    assert actions[1]["q"] == {
        "status": "unavailable",
        "reason": "no_realized_visits",
    }
    assert actions[1]["uncertainty"]["status"] == "unavailable"


def test_belief_digest_mismatch_is_typed_unavailable(
    runtime: AdvisorRuntime,
) -> None:
    fixture_resolver = runtime.provider.belief_resolver
    payload = fixture_resolver.payloads[runtime.request.scenario.id]
    bad_payload = payload.model_construct(
        **{
            **payload.model_dump(),
            "distribution_sha256": "0" * 64,
        }
    )

    class BadResolver:
        def resolve(self, scenario, decision, viewer):
            del decision, viewer
            if scenario.id == runtime.request.scenario.id:
                return bad_payload
            return fixture_resolver.payloads[scenario.id]

    provider = AdviceProvider(
        registered=runtime.provider.registered,
        decision_resolver=runtime.provider.decision_resolver,
        belief_resolver=BadResolver(),
    )
    response = parse_advice_response_bytes(
        provider.advise(runtime.request, recompute=True)
    )
    assert response.status == "unavailable"
    assert response.reason == "belief_distribution_mismatch"
    assert response.strategy is None


def test_unregistered_checkpoint_identity_is_never_substituted(
    runtime: AdvisorRuntime,
) -> None:
    assert runtime.request.advisor_identity is not None
    checkpoint_identity = runtime.request.advisor_identity.model_copy(
        update={
            "artifact": CheckpointArtifact(
                kind="checkpoint",
                checkpoint_id="missing-belief-advisor",
                checkpoint_sha256="a" * 64,
                checkpoint_bytes=1,
                manifest_sha256="b" * 64,
                training_seed=197,
                observation_abi=AbiIdentity(
                    name="manabot_observation",
                    version="w2",
                    sha256="c" * 64,
                ),
                action_abi=AbiIdentity(
                    name="manabot_action",
                    version="w2",
                    sha256="d" * 64,
                ),
                value_mode="neutral",
            )
        }
    )
    scenario = runtime.request.scenario.model_copy(
        update={
            "source": runtime.request.scenario.source.model_copy(
                update={"advice_identity": checkpoint_identity}
            )
        }
    )
    assert runtime.request.comparison_scenario is not None
    comparison = runtime.request.comparison_scenario.model_copy(
        update={
            "source": runtime.request.comparison_scenario.source.model_copy(
                update={"advice_identity": checkpoint_identity}
            )
        }
    )
    request = runtime.request.model_copy(
        update={
            "scenario": scenario,
            "comparison_scenario": comparison,
            "advisor_identity": checkpoint_identity,
        }
    )
    provider = AdviceProvider(
        registered=RegisteredAdvisor(checkpoint_identity, ADVISOR_SOURCE_PATHS),
        decision_resolver=runtime.provider.decision_resolver,
        belief_resolver=runtime.provider.belief_resolver,
    )
    response = parse_advice_response_bytes(provider.advise(request, recompute=True))
    assert response.status == "unavailable"
    assert response.reason == "advisor_artifact_mismatch"
    assert response.strategy is None


def test_missing_model_inferred_belief_bytes_are_typed_unavailable(
    runtime: AdvisorRuntime,
) -> None:
    inferred = runtime.request.scenario.model_copy(
        update={
            "id": "missing-model-inferred-belief",
            "provenance": ModelInferredBeliefProvenance(
                belief_model_id="missing-history-model",
                checkpoint_sha256="c" * 64,
                artifact_manifest_sha256="d" * 64,
                viewer_history_sha256="e" * 64,
            ),
        }
    )
    request = runtime.request.model_copy(
        update={"scenario": inferred, "comparison_scenario": None}
    )
    response = parse_advice_response_bytes(
        runtime.provider.advise(request, recompute=True)
    )
    assert response.status == "unavailable"
    assert response.reason == "belief_artifact_unavailable"
    assert response.strategy is None


def test_viewer_equivalent_private_roots_produce_identical_public_bytes(
    runtime: AdvisorRuntime,
) -> None:
    assert runtime.request.viewer is not None
    resolved = runtime.provider.decision_resolver.resolve(
        runtime.request.parsed_address,
        runtime.request.viewer,
    )
    left_root = resolved.world_space.materialize(0, seed=7001)
    right_root = resolved.world_space.materialize(
        resolved.world_space.support_size - 1,
        seed=7002,
    )
    assert left_root.state_digest() != right_root.state_digest()
    assert left_root.semantic_observation_json(
        0
    ) == right_root.semantic_observation_json(0)

    class FixedDecisionResolver:
        def __init__(self, root):
            self.root = root

        def resolve(self, address, viewer):
            del address
            space = PossibleWorldSpace.from_engine(self.root, viewer.rules_viewer)
            assert space.identity == resolved.world_space.identity
            return ResolvedAdvisorDecision(
                address=resolved.address,
                frame=resolved.frame,
                semantic_frame=resolved.semantic_frame,
                root=self.root,
                world_space=space,
                source_replay_sha256=resolved.source_replay_sha256,
                content_sha256=resolved.content_sha256,
            )

    def provider(root) -> AdviceProvider:
        return AdviceProvider(
            registered=runtime.provider.registered,
            decision_resolver=FixedDecisionResolver(root),
            belief_resolver=runtime.provider.belief_resolver,
        )

    left = provider(left_root).advise(runtime.request, recompute=True)
    right = provider(right_root).advise(runtime.request, recompute=True)
    assert left == right
