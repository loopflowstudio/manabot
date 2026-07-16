"""Deterministic Study decision-index and landmark-ranking proof."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from etude.study_index import (
    build_semantic_receipt,
    build_study_index,
    canonical_json_bytes,
    main,
    semantic_sha256,
    source_replay_sha256,
)
from etude.study_protocol import (
    LandmarkReason,
    RecordedDecisionInput,
    StudyDecisionIndex,
    StudyIdentity,
)

ROOT = Path(__file__).parents[2]
PROTOCOL = ROOT / "protocol"
FIXTURES = PROTOCOL / "fixtures"
INPUT_JSON = json.loads(
    (FIXTURES / "recorded-match-decisions-curated.json").read_text(encoding="utf-8")
)
IDENTITY_JSON = json.loads(
    (FIXTURES / "study-index-identity-curated.json").read_text(encoding="utf-8")
)
INDEX_JSON = json.loads(
    (FIXTURES / "study-decision-index-curated.json").read_text(encoding="utf-8")
)
RECEIPT_JSON = json.loads(
    (ROOT / "experiments" / "data" / "w2-220-study-decision-index-v1.json").read_text(
        encoding="utf-8"
    )
)
INPUT_SCHEMA = json.loads(
    (PROTOCOL / "study-recorded-decisions-v1.schema.json").read_text(encoding="utf-8")
)
INDEX_SCHEMA = json.loads(
    (PROTOCOL / "study-index-v1.schema.json").read_text(encoding="utf-8")
)


def _shape(schema: dict) -> tuple[set[str], set[str]]:
    return set(schema.get("properties", {})), set(schema.get("required", []))


def _validated() -> tuple[RecordedDecisionInput, StudyIdentity, StudyDecisionIndex]:
    recorded = RecordedDecisionInput.model_validate(INPUT_JSON)
    identity = StudyIdentity.model_validate(IDENTITY_JSON)
    index = build_study_index(recorded, identity)
    return recorded, identity, index


def _identity_for(source: dict, template: dict | None = None) -> StudyIdentity:
    recorded = RecordedDecisionInput.model_validate(source)
    identity = deepcopy(IDENTITY_JSON if template is None else template)
    identity["source_replay_id"] = source["source_replay_id"]
    identity["source_replay_sha256"] = source_replay_sha256(recorded)
    return StudyIdentity.model_validate(identity)


def _secret_card() -> dict:
    return {
        "id": 99,
        "registry_key": 99,
        "name": "Secret Counterspell",
        "zone": "HAND",
        "owner_id": 0,
        "power": 0,
        "toughness": 0,
        "mana_value": 2,
        "types": {
            "is_creature": False,
            "is_land": False,
            "is_spell": True,
            "is_artifact": False,
            "is_enchantment": False,
            "is_planeswalker": False,
            "is_battle": False,
        },
    }


def test_shared_contracts_and_fixtures_round_trip_across_the_closed_boundary():
    Draft202012Validator.check_schema(INPUT_SCHEMA)
    Draft202012Validator.check_schema(INDEX_SCHEMA)
    Draft202012Validator(INPUT_SCHEMA).validate(INPUT_JSON)
    Draft202012Validator(INDEX_SCHEMA).validate(INDEX_JSON)

    recorded, identity, index = _validated()
    checked = StudyDecisionIndex.model_validate(INDEX_JSON)
    assert index == checked
    assert recorded.decision_count == len(index.decisions) == 8
    assert source_replay_sha256(recorded) == identity.source_replay_sha256
    assert canonical_json_bytes(index.model_dump(mode="json", exclude_unset=True)) == (
        canonical_json_bytes(INDEX_JSON)
    )

    for source, decision in zip(recorded.decisions, index.decisions, strict=True):
        assert decision.ordinal == source.ordinal
        assert decision.event_cursor == source.event_cursor
        assert decision.frame == source.frame
        assert decision.offer == source.offer
        assert decision.played == source.played
    assert len({decision.id for decision in index.decisions}) == 8


def test_python_index_shapes_match_rust_generated_schemas():
    python_input = RecordedDecisionInput.model_json_schema()
    python_index = StudyDecisionIndex.model_json_schema()
    assert _shape(INPUT_SCHEMA) == _shape(python_input)
    assert _shape(INDEX_SCHEMA) == _shape(python_index)
    for name in ("RecordedDecision",):
        assert _shape(INPUT_SCHEMA["$defs"][name]) == _shape(
            python_input["$defs"][name]
        )
    for name in ("RankedStudyLandmark", "StudyDecision", "StudyIdentity"):
        assert _shape(INDEX_SCHEMA["$defs"][name]) == _shape(
            python_index["$defs"][name]
        )


def test_ranker_returns_five_diverse_recommendations_without_collapsing_combat():
    _, _, index = _validated()

    assert len(index.landmarks) == 5
    assert [landmark.rank for landmark in index.landmarks] == [1, 2, 3, 4, 5]
    recommended = {landmark.decision_id for landmark in index.landmarks}
    assert index.decisions[3].id in recommended
    assert index.decisions[4].id not in recommended
    assert index.decisions[6].id not in recommended
    assert index.decisions[7].id not in recommended
    assert index.decisions[6].automatic
    assert len(index.decisions[7].frame.offers) == 1

    reasons = {reason for landmark in index.landmarks for reason in landmark.reasons}
    assert {
        LandmarkReason.PRIORITY_COMMITMENT,
        LandmarkReason.PRIORITY_RESPONSE,
        LandmarkReason.TARGET_SELECTION,
        LandmarkReason.ATTACK_DECLARATION,
        LandmarkReason.BLOCK_DECLARATION,
    } <= reasons


def test_identity_and_ranking_ignore_model_and_analysis_budget():
    recorded, identity, baseline = _validated()
    mutated_json = identity.model_dump(mode="json")
    mutated_json["model"] = {
        "id": "unrelated-policy",
        "checkpoint_sha256": "a" * 64,
    }
    mutated_json["analysis_budget"] = {
        "id": "unrelated-budget",
        "max_nodes": 9999,
        "sampled_worlds": 99,
        "rollouts_per_world": 77,
    }
    mutated = build_study_index(recorded, StudyIdentity.model_validate(mutated_json))

    assert [decision.id for decision in mutated.decisions] == [
        decision.id for decision in baseline.decisions
    ]
    assert [
        (landmark.decision_id, landmark.reasons) for landmark in mutated.landmarks
    ] == [(landmark.decision_id, landmark.reasons) for landmark in baseline.landmarks]


def test_historical_input_changes_create_new_decision_identity():
    recorded, _, baseline = _validated()
    changed = deepcopy(INPUT_JSON)
    changed["decisions"][0]["frame"]["frame_hash"] = "historical-state-changed"
    changed_recorded = RecordedDecisionInput.model_validate(changed)
    changed_index = build_study_index(changed_recorded, _identity_for(changed))

    assert source_replay_sha256(changed_recorded) != source_replay_sha256(recorded)
    assert changed_index.decisions[0].id != baseline.decisions[0].id


def test_viewer_safe_boundary_rejects_private_sidecars_and_unbound_digest():
    private = deepcopy(INPUT_JSON)
    private["decisions"][0]["frame"]["projection"]["opponent"]["hand"] = [
        _secret_card()
    ]
    with pytest.raises(ValidationError, match="opponent-private hand"):
        RecordedDecisionInput.model_validate(private)

    rng_sidecar = deepcopy(INPUT_JSON)
    rng_sidecar["rng_seed"] = 377
    with pytest.raises(ValidationError):
        RecordedDecisionInput.model_validate(rng_sidecar)

    recorded = RecordedDecisionInput.model_validate(INPUT_JSON)
    raw_digest = deepcopy(IDENTITY_JSON)
    raw_digest["source_replay_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="validated viewer-safe input"):
        build_study_index(recorded, StudyIdentity.model_validate(raw_digest))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source.update(decision_count=7),
            "declared decision count",
        ),
        (
            lambda source: source["decisions"][1].update(ordinal=4),
            "ordinals must be contiguous",
        ),
        (
            lambda source: source["decisions"][1].update(event_cursor=1000),
            "event cursors must strictly increase",
        ),
        (
            lambda source: source["decisions"][0]["played"].update(prompt_id=999),
            "played command identity drifted",
        ),
    ],
)
def test_invalid_history_is_rejected_instead_of_repaired(mutate, message):
    invalid = deepcopy(INPUT_JSON)
    mutate(invalid)
    with pytest.raises(ValidationError, match=message):
        RecordedDecisionInput.model_validate(invalid)


def test_empty_and_insufficient_inputs_remain_complete_without_filler_landmarks():
    empty = {
        "version": 1,
        "source_replay_id": "study-index/empty",
        "decision_count": 0,
        "decisions": [],
    }
    empty_recorded = RecordedDecisionInput.model_validate(empty)
    empty_index = build_study_index(empty_recorded, _identity_for(empty))
    assert empty_index.decisions == []
    assert empty_index.landmarks == []

    forced = deepcopy(INPUT_JSON)
    forced["source_replay_id"] = "study-index/forced-only"
    forced["decision_count"] = 1
    forced["decisions"] = [forced["decisions"][7]]
    forced["decisions"][0]["ordinal"] = 0
    forced_recorded = RecordedDecisionInput.model_validate(forced)
    forced_index = build_study_index(forced_recorded, _identity_for(forced))
    assert len(forced_index.decisions) == 1
    assert forced_index.landmarks == []

    checks = {
        "opponent_private_hand_rejected": True,
        "raw_authority_digest_rejected": True,
        "rng_sidecar_rejected": True,
        "viewer_boundary_stable": True,
    }
    empty_receipt = build_semantic_receipt(
        empty_recorded,
        _identity_for(empty),
        empty_index,
        repeats=1,
        repeat_artifact_digests=1,
        boundary_checks=checks,
    )
    forced_receipt = build_semantic_receipt(
        forced_recorded,
        _identity_for(forced),
        forced_index,
        repeats=1,
        repeat_artifact_digests=1,
        boundary_checks=checks,
    )
    assert empty_receipt["ranking_status"] == "no_recorded_decisions"
    assert empty_receipt["completeness_ratio"] == 1.0
    assert forced_receipt["ranking_status"] == "insufficient_supported_landmarks"


def test_generated_five_hundred_decision_input_remains_lossless():
    large = deepcopy(INPUT_JSON)
    large["source_replay_id"] = "study-index/generated-500"
    template = large["decisions"][0]
    decisions = []
    for ordinal in range(500):
        decision = deepcopy(template)
        decision["ordinal"] = ordinal
        decision["event_cursor"] = 10_000 + ordinal
        decision["frame"]["revision"] = 1_000 + ordinal
        decision["frame"]["frame_hash"] = f"generated-frame-{ordinal}"
        decision["frame"]["prompt"]["id"] = 2_000 + ordinal
        decision["frame"]["action_space"] = "UNSUPPORTED_GENERATED"
        decision["played"]["expected_revision"] = 1_000 + ordinal
        decision["played"]["prompt_id"] = 2_000 + ordinal
        decision["played"]["command_id"] = f"generated-command-{ordinal}"
        for event in decision["presentation"]:
            event["caused_by"] = decision["played"]["command_id"]
        decisions.append(decision)
    large["decision_count"] = len(decisions)
    large["decisions"] = decisions

    recorded = RecordedDecisionInput.model_validate(large)
    index = build_study_index(recorded, _identity_for(large))
    assert len(index.decisions) == 500
    assert len({decision.id for decision in index.decisions}) == 500
    assert index.landmarks == []


def test_checked_semantic_receipt_is_deterministic_and_observations_are_separate(
    tmp_path: Path,
):
    observations = tmp_path / "observations.json"
    assert (
        main(
            [
                str(FIXTURES / "recorded-match-decisions-curated.json"),
                "--identity",
                str(FIXTURES / "study-index-identity-curated.json"),
                "--verify",
                "--repeats",
                "1000",
                "--semantic-receipt",
                str(
                    ROOT
                    / "experiments"
                    / "data"
                    / "w2-220-study-decision-index-v1.json"
                ),
                "--observations",
                str(observations),
            ]
        )
        == 0
    )
    observation_json = json.loads(observations.read_text(encoding="utf-8"))
    assert observation_json["repeats"] == 1000
    assert observation_json["p50_ms"] >= 0
    assert observation_json["p95_ms"] >= observation_json["p50_ms"]
    assert "observed_at" in observation_json
    assert not {"observed_at", "p50_ms", "p95_ms"} & RECEIPT_JSON.keys()
    assert RECEIPT_JSON["decision_count"] == 8
    assert RECEIPT_JSON["landmark_count"] == 5
    assert RECEIPT_JSON["repeat_artifact_digests"] == 1
    assert semantic_sha256(INDEX_JSON) == RECEIPT_JSON["artifact_sha256"]
