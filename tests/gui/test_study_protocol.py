"""Cross-language and privacy conformance for study artifact v1."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from gui.study_protocol import KnowledgeScope, StudyArtifact

PROTOCOL_DIR = Path(__file__).parents[2] / "protocol"
FIXTURE = json.loads(
    (PROTOCOL_DIR / "fixtures" / "study-curated-decision.json").read_text(
        encoding="utf-8"
    )
)
SOURCE_REPLAY_BYTES = (PROTOCOL_DIR / "fixtures" / "bolt-target.json").read_bytes()
SOURCE_REPLAY = json.loads(SOURCE_REPLAY_BYTES)
RUST_SCHEMA = json.loads(
    (PROTOCOL_DIR / "study-v1.schema.json").read_text(encoding="utf-8")
)
RUST_VALIDATOR = Draft202012Validator(RUST_SCHEMA)
PYTHON_SCHEMA = StudyArtifact.model_json_schema()

STRUCT_MODELS = (
    "AnalysisBudgetIdentity",
    "ContentPackIdentity",
    "DecisionAlternative",
    "DecisionEvidence",
    "EngineIdentity",
    "EvidenceProvenance",
    "ModelIdentity",
    "PolicyMass",
    "SampledWorldRobustness",
    "SearchValue",
    "StudyIdentity",
    "StudyLandmark",
    "UncertaintyEvidence",
    "VisitCount",
)


def _shape(schema: dict) -> tuple[set[str], set[str]]:
    return set(schema.get("properties", {})), set(schema.get("required", []))


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


def test_python_round_trips_shared_historical_decision_and_distinct_evidence():
    Draft202012Validator.check_schema(RUST_SCHEMA)
    RUST_VALIDATOR.validate(FIXTURE)
    artifact = StudyArtifact.model_validate(FIXTURE)

    assert artifact.model_dump(mode="json", exclude_unset=True) == FIXTURE
    assert artifact.identity.source_replay_sha256 == hashlib.sha256(
        SOURCE_REPLAY_BYTES
    ).hexdigest()
    landmark = artifact.landmarks[0]
    assert landmark.frame.model_dump(mode="json", exclude_unset=True) == SOURCE_REPLAY[
        "recovery"
    ]["frame"]
    assert landmark.played.model_dump(mode="json") == SOURCE_REPLAY["command"]
    assert landmark.frame.offers[1] == landmark.offer
    assert landmark.played.offer_id == landmark.offer_id
    assert {row.alternative for row in landmark.evidence.policy_mass} == {
        alternative.id for alternative in landmark.alternatives
    }
    assert landmark.evidence.policy_mass is not landmark.evidence.search_value
    assert landmark.evidence.visits is not landmark.evidence.uncertainty


def test_python_study_shapes_match_rust_generated_schema():
    assert _shape(RUST_SCHEMA) == _shape(PYTHON_SCHEMA)
    for name in STRUCT_MODELS:
        assert _shape(RUST_SCHEMA["$defs"][name]) == _shape(
            PYTHON_SCHEMA["$defs"][name]
        ), name
    assert [
        variant["const"]
        for variant in RUST_SCHEMA["$defs"]["KnowledgeScope"]["oneOf"]
    ] == [
        value.value for value in KnowledgeScope
    ]


def test_default_study_evidence_rejects_opponent_private_hand_identity():
    invalid = deepcopy(FIXTURE)
    invalid["landmarks"][0]["frame"]["projection"]["opponent"]["hand"] = [
        _secret_card()
    ]

    # The reusable transitional ExperienceFrame can represent a hand. The
    # StudyArtifact's cross-field privacy validator closes that wider shape.
    RUST_VALIDATOR.validate(invalid)
    with pytest.raises(ValidationError, match="opponent-private hand"):
        StudyArtifact.model_validate(invalid)


def test_default_study_evidence_rejects_rng_secrets_and_binding_drift():
    rng_secret = deepcopy(FIXTURE)
    rng_secret["landmarks"][0]["evidence"]["provenance"]["rng_seed"] = 377
    assert list(RUST_VALIDATOR.iter_errors(rng_secret))
    with pytest.raises(ValidationError):
        StudyArtifact.model_validate(rng_secret)

    prompt_drift = deepcopy(FIXTURE)
    prompt_drift["landmarks"][0]["prompt_id"] = 26
    RUST_VALIDATOR.validate(prompt_drift)
    with pytest.raises(ValidationError, match="viewer, prompt, or offer binding"):
        StudyArtifact.model_validate(prompt_drift)
