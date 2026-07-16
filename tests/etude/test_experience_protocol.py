"""Cross-language conformance for the canonical experience protocol-v1 schema."""

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest

from etude.experience_protocol import ProtocolV1ConformanceBundle

PROTOCOL_DIR = Path(__file__).parents[2] / "protocol"
FIXTURE = json.loads(
    (PROTOCOL_DIR / "fixtures" / "bolt-target.json").read_text(encoding="utf-8")
)
RUST_SCHEMA = json.loads(
    (PROTOCOL_DIR / "experience-v1.schema.json").read_text(encoding="utf-8")
)
RUST_VALIDATOR = Draft202012Validator(RUST_SCHEMA)
PYTHON_SCHEMA = ProtocolV1ConformanceBundle.model_json_schema()

STRUCT_MODELS = (
    "AssetPackReference",
    "Candidate",
    "CandidateSource",
    "Command",
    "CommandReceipt",
    "DeckNames",
    "ExperienceFrame",
    "InteractionOffer",
    "LegacyCardTypesView",
    "LegacyCardView",
    "LegacyHeroObservation",
    "LegacyPermanentView",
    "LegacyPlayerView",
    "LegacyTurnView",
    "ObjectRenderId",
    "PresentationEvent",
    "PromptView",
    "RecoveryEnvelope",
    "StopsConfig",
)

ENUM_MODELS = (
    "AuthorityStatus",
    "OfferVerb",
    "PresentationImportance",
    "RecoveryReason",
)

UNION_VARIANTS = {
    "SubjectRef": ("ObjectSubject", "StackSubject", "PlayerSubject"),
    "CandidateValue": (
        "SubjectCandidateValue",
        "ModeCandidateValue",
        "PaymentPlanCandidateValue",
        "BooleanCandidateValue",
    ),
    "ChoiceStep": (
        "SelectChoiceStep",
        "NumberChoiceStep",
        "AssignChoiceStep",
        "OrderChoiceStep",
        "PaymentChoiceStep",
    ),
    "ChoiceAnswer": (
        "CandidatesChoiceAnswer",
        "NumberChoiceAnswer",
        "AssignmentsChoiceAnswer",
        "OrderChoiceAnswer",
        "PaymentChoiceAnswer",
    ),
    "PresentationKind": (
        "CastPresentation",
        "TargetedPresentation",
        "ResolvedPresentation",
        "DamagePresentation",
        "DestroyedPresentation",
        "DiedPresentation",
        "AttackGroupPresentation",
        "BlockedPresentation",
        "TurnStartedPresentation",
    ),
}


def _shape(schema: dict) -> tuple[set[str], set[str]]:
    return set(schema.get("properties", {})), set(schema.get("required", []))


def _tagged_variants(schema: dict) -> dict[str, dict]:
    variants = {}
    for variant in schema["oneOf"]:
        tag = variant["properties"]["kind"]["const"]
        variants[tag] = variant
    return variants


def _python_variants(names: tuple[str, ...]) -> dict[str, dict]:
    variants = {}
    for name in names:
        variant = PYTHON_SCHEMA["$defs"][name]
        tag = variant["properties"]["kind"]["const"]
        variants[tag] = variant
    return variants


def _assert_both_reject(value: dict) -> None:
    assert list(RUST_VALIDATOR.iter_errors(value))
    with pytest.raises(ValidationError):
        ProtocolV1ConformanceBundle.model_validate(value)


def test_python_representation_round_trips_shared_non_empty_fixture():
    Draft202012Validator.check_schema(RUST_SCHEMA)
    RUST_VALIDATOR.validate(FIXTURE)

    bundle = ProtocolV1ConformanceBundle.model_validate(FIXTURE)
    assert bundle.model_dump(mode="json", exclude_unset=True) == FIXTURE
    assert bundle.recovery.presentation_cursor == 900
    assert [event.seq for event in bundle.recovery.presentation_tail] == list(
        range(900, 906)
    )
    assert [event.kind.kind for event in bundle.recovery.presentation_tail] == [
        "cast",
        "targeted",
        "resolved",
        "damage",
        "destroyed",
        "died",
    ]
    assert {event.importance.value for event in bundle.recovery.presentation_tail} == {
        "ambient",
        "normal",
        "emphasized",
        "critical",
    }


def test_python_fields_and_requiredness_match_rust_generated_schema():
    rust_defs = RUST_SCHEMA["$defs"]
    python_defs = PYTHON_SCHEMA["$defs"]

    assert _shape(RUST_SCHEMA) == _shape(PYTHON_SCHEMA)
    for name in STRUCT_MODELS:
        assert _shape(rust_defs[name]) == _shape(python_defs[name]), name

    for rust_name, python_names in UNION_VARIANTS.items():
        rust_variants = _tagged_variants(rust_defs[rust_name])
        python_variants = _python_variants(python_names)
        assert rust_variants.keys() == python_variants.keys(), rust_name
        for tag in rust_variants:
            assert _shape(rust_variants[tag]) == _shape(python_variants[tag]), (
                rust_name,
                tag,
            )


def test_python_enums_match_rust_generated_schema():
    for name in ENUM_MODELS:
        assert (
            RUST_SCHEMA["$defs"][name]["enum"] == PYTHON_SCHEMA["$defs"][name]["enum"]
        ), name


@pytest.mark.parametrize(
    ("container", "field"),
    (
        (("recovery",), "checkpoint"),
        (("recovery", "frame"), "prompt"),
        (("recovery", "frame"), "winner"),
        (("recovery", "frame", "offers", 0), "source"),
        (("recovery", "frame", "offers", 0), "help"),
        (("recovery", "presentation_tail", 0), "caused_by"),
        (("recovery", "presentation_tail", 0), "sound"),
        (("recovery", "presentation_tail", 3, "kind"), "source"),
    ),
)
def test_required_nullable_fields_cannot_be_omitted(container, field):
    invalid = deepcopy(FIXTURE)
    target = invalid
    for part in container:
        target = target[part]
    del target[field]
    _assert_both_reject(invalid)


def test_recovery_presentation_cursor_is_required():
    invalid = deepcopy(FIXTURE)
    del invalid["recovery"]["presentation_cursor"]
    _assert_both_reject(invalid)


def test_optional_fields_may_be_absent_or_round_trip_when_present():
    enriched = deepcopy(FIXTURE)
    frame = enriched["recovery"]["frame"]
    frame.update(
        {
            "deck_names": {"hero": "UR Lessons", "villain": "GW Allies"},
            "asset_pack": {
                "id": "tla-ur-lessons-vs-gw-allies",
                "version": "1",
                "manifest_sha256": "fixture-manifest",
            },
            "log": ["Hero casts Lightning Bolt"],
            "auto_passed": 2,
        }
    )

    RUST_VALIDATOR.validate(enriched)
    bundle = ProtocolV1ConformanceBundle.model_validate(enriched)
    assert bundle.model_dump(mode="json", exclude_unset=True) == enriched


def test_version_unknown_field_and_presentation_discriminant_drift_are_rejected():
    invalid_version = deepcopy(FIXTURE)
    invalid_version["recovery"]["protocol"] = 2
    _assert_both_reject(invalid_version)

    unknown_field = deepcopy(FIXTURE)
    unknown_field["command"]["action_index"] = 1
    _assert_both_reject(unknown_field)

    unknown_presentation = deepcopy(FIXTURE)
    unknown_presentation["recovery"]["presentation_tail"][0]["kind"]["kind"] = "opaque"
    _assert_both_reject(unknown_presentation)
