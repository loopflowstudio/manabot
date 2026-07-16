use std::collections::BTreeSet;

use managym::study::{
    LandmarkReason, RecordedDecisionInput, StudyDecisionIndex, StudyDecisionKind, StudyVersion,
};
use schemars::schema_for;

const INPUT_FIXTURE: &str =
    include_str!("../../protocol/fixtures/recorded-match-decisions-curated.json");
const INDEX_FIXTURE: &str =
    include_str!("../../protocol/fixtures/study-decision-index-curated.json");
const INPUT_SCHEMA: &str = include_str!("../../protocol/study-recorded-decisions-v1.schema.json");
const INDEX_SCHEMA: &str = include_str!("../../protocol/study-index-v1.schema.json");

#[test]
fn shared_fixtures_preserve_every_exact_recorded_decision() {
    let input: RecordedDecisionInput =
        serde_json::from_str(INPUT_FIXTURE).expect("input fixture conforms to Rust types");
    input.validate().expect("input invariants");
    let index: StudyDecisionIndex =
        serde_json::from_str(INDEX_FIXTURE).expect("index fixture conforms to Rust types");
    index.validate().expect("index invariants");

    assert_eq!(input.version, StudyVersion(1));
    assert_eq!(index.version, StudyVersion(1));
    assert_eq!(input.decision_count, 8);
    assert_eq!(index.decisions.len(), 8);
    assert_eq!(index.landmarks.len(), 5);

    let mut ids = BTreeSet::new();
    for (ordinal, (source, decision)) in input
        .decisions
        .iter()
        .zip(index.decisions.iter())
        .enumerate()
    {
        assert_eq!(decision.ordinal as usize, ordinal);
        assert_eq!(decision.event_cursor, source.event_cursor);
        assert!(ids.insert(decision.id.clone()));
        assert_eq!(
            serde_json::to_value(&decision.frame).unwrap(),
            serde_json::to_value(&source.frame).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&decision.offer).unwrap(),
            serde_json::to_value(&source.offer).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&decision.played).unwrap(),
            serde_json::to_value(&source.played).unwrap()
        );
    }
}

#[test]
fn landmarks_are_ranked_recommendations_not_navigation_gates() {
    let index: StudyDecisionIndex = serde_json::from_str(INDEX_FIXTURE).expect("index fixture");
    index.validate().expect("index invariants");

    assert_eq!(index.decisions[0].kind, StudyDecisionKind::Priority);
    assert_eq!(index.decisions[2].kind, StudyDecisionKind::Targeting);
    assert_eq!(index.decisions[3].kind, StudyDecisionKind::Attack);
    assert_eq!(index.decisions[4].kind, StudyDecisionKind::Attack);
    assert_eq!(index.decisions[5].kind, StudyDecisionKind::Block);

    let recommended = index
        .landmarks
        .iter()
        .map(|landmark| landmark.decision_id.clone())
        .collect::<BTreeSet<_>>();
    assert!(recommended.contains(&index.decisions[3].id));
    assert!(!recommended.contains(&index.decisions[4].id));
    assert!(!recommended.contains(&index.decisions[6].id));
    assert!(!recommended.contains(&index.decisions[7].id));
    assert!(index.decisions[6].automatic);
    assert_eq!(index.decisions[7].frame.offers.len(), 1);

    for (offset, landmark) in index.landmarks.iter().enumerate() {
        assert_eq!(usize::from(landmark.rank), offset + 1);
    }
    assert!(index.landmarks.iter().any(|landmark| {
        landmark
            .reasons
            .contains(&LandmarkReason::PriorityCommitment)
    }));
    assert!(index
        .landmarks
        .iter()
        .any(|landmark| { landmark.reasons.contains(&LandmarkReason::PriorityResponse) }));
    assert!(index
        .landmarks
        .iter()
        .any(|landmark| { landmark.reasons.contains(&LandmarkReason::TargetSelection) }));
    assert!(index.landmarks.iter().any(|landmark| {
        landmark
            .reasons
            .contains(&LandmarkReason::AttackDeclaration)
    }));
    assert!(index
        .landmarks
        .iter()
        .any(|landmark| { landmark.reasons.contains(&LandmarkReason::BlockDeclaration) }));
}

#[test]
fn closed_input_rejects_private_sidecars_and_binding_drift() {
    let source: serde_json::Value = serde_json::from_str(INPUT_FIXTURE).expect("input JSON");

    let mut private_hand = source.clone();
    private_hand["decisions"][0]["frame"]["projection"]["opponent"]["hand"] = serde_json::json!([{
        "id": 99,
        "registry_key": 99,
        "name": "Secret Counterspell",
        "zone": "HAND",
        "owner_id": 0,
        "power": 0,
        "toughness": 0,
        "mana_value": 2,
        "types": {
            "is_creature": false,
            "is_land": false,
            "is_spell": true,
            "is_artifact": false,
            "is_enchantment": false,
            "is_planeswalker": false,
            "is_battle": false
        }
    }]);
    let input: RecordedDecisionInput =
        serde_json::from_value(private_hand).expect("frame shape permits a hand");
    assert!(input
        .validate()
        .expect_err("Study must reject private hand identities")
        .contains("opponent-private hand"));

    let mut rng_sidecar = source.clone();
    rng_sidecar["rng_seed"] = serde_json::json!(377);
    serde_json::from_value::<RecordedDecisionInput>(rng_sidecar)
        .expect_err("closed root must reject RNG sidecars");

    let mut cursor_drift = source;
    cursor_drift["decisions"][1]["event_cursor"] =
        cursor_drift["decisions"][0]["event_cursor"].clone();
    let input: RecordedDecisionInput =
        serde_json::from_value(cursor_drift).expect("wire shape remains valid");
    assert!(input
        .validate()
        .expect_err("duplicate cursor must fail")
        .contains("strictly increase"));
}

#[test]
fn checked_in_index_schemas_are_generated_from_rust_authority() {
    let mut input =
        serde_json::to_string_pretty(&schema_for!(RecordedDecisionInput)).expect("input schema");
    input.push('\n');
    assert_eq!(INPUT_SCHEMA, input);

    let mut index =
        serde_json::to_string_pretty(&schema_for!(StudyDecisionIndex)).expect("index schema");
    index.push('\n');
    assert_eq!(INDEX_SCHEMA, index);
}
