use managym::{
    experience::ProtocolV1ConformanceBundle,
    study::{StudyArtifact, StudyVersion},
};
use schemars::schema_for;

const FIXTURE: &str = include_str!("../../protocol/fixtures/study-curated-decision.json");
const SOURCE_REPLAY: &str = include_str!("../../protocol/fixtures/bolt-target.json");
const CHECKED_IN_SCHEMA: &str = include_str!("../../protocol/study-v1.schema.json");

#[test]
fn study_protocol_shared_fixture_round_trips_and_restores_exact_decision() {
    let fixture_value: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    let artifact: StudyArtifact =
        serde_json::from_value(fixture_value.clone()).expect("fixture conforms to Rust types");
    artifact
        .validate()
        .expect("fixture satisfies study invariants");

    assert_eq!(artifact.version, StudyVersion(1));
    let landmark = &artifact.landmarks[0];
    assert_eq!(artifact.identity.match_id, landmark.frame.match_id.0);
    assert_eq!(landmark.prompt_id.0, 3);
    assert_eq!(landmark.offer_id, landmark.offer.id);
    assert_eq!(landmark.played.offer_id, landmark.offer_id);
    assert_eq!(landmark.alternatives.len(), 2);
    assert_eq!(landmark.evidence.policy_mass.len(), 2);
    assert_eq!(landmark.evidence.search_value.len(), 2);
    assert_eq!(landmark.evidence.visits.len(), 2);
    assert_eq!(landmark.evidence.sampled_world_robustness.len(), 2);
    assert_eq!(landmark.evidence.uncertainty.len(), 2);

    let source_replay: ProtocolV1ConformanceBundle =
        serde_json::from_str(SOURCE_REPLAY).expect("source replay fixture");
    assert_eq!(
        serde_json::to_value(&landmark.frame).unwrap(),
        serde_json::to_value(&source_replay.recovery.frame).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&landmark.played).unwrap(),
        serde_json::to_value(&source_replay.command).unwrap()
    );

    let frame_offer = landmark
        .frame
        .offers
        .iter()
        .find(|offer| offer.id == landmark.offer_id)
        .expect("historical offer remains in the frame");
    assert_eq!(
        serde_json::to_value(frame_offer).unwrap(),
        serde_json::to_value(&landmark.offer).unwrap()
    );
    assert_eq!(serde_json::to_value(artifact).unwrap(), fixture_value);
}

#[test]
fn default_evidence_rejects_opponent_private_facts_and_rng_secrets() {
    let source: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");

    let mut private_hand = source.clone();
    private_hand["landmarks"][0]["frame"]["projection"]["opponent"]["hand"] = serde_json::json!([{
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
    let artifact: StudyArtifact =
        serde_json::from_value(private_hand).expect("legacy frame can represent a hand");
    assert!(artifact
        .validate()
        .expect_err("study boundary must reject private hand identities")
        .contains("opponent-private hand"));

    let mut rng_secret = source;
    rng_secret["landmarks"][0]["evidence"]["provenance"]["rng_seed"] = serde_json::json!(377);
    serde_json::from_value::<StudyArtifact>(rng_secret)
        .expect_err("closed evidence must reject RNG sidecars");
}

#[test]
fn study_protocol_rejects_version_and_identity_drift() {
    let source: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    let mut version = source.clone();
    version["version"] = serde_json::json!(2);
    serde_json::from_value::<StudyArtifact>(version).expect_err("v2 is unsupported");

    let mut prompt = source;
    prompt["landmarks"][0]["prompt_id"] = serde_json::json!(26);
    let artifact: StudyArtifact = serde_json::from_value(prompt).expect("wire shape remains valid");
    assert!(artifact
        .validate()
        .expect_err("prompt drift must fail")
        .contains("viewer, prompt, or offer binding"));
}

#[test]
fn checked_in_study_schema_is_generated_from_rust_authority() {
    let schema = schema_for!(StudyArtifact);
    let mut generated = serde_json::to_string_pretty(&schema).expect("schema serializes");
    generated.push('\n');
    assert_eq!(CHECKED_IN_SCHEMA, generated);
}
