use managym::canonical_replay::{
    CanonicalReplayProjectionV1, CanonicalReplayV1, ReplayDecisionAddress, ViewerPresentationTrack,
};
use managym::study::StudyArtifact;
use schemars::schema_for;

const PLAYER_ZERO: &str = include_str!("../../protocol/fixtures/canonical-replay-player-0.json");
const PLAYER_ONE: &str = include_str!("../../protocol/fixtures/canonical-replay-player-1.json");
const METADATA: &str =
    include_str!("../../protocol/fixtures/canonical-replay-authority-metadata.json");
const STUDY: &str = include_str!("../../protocol/fixtures/study-curated-decision.json");
const CHECKED_IN_SCHEMA: &str = include_str!("../../protocol/canonical-replay-v1.schema.json");

fn projections() -> (CanonicalReplayProjectionV1, CanonicalReplayProjectionV1) {
    let player_zero: CanonicalReplayProjectionV1 =
        serde_json::from_str(PLAYER_ZERO).expect("player-0 projection conforms");
    let player_one: CanonicalReplayProjectionV1 =
        serde_json::from_str(PLAYER_ONE).expect("player-1 projection conforms");
    player_zero
        .validate()
        .expect("player-0 projection validates");
    player_one
        .validate()
        .expect("player-1 projection validates");
    (player_zero, player_one)
}

#[test]
fn safe_projections_interleave_into_one_complete_authority_timeline() {
    let (player_zero, player_one) = projections();
    assert_eq!(player_zero.viewer.0, 0);
    assert_eq!(player_one.viewer.0, 1);
    let metadata: serde_json::Value = serde_json::from_str(METADATA).expect("metadata JSON");
    let expected: Vec<u64> = metadata["decisions"]
        .as_array()
        .expect("metadata decisions")
        .iter()
        .map(|row| row["ordinal"].as_u64().expect("metadata ordinal"))
        .collect();
    let mut actual: Vec<u64> = player_zero
        .decisions
        .iter()
        .chain(&player_one.decisions)
        .map(|row| row.ordinal)
        .collect();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    assert!(player_zero
        .decisions
        .iter()
        .all(|row| row.frame.projection.opponent.hand.is_empty()));
    assert!(player_one
        .decisions
        .iter()
        .all(|row| row.frame.projection.opponent.hand.is_empty()));
}

#[test]
fn safe_projection_rejects_empty_replay_id_and_drifted_identity() {
    let (player_zero, _) = projections();

    let mut empty_replay_id = player_zero.clone();
    empty_replay_id.replay_id.clear();
    assert!(empty_replay_id
        .validate()
        .unwrap_err()
        .contains("requires replay_id"));

    let mut drifted_match = player_zero.clone();
    drifted_match.match_id.0 = "different-match".into();
    assert!(drifted_match
        .validate()
        .unwrap_err()
        .contains("identity differs"));

    let mut drifted_content = player_zero.clone();
    drifted_content.content_hash.0 = "0".repeat(64);
    assert!(drifted_content
        .validate()
        .unwrap_err()
        .contains("identity differs"));

    let mut drifted_assets = player_zero;
    drifted_assets.asset_manifest_hash.0 = "0".repeat(64);
    assert!(drifted_assets
        .validate()
        .unwrap_err()
        .contains("identity differs"));
}

#[test]
fn combined_authority_rejects_duplicate_and_missing_global_addresses() {
    let (player_zero, player_one) = projections();
    let mut decisions = player_zero.decisions.clone();
    decisions.extend(player_one.decisions.clone());
    decisions.sort_by_key(|row| row.ordinal);
    let replay = CanonicalReplayV1 {
        version: player_zero.version,
        replay_id: player_zero.replay_id.clone(),
        match_id: player_zero.match_id.clone(),
        content_hash: player_zero.content_hash.clone(),
        asset_manifest_hash: player_zero.asset_manifest_hash.clone(),
        decisions,
        presentation_tracks: vec![
            ViewerPresentationTrack {
                viewer: player_zero.viewer,
                head: player_zero.presentation_head,
                events: player_zero.presentation.clone(),
            },
            ViewerPresentationTrack {
                viewer: player_one.viewer,
                head: player_one.presentation_head,
                events: player_one.presentation.clone(),
            },
        ],
    };
    replay.validate().expect("combined authority validates");

    let mut duplicate = replay.clone();
    duplicate.decisions[1].ordinal = 0;
    assert!(duplicate.validate().unwrap_err().contains("ordinal gap"));
    let mut missing = replay;
    missing.decisions[1].ordinal = 2;
    assert!(missing.validate().unwrap_err().contains("ordinal gap"));
}

#[test]
fn study_address_is_the_game_owned_player_zero_row_address() {
    let (player_zero, _) = projections();
    let study: StudyArtifact = serde_json::from_str(STUDY).expect("study fixture conforms");
    study.validate().expect("study fixture validates");
    let landmark = &study.landmarks[0];
    let address = ReplayDecisionAddress::parse(&landmark.decision_id)
        .expect("Study carries canonical erd1 address");
    let row = player_zero
        .decisions
        .iter()
        .find(|row| row.ordinal == address.ordinal)
        .expect("address resolves in player-0 projection");
    let expected =
        ReplayDecisionAddress::from_decision(&player_zero, row).expect("row address builds");
    assert_eq!(address, expected);
    assert_eq!(address.encode().unwrap(), landmark.decision_id);
}

#[test]
fn checked_in_schema_is_generated_from_rust_projection_authority() {
    let schema = schema_for!(CanonicalReplayProjectionV1);
    let mut generated = serde_json::to_string_pretty(&schema).expect("schema serializes");
    generated.push('\n');
    assert_eq!(CHECKED_IN_SCHEMA, generated);
}
