// semantic_conformance_tests.rs
// Checked replay, property, metamorphic, and Phase evidence for W2-200.

use std::path::Path;

use managym::conformance::{
    check_root, replay_receipt, validate_phase_matrix, verify_metamorphic, verify_properties,
    CaseSpec, Orientation, PhaseMatrix, CONTRACT_ID, PHASE_REVISION,
};

const ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../conformance/semantic-kernel-v1"
);

#[test]
fn checked_semantic_conformance_evidence_reproduces() {
    let summary = check_root(Path::new(ROOT)).expect("checked conformance evidence");
    assert!(summary.replay_cases >= 2);
    assert!(summary.replay_commands > 0);
    assert_eq!(summary.phase_rows, 15);
}

#[test]
fn checked_receipt_replays_through_the_public_handoff() {
    let message = replay_receipt(&Path::new(ROOT).join("replays").join("ur-vs-gw-5eed.json"))
        .expect("checked replay receipt");
    assert!(message.contains("ur-vs-gw-5eed"));
}

#[test]
fn seeded_semantic_tapes_are_deterministic_and_isolated() {
    let spec = CaseSpec {
        id: "property-probe".to_string(),
        orientation: Orientation::UrLessonsVsGwAllies,
        game_seed: 0x5eed,
        choice_seed: 0xc0de,
        max_commands: 96,
    };
    verify_properties(&spec).expect("seeded replay properties");
}

#[test]
fn compatibility_names_and_content_allocations_are_metamorphic() {
    verify_metamorphic(Orientation::GwAlliesVsUrLessons, 0x5eee, 0xc0df, 96)
        .expect("metamorphic trace");
}

#[test]
fn phase_matrix_is_exactly_pinned_and_explicit() {
    let bytes =
        std::fs::read(Path::new(ROOT).join("phase-overlap.json")).expect("Phase overlap matrix");
    let matrix: PhaseMatrix = serde_json::from_slice(&bytes).expect("valid Phase matrix JSON");
    validate_phase_matrix(&matrix).expect("checked Phase matrix");
    assert_eq!(matrix.phase_revision, PHASE_REVISION);
    assert!(!matrix.mismatches.is_empty());
    assert_eq!(CONTRACT_ID, "manabot.semantic-kernel-conformance.v1");
}
