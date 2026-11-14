// semantic_conformance_tests.rs
// Checked replay, property, metamorphic, and Phase evidence for the current rules.

use std::{collections::BTreeSet, path::Path};

use managym::conformance::{
    generate_case, replay_case, replay_receipt, validate_contract_root, validate_phase_matrix,
    verify_metamorphic, verify_properties, CaseSpec, Evidence, Manifest, Orientation, PhaseMatrix,
    CONTRACT_ID, PHASE_REVISION,
};

const ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../conformance/semantic-kernel-w17-v3"
);

// Each frozen game owns its regeneration, replay and 300-second timeout.
// Its digest pins every checkpoint and command in the generated trace.
macro_rules! frozen_cases {
    ($($test:ident => $id:literal),+ $(,)?) => {
        const FROZEN_CASE_IDS: &[&str] = &[$($id),+];
        $(#[test]
        fn $test() {
            check_frozen_case($id);
        })+
    };
}

frozen_cases! {
    frozen_ur_vs_gw_5eed => "ur-vs-gw-5eed",
    frozen_gw_vs_ur_5eee => "gw-vs-ur-5eee",
    frozen_ur_vs_gw_5eef => "ur-vs-gw-5eef",
    frozen_gw_vs_ur_5ef0 => "gw-vs-ur-5ef0",
    frozen_elemental_161 => "elemental-161",
    frozen_elemental_162 => "elemental-162",
    frozen_elemental_163 => "elemental-163",
    frozen_elemental_164 => "elemental-164",
    frozen_sacrifice_162 => "sacrifice-162",
    frozen_sacrifice_163 => "sacrifice-163",
    frozen_temur_163 => "temur-163",
    frozen_temur_164 => "temur-164",
    frozen_elemental_world_gr_vs_ug => "elemental-world-gr_vs_ug",
    frozen_elemental_world_ug_vs_gr => "elemental-world-ug_vs_gr",
}

fn manifest() -> Manifest {
    serde_json::from_slice(include_bytes!(
        "../../conformance/semantic-kernel-w17-v3/manifest.json"
    ))
    .expect("pinned manifest")
}

fn evidence() -> Evidence {
    serde_json::from_slice(include_bytes!(
        "../../conformance/semantic-kernel-w17-v3/evidence.json"
    ))
    .expect("pinned evidence")
}

fn check_frozen_case(id: &str) {
    let manifest = manifest();
    let spec = manifest.cases.iter().find(|case| case.id == id).unwrap();
    let case = generate_case(spec).expect("regenerated frozen case");
    let evidence = evidence();
    let expected = evidence
        .replays
        .iter()
        .find(|case| case.case_id == id)
        .unwrap();
    // The historical trace_sha256 name holds BLAKE3 over the entire case:
    // every decision checkpoint, command, event boundary and state hash.
    let bytes = serde_json::to_vec(&case).unwrap();
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        expected.trace_sha256,
        "{id}"
    );
    assert_eq!(
        case.final_checkpoint.state_hash, expected.final_state_hash,
        "{id}"
    );
    replay_case(&case).expect("frozen case replay");
}

#[test]
fn frozen_case_inventory_covers_the_complete_manifest_and_evidence() {
    validate_contract_root(Path::new(ROOT)).expect("conformance contract");
    let manifest = manifest();
    let evidence = evidence();
    let tested: BTreeSet<_> = FROZEN_CASE_IDS.iter().copied().collect();
    let declared: BTreeSet<_> = manifest.cases.iter().map(|case| case.id.as_str()).collect();
    let recorded: BTreeSet<_> = evidence
        .replays
        .iter()
        .map(|case| case.case_id.as_str())
        .collect();
    assert_eq!(tested.len(), FROZEN_CASE_IDS.len());
    assert_eq!(declared.len(), manifest.cases.len());
    assert_eq!(recorded.len(), evidence.replays.len());
    assert_eq!(tested, declared);
    assert_eq!(tested, recorded);
    assert_eq!(
        FROZEN_CASE_IDS,
        manifest
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        FROZEN_CASE_IDS,
        evidence
            .replays
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(manifest.fuzz, evidence.fuzz);
    assert_eq!(evidence.schema_version, manifest.schema_version);
    assert_eq!(evidence.contract_id, manifest.contract_id);
    assert_eq!(
        evidence.content_digest,
        managym::cards::content_pack::default_content_pack().content_digest()
    );
}

#[test]
fn regenerated_case_replays_through_the_public_handoff() {
    let spec = CaseSpec {
        id: "ur-vs-gw-5eed".to_string(),
        orientation: Orientation::UrLessonsVsGwAllies,
        game_seed: 24301,
        choice_seed: 49374,
        max_commands: 2000,
    };
    let case = generate_case(&spec).expect("generated case");
    let path = std::env::temp_dir().join(format!(
        "semantic-conformance-handoff-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, serde_json::to_vec(&case).expect("case JSON")).expect("write case");
    let message = replay_receipt(&path).expect("regenerated replay receipt");
    std::fs::remove_file(&path).expect("remove case");
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

#[test]
fn compiled_elemental_tapes_preserve_seeded_replay_and_exact_clone_isolation() {
    for orientation in [
        Orientation::ElementalLeftVsRight,
        Orientation::ElementalRightVsLeft,
    ] {
        let spec = CaseSpec {
            id: "elemental-property-probe".into(),
            orientation,
            game_seed: 161,
            choice_seed: 16100,
            max_commands: 96,
        };
        verify_properties(&spec).expect("compiled Elemental replay and clone contracts");
    }
}
