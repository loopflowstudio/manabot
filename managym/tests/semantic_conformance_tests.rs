// semantic_conformance_tests.rs
// Checked replay, property, metamorphic, and Phase evidence for the current rules.

use std::path::Path;

use managym::conformance::{
    check_root, replay_receipt, validate_phase_matrix, verify_metamorphic, verify_properties,
    CaseSpec, Orientation, PhaseMatrix, CONTRACT_ID, PHASE_REVISION,
};

const ROOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../conformance/semantic-kernel-w12-v1"
