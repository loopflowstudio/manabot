use managym::experience::{ProtocolV1ConformanceBundle, ProtocolVersion};
use schemars::schema_for;

const FIXTURE: &str = include_str!("../../protocol/fixtures/bolt-target.json");
const CHECKED_IN_SCHEMA: &str = include_str!("../../protocol/experience-v1.schema.json");

#[test]
fn bolt_target_bundle_round_trips_through_rust_authority_types() {
    let source: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    let bundle: ProtocolV1ConformanceBundle =
        serde_json::from_value(source.clone()).expect("fixture conforms to Rust wire types");

    assert_eq!(bundle.recovery.protocol, ProtocolVersion(1));
    assert_eq!(bundle.recovery.frame.protocol, ProtocolVersion(1));
    assert_eq!(bundle.command.match_id, bundle.recovery.frame.match_id);
    assert_eq!(
        bundle.command.expected_revision,
        bundle.recovery.frame.revision
    );
    assert_eq!(
        Some(bundle.command.prompt_id),
        bundle
            .recovery
            .frame
            .prompt
            .as_ref()
            .map(|prompt| prompt.id)
    );
    assert!(bundle
        .recovery
        .frame
        .offers
        .iter()
        .any(|offer| offer.id == bundle.command.offer_id));

    let round_trip = serde_json::to_value(bundle).expect("bundle serializes");
    assert_eq!(round_trip, source);
}

#[test]
fn unsupported_protocol_versions_fail_at_the_rust_boundary() {
    let invalid = FIXTURE.replacen("\"protocol\": 1", "\"protocol\": 2", 1);
    let error = serde_json::from_str::<ProtocolV1ConformanceBundle>(&invalid)
        .expect_err("protocol 2 must not parse as protocol v1");
    assert!(error
        .to_string()
        .contains("unsupported experience protocol"));
}

#[test]
fn checked_in_schema_is_generated_from_the_rust_authority() {
    let schema = schema_for!(ProtocolV1ConformanceBundle);
    let mut generated = serde_json::to_string_pretty(&schema).expect("schema serializes");
    generated.push('\n');
    assert_eq!(CHECKED_IN_SCHEMA, generated);
}
