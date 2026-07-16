use managym::experience::{PresentationKind, ProtocolV1ConformanceBundle, ProtocolVersion};
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
    assert!(matches!(
        bundle.recovery.presentation_tail.as_slice(),
        [
            managym::experience::PresentationEvent {
                kind: PresentationKind::Cast { .. },
                ..
            },
            managym::experience::PresentationEvent {
                kind: PresentationKind::Targeted { .. },
                ..
            },
            managym::experience::PresentationEvent {
                kind: PresentationKind::Resolved { .. },
                ..
            },
            managym::experience::PresentationEvent {
                kind: PresentationKind::Damage { .. },
                ..
            },
            managym::experience::PresentationEvent {
                kind: PresentationKind::Destroyed { .. },
                ..
            },
            managym::experience::PresentationEvent {
                kind: PresentationKind::Died { .. },
                ..
            }
        ]
    ));

    let round_trip = serde_json::to_value(bundle).expect("bundle serializes");
    assert_eq!(round_trip, source);
}

#[test]
fn curated_asset_pack_reference_round_trips_through_rust_authority_types() {
    let mut source: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    source["recovery"]["frame"]["asset_pack"] = serde_json::json!({
        "id": "tla-ur-lessons-vs-gw-allies",
        "version": "1",
        "manifest_sha256": "f5816a2792a67a79dc7e1f02e1d71c6296e56537cacd25f24328c7d6104ee787"
    });

    let bundle: ProtocolV1ConformanceBundle =
        serde_json::from_value(source.clone()).expect("asset pack reference conforms");
    let asset_pack = bundle
        .recovery
        .frame
        .asset_pack
        .as_ref()
        .expect("curated frame retains its pack reference");
    assert_eq!(asset_pack.id, "tla-ur-lessons-vs-gw-allies");
    assert_eq!(asset_pack.version, "1");

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
fn required_nullable_fields_and_closed_presentation_kinds_are_enforced() {
    let source: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    for (parent_pointer, field) in [
        ("/recovery", "checkpoint"),
        ("/recovery/frame", "prompt"),
        ("/recovery/frame", "winner"),
        ("/recovery/frame/offers/0", "source"),
        ("/recovery/frame/offers/0", "help"),
        ("/recovery/presentation_tail/0", "caused_by"),
        ("/recovery/presentation_tail/0", "sound"),
        ("/recovery/presentation_tail/3/kind", "source"),
    ] {
        let mut invalid = source.clone();
        invalid
            .pointer_mut(parent_pointer)
            .and_then(serde_json::Value::as_object_mut)
            .expect("fixture parent object")
            .remove(field);
        assert!(
            serde_json::from_value::<ProtocolV1ConformanceBundle>(invalid).is_err(),
            "missing {parent_pointer}/{field} must fail"
        );
    }

    let mut unknown = source;
    unknown["recovery"]["presentation_tail"][0]["kind"]["client_only"] = serde_json::json!(true);
    serde_json::from_value::<ProtocolV1ConformanceBundle>(unknown).unwrap_err();
}

#[test]
fn checked_in_schema_is_generated_from_the_rust_authority() {
    let schema = schema_for!(ProtocolV1ConformanceBundle);
    let mut generated = serde_json::to_string_pretty(&schema).expect("schema serializes");
    generated.push('\n');
    assert_eq!(CHECKED_IN_SCHEMA, generated);
}
