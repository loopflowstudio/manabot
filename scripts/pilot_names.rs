// Built against the selected writer's engine by pinned_pilot.py.
use managym::{
    agent::record_vocabulary::{record_names, RecordNamespace},
    cards::{
        content_pack::default_content_pack,
        tokens::{definition_tokens, writer_names},
    },
};
use std::collections::BTreeSet;

fn main() {
    let pack = default_content_pack();
    let mut tokens = BTreeSet::new();
    for (_, definition) in pack.definition_entries() {
        tokens.extend(
            definition_tokens(definition, pack)
                .unwrap()
                .into_iter()
                .map(|(kind, value)| (kind as u16, value)),
        );
    }
    for record in record_names() {
        let namespace: u64 = match record.namespace {
            RecordNamespace::Token => 1,
            RecordNamespace::Name => 2,
            RecordNamespace::Restriction => 3,
        };
        let key = (namespace << 32) | u64::from(record.value as u32);
        println!("record_name {key} {}", record.name);
        tokens.extend(record.token);
    }
    let names = writer_names();
    for (kind, value) in tokens {
        if matches!(kind, 1 | 2 | 3 | 6) {
            let key = (u64::from(kind) << 32) | u64::from(value as u32);
            println!("record_token {key} {}", names[&value]);
        }
    }
}
