// Built against the selected writer's engine by pinned_pilot.py.
#[cfg(not(feature = "pilot-runtime-tokens"))]
use managym::cards::tokens::definition_tokens;
use managym::{
    agent::record_vocabulary::{record_names, RecordNamespace},
    cards::{content_pack::default_content_pack, tokens::writer_names},
};
#[cfg(feature = "pilot-runtime-tokens")]
use managym::{
    agent::{decision_memory::decision_memory_names, record_vocabulary::runtime_tokens},
    cards::tokens::catalog_tokens,
};
use std::collections::BTreeSet;

fn main() {
    let pack = default_content_pack();
    #[cfg(feature = "pilot-runtime-tokens")]
    let tokens: BTreeSet<_> = {
        let catalog = catalog_tokens(pack).unwrap();
        catalog
            .token_kind
            .into_iter()
            .zip(catalog.token_value)
            .chain(
                runtime_tokens()
                    .into_iter()
                    .map(|token| (token.kind as u16, token.value)),
            )
            .collect()
    };
    #[cfg(not(feature = "pilot-runtime-tokens"))]
    let mut tokens: BTreeSet<_> = pack
        .definition_entries()
        .flat_map(|(_, definition)| definition_tokens(definition, pack).unwrap())
        .map(|(kind, value)| (kind as u16, value))
        .collect();

    for record in record_names() {
        print_record(record.namespace, &record.name, record.value);
        #[cfg(not(feature = "pilot-runtime-tokens"))]
        tokens.extend(record.token);
    }
    #[cfg(feature = "pilot-runtime-tokens")]
    for (name, value) in decision_memory_names() {
        print_record(RecordNamespace::Token, &name, value);
    }
    let names = writer_names();
    for (kind, value) in tokens {
        if matches!(kind, 1 | 2 | 3 | 6) {
            let key = (u64::from(kind) << 32) | u64::from(value as u32);
            println!("record_token {key} {}", names[&value]);
        }
    }
}

fn print_record(namespace: RecordNamespace, name: &str, value: i32) {
    let namespace: u64 = match namespace {
        RecordNamespace::Token => 1,
        RecordNamespace::Name => 2,
        RecordNamespace::Restriction => 3,
    };
    let key = (namespace << 32) | u64::from(value as u32);
    println!("record_name {key} {name}");
}
