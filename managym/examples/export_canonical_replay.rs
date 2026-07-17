use std::{env, fs, path::PathBuf};

use managym::canonical_replay::CanonicalReplayProjectionV1;
use schemars::schema_for;

fn main() {
    let output = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: cargo run --example export_canonical_replay -- <output.json>");
    let schema = schema_for!(CanonicalReplayProjectionV1);
    let mut json = serde_json::to_string_pretty(&schema).expect("schema serializes");
    json.push('\n');
    fs::write(&output, json)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output.display()));
}
