use std::{env, fs, path::PathBuf};

use managym::experience::ProtocolV1ConformanceBundle;
use schemars::schema_for;

fn main() {
    let output = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: cargo run --example export_experience_protocol -- <output.json>");
    let schema = schema_for!(ProtocolV1ConformanceBundle);
    let mut json = serde_json::to_string_pretty(&schema).expect("schema serializes");
    json.push('\n');
    fs::write(&output, json)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", output.display()));
}
