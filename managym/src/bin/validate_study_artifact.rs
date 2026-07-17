// validate_study_artifact.rs
// Validate one serialized Study artifact through the Rust-owned contract.

use std::{env, fs, process};

use managym::study::StudyArtifact;

fn main() {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "validate_study_artifact".to_string());
    let Some(path) = args.next() else {
        eprintln!("usage: {program} <study-artifact.json>");
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <study-artifact.json>");
        process::exit(2);
    }
    let result = fs::read_to_string(&path)
        .map_err(|error| format!("could not read {:?}: {error}", path))
        .and_then(|payload| {
            serde_json::from_str::<StudyArtifact>(&payload)
                .map_err(|error| format!("Study v1 JSON is invalid: {error}"))
        })
        .and_then(|artifact| artifact.validate());
    if let Err(error) = result {
        eprintln!("{error}");
        process::exit(1);
    }
}
