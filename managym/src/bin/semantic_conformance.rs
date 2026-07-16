// semantic_conformance.rs
// Noninteractive CLI for checked semantic-kernel replay and bounded fuzzing.

use std::{collections::BTreeMap, env, path::PathBuf, process::ExitCode};

use managym::conformance::{check_root, fuzz, record_root, replay_receipt, validate_contract_root};

fn options(arguments: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    let mut index = 0;
    while index < arguments.len() {
        let key = arguments[index]
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --option, got {}", arguments[index]))?;
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("--{key} requires a value"))?;
        if out.insert(key.to_string(), value.clone()).is_some() {
            return Err(format!("--{key} was provided more than once"));
        }
        index += 2;
    }
    Ok(out)
}

fn required(options: &BTreeMap<String, String>, key: &str) -> Result<String, String> {
    options
        .get(key)
        .cloned()
        .ok_or_else(|| format!("missing --{key}"))
}

fn number<T>(options: &BTreeMap<String, String>, key: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    required(options, key)?
        .parse()
        .map_err(|error| format!("invalid --{key}: {error}"))
}

fn run() -> Result<String, String> {
    let mut arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.is_empty() {
        return Err("usage: semantic_conformance <check|record|fuzz|replay> [options]".to_string());
    }
    let command = arguments.remove(0);
    if command == "replay" {
        if arguments.len() != 1 {
            return Err("usage: semantic_conformance replay <receipt.json>".to_string());
        }
        return replay_receipt(&PathBuf::from(&arguments[0]));
    }

    let options = options(&arguments)?;
    let root = PathBuf::from(required(&options, "root")?);
    match command.as_str() {
        "check" => {
            let summary = check_root(&root)?;
            Ok(format!(
                "checked {} replay cases / {} commands and {} Phase rows",
                summary.replay_cases, summary.replay_commands, summary.phase_rows
            ))
        }
        "record" => {
            let summary = record_root(&root)?;
            Ok(format!(
                "recorded {} replay cases / {} commands and {} Phase rows",
                summary.replay_cases, summary.replay_commands, summary.phase_rows
            ))
        }
        "fuzz" => {
            validate_contract_root(&root)?;
            let seed: u64 = number(&options, "seed")?;
            let cases: usize = number(&options, "cases")?;
            let max_commands: usize = number(&options, "max-commands")?;
            let failure_directory = PathBuf::from(required(&options, "failure-dir")?);
            let summary = fuzz(seed, cases, max_commands, &failure_directory)?;
            Ok(format!(
                "fuzzed {} cases / {} commands from seed {}",
                summary.cases, summary.commands, seed
            ))
        }
        other => Err(format!("unknown command {other}")),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("semantic conformance failed: {error}");
            ExitCode::FAILURE
        }
    }
}
