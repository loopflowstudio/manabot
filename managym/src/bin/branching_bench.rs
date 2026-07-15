use std::{
    env,
    io::{self, Read, Write},
    process,
};

use managym::benchmark::{
    build_fixture, equivalence_check, manifest, prepare_worker, WorkerRequest,
};
use serde_json::json;

fn argument(args: &[String], flag: &str) -> Result<String, String> {
    let index = args
        .iter()
        .position(|value| value == flag)
        .ok_or_else(|| format!("missing {flag}"))?;
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn run() -> Result<(), String> {
    let invoked_argv: Vec<String> = env::args().collect();
    let args: Vec<String> = invoked_argv.iter().skip(1).cloned().collect();
    if args.iter().any(|value| value == "--manifest") {
        println!(
            "{}",
            serde_json::to_string(&manifest()).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    if args.iter().any(|value| value == "--fixture") {
        let fixture_id = argument(&args, "--fixture")?;
        let (_, summary) = build_fixture(&fixture_id)?;
        println!(
            "{}",
            serde_json::to_string(&summary).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    if args.iter().any(|value| value == "--equivalence") {
        let fixture_id = argument(&args, "--equivalence")?;
        let seed = argument(&args, "--seed")?
            .parse::<u64>()
            .map_err(|error| format!("invalid --seed: {error}"))?;
        let max_steps = argument(&args, "--max-steps")?
            .parse::<usize>()
            .map_err(|error| format!("invalid --max-steps: {error}"))?;
        let receipt = equivalence_check(&fixture_id, seed, max_steps)?;
        println!(
            "{}",
            serde_json::to_string(&receipt).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    let request_json = argument(&args, "--request-json")?;
    let request: WorkerRequest =
        serde_json::from_str(&request_json).map_err(|error| format!("invalid request: {error}"))?;
    let prepared = prepare_worker(request)?;

    println!(
        "{}",
        json!({"type": "ready", "pid": process::id(), "argv": invoked_argv})
    );
    io::stdout().flush().map_err(|error| error.to_string())?;
    let mut release = [0_u8; 1];
    io::stdin()
        .read_exact(&mut release)
        .map_err(|error| format!("start barrier failed: {error}"))?;

    let result = prepared.measure()?;
    println!(
        "{}",
        serde_json::to_string(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("branching_bench: {error}");
        process::exit(2);
    }
}
