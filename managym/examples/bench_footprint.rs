//! Run one cache mode per process. A bank requires an explicit memory budget.
use std::{
    env,
    hint::black_box,
    mem::size_of,
    process::{self, Command},
};

use managym::{
    benchmark_footprint::{inline_sizes, measure_fixture},
    Game,
};
use serde_json::json;

const USAGE: &str = "usage: bench_footprint exact|cold [--bank-mib AVAILABLE_MIB]";
const MIB: u64 = 1024 * 1024;
const MAX_BANK: usize = 10_000;

fn rss_kib() -> Result<u64, String> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &process::id().to_string()])
        .output()
        .map_err(|e| format!("cannot run ps: {e}"))?;
    if !output.status.success() {
        return Err(format!("ps: {}", String::from_utf8_lossy(&output.stderr)));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .map_err(|e| format!("invalid RSS from ps: {e}"))
}

fn admit_bank(midgame_heap: u64, heavy_heap: u64, available_mib: u64) -> Result<(), String> {
    let pair_bytes = 2 * size_of::<Game>() as u64 + midgame_heap + heavy_heap;
    let projected = pair_bytes * (MAX_BANK as u64 / 2);
    if projected >= 1536 * MIB || projected > available_mib.saturating_mul(MIB) {
        return Err(format!(
            "10,000 forks project to {} MiB before allocator overhead; bank skipped",
            projected.div_ceil(MIB)
        ));
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let mode = args.next().ok_or(USAGE)?;
    let cold = match mode.as_str() {
        "exact" => false,
        "cold" => true,
        _ => return Err(USAGE.into()),
    };
    let budget = match args.next().as_deref() {
        None => None,
        Some("--bank-mib") => Some(
            args.next()
                .ok_or(USAGE)?
                .parse::<u64>()
                .map_err(|_| USAGE)?,
        ),
        _ => return Err(USAGE.into()),
    };
    if args.next().is_some() {
        return Err(USAGE.into());
    }
    println!("{}", json!({"mode": mode, "sizes": inline_sizes()}));
    let (midgame, midgame_heap) = measure_fixture("interactive-midgame-48-w4-v1", cold)?;
    let (heavy, heavy_heap) = measure_fixture("interactive-heavy-80-w4-v1", cold)?;
    let Some(budget) = budget else { return Ok(()) };
    admit_bank(midgame_heap, heavy_heap, budget)?;
    let mut bank = Vec::with_capacity(MAX_BANK);
    for count in [0, 1000, MAX_BANK] {
        while bank.len() < count {
            bank.push(midgame.clone());
            bank.push(heavy.clone());
        }
        println!("{}", json!({"bank": count, "rss_kib": rss_kib()?}));
        black_box(&bank);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_admission_includes_both_roots_and_inline_storage() {
        // Exactly 100 KiB per pair, with all heap traffic on the heavy root.
        let heavy_heap = 100 * 1024 - 2 * size_of::<Game>() as u64;
        assert!(admit_bank(0, heavy_heap, 489).is_ok());
        assert!(admit_bank(0, heavy_heap, 488).is_err());
        assert!(admit_bank(0, 1024 * 1024, u64::MAX).is_err());
    }
}
