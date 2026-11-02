//! Seeded Search-64 reply timing on the Academy curated matchup (UR Lessons
//! versus GW Allies main decks). Both seats play Academy's default flat search
//! (16 worlds x 4 rollouts, 2,000-step playouts) for a fixed number of
//! decisions per game; each line reports one game's worst, median and total
//! reply time and a digest of the search results, and the last line sums the
//! games. CI compares the output with `managym/perf/baseline.json`
//! (`scripts/engine_budget.py`). Run in release:
//! `cargo run --release --example bench_search --manifest-path managym/Cargo.toml`.
mod curated;

use managym::agent::env::Env;
use std::time::Instant;

const WORLDS: usize = 16;
const ROLLOUTS: usize = 4;
const MAX_STEPS: usize = 2000;
const DECISIONS: usize = 40;
const SEEDS: [u64; 3] = [1, 2, 3];

fn main() {
    let mut worst_overall: f64 = 0.0;
    let mut total_overall: f64 = 0.0;
    for seed in SEEDS {
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        let mut replies = Vec::new();
        let mut scores_digest = 0u64;
        for decision in 0..DECISIONS {
            let start = Instant::now();
            let result = env
                .flat_mc_scores(
                    WORLDS,
                    ROLLOUTS,
                    seed * 1_000_003 + decision as u64,
                    MAX_STEPS,
                )
                .unwrap();
            replies.push(start.elapsed().as_secs_f64());
            let best = result
                .scores
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |best, (i, &s)| {
                    if s > best.1 {
                        (i, s)
                    } else {
                        best
                    }
                })
                .0;
            scores_digest = scores_digest
                .wrapping_mul(31)
                .wrapping_add(result.simulations + best as u64);
            let (_, _, terminated, truncated, _) = env.step(best as i64).unwrap();
            if terminated || truncated {
                break;
            }
        }
        let total: f64 = replies.iter().sum();
        let mut sorted = replies.clone();
        sorted.sort_by(f64::total_cmp);
        let worst = *sorted.last().unwrap();
        worst_overall = worst_overall.max(worst);
        total_overall += total;
        println!(
            "{{\"seed\":{seed},\"decisions\":{},\"worst_s\":{worst:.3},\"median_s\":{:.3},\"total_s\":{total:.3},\"digest\":{scores_digest}}}",
            replies.len(),
            sorted[sorted.len() / 2],
        );
    }
    println!("{{\"worst_overall_s\":{worst_overall:.3},\"total_overall_s\":{total_overall:.3}}}");
}
