//! Headless steps per second on the Academy curated matchup (UR Lessons versus
//! GW Allies main decks): the training environment's reset/step shape with
//! native trivial-decision skipping, both seats choosing uniformly among the
//! legal actions from a seeded generator outside the engine. Fixed game seeds
//! and a per-game step cap make the workload identical across commits; the
//! digest folds every action count, choice and result, so a rules change
//! shows as a changed digest rather than a quiet throughput move. CI compares
//! the output with `managym/perf/baseline.json` (`scripts/engine_budget.py`).
//! Run in release:
//! `cargo run --release --example bench_steps --manifest-path managym/Cargo.toml`.
mod curated;

use managym::agent::env::Env;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const MAX_STEPS: usize = 2000;

fn main() {
    // Keep the original 512-game budget independently comparable. The added
    // board is an explicitly selected workload with its own count and digest.
    let layers = std::env::args().any(|arg| arg == "--layers");
    let tail_samples = std::env::args().any(|arg| arg == "--tail-samples");
    let mut samples = Vec::new();
    let seeds = if layers { 512..528 } else { 0..512 };
    let games = seeds.end - seeds.start;
    #[cfg(feature = "benchmark")]
    managym::benchmark_layers::reset();
    let mut steps = 0u64;
    let mut digest = 0u64;
    let mut capped = 0u64;
    let start = Instant::now();
    for seed in seeds {
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        if layers {
            curated::add_layer_board(&mut env);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game_steps = 0usize;
        while !env.is_game_over() && game_steps < MAX_STEPS {
            let count = env.action_count().unwrap();
            let action = rng.gen_range(0..count);
            let step_start = (tail_samples && samples.len() < 128).then(Instant::now);
            env.step(action as i64).unwrap();
            if let Some(start) = step_start {
                samples.push(start.elapsed().as_secs_f64());
            }
            game_steps += 1;
            digest = digest
                .wrapping_mul(31)
                .wrapping_add(count as u64 * 7 + action as u64);
        }
        if !env.is_game_over() {
            capped += 1;
        }
        steps += game_steps as u64;
        digest = digest
            .wrapping_mul(31)
            .wrapping_add(env.winner_index().map_or(2, |winner| winner as u64));
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{{\"games\":{games},\"capped\":{capped},\"steps\":{steps},\"elapsed_s\":{elapsed:.6},\"steps_per_s\":{:.1},\"digest\":{digest}}}",
        steps as f64 / elapsed
    );
    if tail_samples {
        println!("{}", serde_json::json!({"step_prefix_samples_s":samples}));
    }
    #[cfg(feature = "benchmark")]
    println!(
        "{}",
        serde_json::json!({"layer_metrics":managym::benchmark_layers::snapshot()})
    );
}
