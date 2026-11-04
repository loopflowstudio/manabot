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

const GAMES: u64 = 512;
const MAX_STEPS: usize = 2000;

fn main() {
    // An optional game count permits bounded same-runner attribution. CI
    // omits it and retains the complete frozen workload.
    let games = std::env::args()
        .nth(1)
        .map_or(GAMES, |s| s.parse().expect("game count"));
    assert!(games > 0);
    let mut game_seconds = Vec::new();
    let mut steps = 0u64;
    let mut digest = 0u64;
    let mut capped = 0u64;
    let start = Instant::now();
    for seed in 0..games {
        let game_start = Instant::now();
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game_steps = 0usize;
        while !env.is_game_over() && game_steps < MAX_STEPS {
            let count = env.action_count().unwrap();
            let action = rng.gen_range(0..count);
            env.step(action as i64).unwrap();
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
        game_seconds.push(game_start.elapsed().as_secs_f64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{{\"games\":{games},\"capped\":{capped},\"steps\":{steps},\"elapsed_s\":{elapsed:.3},\"steps_per_s\":{:.1},\"digest\":{digest},\"game_seconds\":{game_seconds:?}}}",
        steps as f64 / elapsed
    );
}
