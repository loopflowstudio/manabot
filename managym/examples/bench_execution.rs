//! Bounded implementation probe; the full Engine Budget remains the acceptance gate.
//! Identical sources run on reference, published layers and combined execution heads.
//! Normal seeded games and small fixed searches, never artificial host pressure.
#[allow(dead_code)]
mod curated;
#[path = "curated/search.rs"]
mod search;

use managym::{agent::env::Env, benchmark_allocations};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{hint::black_box, time::Instant};

fn main() {
    let mut digest = blake3::Hasher::new();
    let mut steps = 0;
    let mut caps = 0;
    let mut step_ns = 0;
    let mut projection_ns = 0;
    let mut fork_ns = 0;
    let mut projection_allocations = (0_u64, 0_u64);
    let mut fork_allocations = (0_u64, 0_u64);
    let mut samples = Vec::new();
    let mut forks = 0;
    let mut retained_boundaries = 0;
    for seed in 0..32 {
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut count = 0;
        while !env.is_game_over() && count < 500 {
            let actions = env.action_count().unwrap();
            let choice = rng.gen_range(0..actions);
            digest.update(&(actions as u64).to_le_bytes());
            digest.update(&(choice as u64).to_le_bytes());
            let start = Instant::now();
            let _ = black_box(env.step(choice as i64).unwrap());
            let elapsed = start.elapsed().as_nanos();
            step_ns += elapsed;
            if samples.len() < 256 {
                samples.push(elapsed);
            }
            steps += 1;
            count += 1;
            // One representative ordinary decision boundary per game. This
            // isolates actual clone/projection allocation traffic from stepping.
            if count == 40 && !env.is_game_over() {
                benchmark_allocations::start();
                let start = Instant::now();
                let fork = black_box(env.fork().unwrap());
                fork_ns += start.elapsed().as_nanos();
                let (allocations, bytes) = benchmark_allocations::stop();
                fork_allocations.0 += allocations;
                fork_allocations.1 += bytes;
                drop(fork);
                benchmark_allocations::start();
                let start = Instant::now();
                let observation = black_box(
                    env.observation_for_player(env.current_agent_index().unwrap())
                        .unwrap(),
                );
                retained_boundaries += usize::from(
                    observation
                        .continuation
                        .iter()
                        .any(|frame| matches!(frame.kind, 5 | 6)),
                );
                projection_ns += start.elapsed().as_nanos();
                let (allocations, bytes) = benchmark_allocations::stop();
                projection_allocations.0 += allocations;
                projection_allocations.1 += bytes;
                forks += 1;
            }
        }
        caps += usize::from(!env.is_game_over());
        digest.update(&(env.winner_index().unwrap_or(2) as u64).to_le_bytes());
    }
    println!(
        "{}",
        serde_json::json!({"probe":"ordinary", "games":32, "max_steps":500,
        "steps":steps,"caps":caps,"step_ns":step_ns,"step_samples_ns":samples,
        "digest":digest.finalize().to_hex().as_str(),"boundaries":forks,"retained_boundaries":retained_boundaries,
        "fork_ns":fork_ns,"fork_allocations":fork_allocations,
        "projection_ns":projection_ns,"projection_allocations":projection_allocations})
    );
    for seed in [1, 2] {
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        let mut digest = blake3::Hasher::new();
        let mut times = Vec::new();
        let mut counts = Vec::new();
        let mut simulations = 0;
        let mut cap_hits = 0;
        for decision in 0..6 {
            let start = Instant::now();
            let scores = env
                .flat_mc_scores(2, 1, seed * 1_000_003 + decision, 256)
                .unwrap();
            times.push(start.elapsed().as_nanos());
            simulations += scores.simulations;
            cap_hits += scores.cap_hits;
            counts.push(scores.scores.len());
            let best = search::best_action(&scores);
            search::digest_decision(&mut digest, &scores, best);
            env.step(best as i64).unwrap();
            if env.is_game_over() {
                break;
            }
        }
        println!(
            "{}",
            serde_json::json!({"probe":"search", "seed":seed,"worlds":2,"rollouts":1,
            "max_steps":256,"samples_ns":times,"action_counts":counts,"simulations":simulations,
            "cap_hits":cap_hits,"digest":digest.finalize().to_hex().as_str()})
        );
    }
}
