//! Seeded Search-64 reply timing on the Academy curated matchup (UR Lessons
//! versus GW Allies main decks). Both seats play Academy's default flat search
//! (16 worlds x 4 rollouts, 2,000-step playouts) for a fixed number of
//! decisions per game; each line reports one game's worst, median and total
//! reply time. Run in release:
//! `cargo run --release --example bench_search --manifest-path managym/Cargo.toml`.
use managym::{agent::env::Env, state::player::PlayerConfig};
use std::{collections::BTreeMap, time::Instant};

const WORLDS: usize = 16;
const ROLLOUTS: usize = 4;
const MAX_STEPS: usize = 2000;
const DECISIONS: usize = 40;
const SEEDS: [u64; 3] = [1, 2, 3];

fn deck(cards: &[(&str, usize)]) -> BTreeMap<String, usize> {
    cards
        .iter()
        .map(|&(name, n)| (name.to_string(), n))
        .collect()
}

fn players() -> Vec<PlayerConfig> {
    let lessons = deck(&[
        ("Island", 9),
        ("Mountain", 8),
        ("Tiger-Seal", 2),
        ("Otter-Penguin", 2),
        ("Fire Nation Cadets", 2),
        ("First-Time Flyer", 2),
        ("Forecasting Fortune Teller", 1),
        ("Dragonfly Swarm", 1),
        ("Firebending Lesson", 4),
        ("Igneous Inspiration", 2),
        ("Pop Quiz", 2),
        ("Divide by Zero", 2),
        ("It'll Quench Ya!", 2),
        ("Accumulate Wisdom", 2),
    ]);
    let allies = deck(&[
        ("Plains", 9),
        ("Forest", 8),
        ("Water Tribe Rallier", 2),
        ("Invasion Reinforcements", 2),
        ("Compassionate Healer", 2),
        ("Earth Kingdom Jailer", 2),
        ("White Lotus Reinforcements", 2),
        ("Earth King's Lieutenant", 2),
        ("Kyoshi Warriors", 2),
        ("Badgermole Cub", 2),
        ("Suki, Kyoshi Warrior", 1),
        ("South Pole Voyager", 1),
        ("Allies at Last", 2),
        ("Yip Yip!", 1),
        ("Fancy Footwork", 2),
    ]);
    vec![
        PlayerConfig::new("lessons", lessons),
        PlayerConfig::new("allies", allies),
    ]
}

fn main() {
    let mut worst_overall: f64 = 0.0;
    for seed in SEEDS {
        let mut env = Env::new(seed, true, false, false);
        env.reset(players()).unwrap();
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
        println!(
            "{{\"seed\":{seed},\"decisions\":{},\"worst_s\":{worst:.3},\"median_s\":{:.3},\"total_s\":{total:.3},\"digest\":{scores_digest}}}",
            replies.len(),
            sorted[sorted.len() / 2],
        );
    }
    println!("{{\"worst_overall_s\":{worst_overall:.3}}}");
}
