//! Seeded Search-64 reply timing on the Academy curated matchup (UR Lessons
//! versus GW Allies main decks). Both seats play Academy's default flat search
//! (16 worlds x 4 rollouts, 2,000-step playouts) for a fixed number of
//! decisions per game; each line reports one game's worst, median and total
//! reply time and a digest of the search results, and the last line sums the
//! games. CI compares the output with `managym/perf/baseline.json`
//! (`scripts/engine_budget.py`). Run in release:
//! `cargo run --release --example bench_search --manifest-path managym/Cargo.toml`.
mod curated;
#[path = "curated/search.rs"]
mod search;

use search::{best_action, digest_decision};

use managym::agent::env::Env;
use std::time::Instant;

const WORLDS: usize = 16;
const ROLLOUTS: usize = 4;
const MAX_STEPS: usize = 2000;
const DECISIONS: usize = 40;
const SEEDS: [u64; 3] = [1, 2, 3];

fn scores_digest(digest: &blake3::Hasher) -> u64 {
    // Keep the budget's integer wire format, using the first 64 BLAKE3 bits.
    u64::from_le_bytes(digest.finalize().as_bytes()[..8].try_into().unwrap())
}

fn main() {
    let layers = std::env::args().any(|arg| arg == "--layers");
    let mut worst_overall: f64 = 0.0;
    let mut total_overall: f64 = 0.0;
    for &seed in if layers { &[244][..] } else { &SEEDS[..] } {
        #[cfg(feature = "benchmark")]
        managym::benchmark_layers::reset();
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        if layers {
            curated::add_layer_board(&mut env);
        }
        let mut replies = Vec::new();
        let mut digest = blake3::Hasher::new();
        let mut simulations = 0;
        let mut cap_hits = 0;
        let mut action_counts = Vec::new();
        for decision in 0..if layers { 8 } else { DECISIONS } {
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
            simulations += result.simulations;
            cap_hits += result.cap_hits;
            action_counts.push(result.scores.len());
            let best = best_action(&result);
            digest_decision(&mut digest, &result, best);
            let (_, _, terminated, truncated, _) = env.step(best as i64).unwrap();
            if terminated || truncated {
                break;
            }
        }
        let total: f64 = replies.iter().sum();
        let samples = replies.clone();
        replies.sort_by(f64::total_cmp);
        let worst = *replies.last().unwrap();
        worst_overall = worst_overall.max(worst);
        total_overall += total;
        let scores_digest = scores_digest(&digest);
        println!(
            "{}",
            serde_json::json!({
                "seed": seed, "decisions": replies.len(), "worst_s": worst,
                "median_s": replies[replies.len() / 2], "total_s": total,
                "digest": scores_digest, "score_digest_blake3": digest.finalize().to_hex().as_str(),
                "simulations": simulations, "cap_hits": cap_hits,
                "action_counts": action_counts, "samples_s": samples,
            })
        );
        #[cfg(feature = "benchmark")]
        println!(
            "{}",
            serde_json::json!({"attribution_seed":seed, "layer_metrics":managym::benchmark_layers::snapshot()})
        );
    }
    println!("{{\"worst_overall_s\":{worst_overall:.3},\"total_overall_s\":{total_overall:.3}}}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use managym::agent::env::FlatMcResult;

    fn result(scores: &[f64]) -> FlatMcResult {
        FlatMcResult {
            scores: scores.to_vec(),
            simulations: scores.len() as u64 * 64,
            cap_hits: 0,
        }
    }

    fn digest(results: &[FlatMcResult]) -> u64 {
        let mut digest = blake3::Hasher::new();
        for result in results {
            // All fixtures deliberately retain action zero as the winner.
            assert!(result.scores.iter().all(|score| *score <= result.scores[0]));
            digest_decision(&mut digest, result, 0);
        }
        scores_digest(&digest)
    }

    #[test]
    fn layered_root_keeps_the_curated_decks_and_exposes_a_real_decision() {
        let mut env = Env::new(244, true, false, false);
        env.reset(curated::players()).unwrap();
        curated::add_layer_board(&mut env);
        let observation = env.observation_for_player(1).unwrap();
        assert_eq!(observation.agent_permanents.len(), 5);
        assert_eq!(observation.opponent_permanents.len(), 3);
        assert!(env.action_count().unwrap() > 1);
    }

    #[test]
    fn identical_search_results_repeat() {
        let results = [result(&[0.75, 0.25]), result(&[1.0, 0.125, 0.5])];
        assert_eq!(digest(&results), digest(&results.clone()));
    }

    #[test]
    fn changed_losing_score_changes_digest() {
        let before = result(&[0.75, 0.25]);
        let after = result(&[0.75, 0.125]);
        assert_eq!(before.simulations, after.simulations);
        assert_ne!(digest(&[before]), digest(&[after]));
    }

    #[test]
    fn action_order_and_decision_boundaries_change_digest() {
        assert_ne!(
            digest(&[result(&[0.75, 0.25, 0.125])]),
            digest(&[result(&[0.75, 0.125, 0.25])]),
        );
        assert_ne!(
            digest(&[result(&[0.75, 0.25]), result(&[0.5, 0.125])]),
            digest(&[result(&[0.75, 0.25, 0.5, 0.125])]),
        );
    }

    #[test]
    fn simulation_count_and_chosen_action_remain_in_digest() {
        let mut result = result(&[0.75, 0.25]);
        let original = digest(std::slice::from_ref(&result));
        result.simulations += 1;
        assert_ne!(original, digest(std::slice::from_ref(&result)));
        result.simulations -= 1;
        let mut changed_choice = blake3::Hasher::new();
        digest_decision(&mut changed_choice, &result, 1);
        assert_ne!(original, scores_digest(&changed_choice));
    }
}
