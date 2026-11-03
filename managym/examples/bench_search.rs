//! Seeded Search-64 reply timing on the Academy curated matchup (UR Lessons
//! versus GW Allies main decks). Both seats play Academy's default flat search
//! (16 worlds x 4 rollouts, 2,000-step playouts) for a fixed number of
//! decisions per game; each line reports one game's worst, median and total
//! reply time and a digest of the search results, and the last line sums the
//! games. CI compares the output with `managym/perf/baseline.json`
//! (`scripts/engine_budget.py`). Run in release:
//! `cargo run --release --example bench_search --manifest-path managym/Cargo.toml`.
mod curated;

use managym::agent::env::{Env, FlatMcResult};
use std::time::Instant;

const WORLDS: usize = 16;
const ROLLOUTS: usize = 4;
const MAX_STEPS: usize = 2000;
const DECISIONS: usize = 40;
const SEEDS: [u64; 3] = [1, 2, 3];

/// Frame each decision with its action count, simulation count and chosen index,
/// then hash every score in legal-action order. Fixed-width little-endian words
/// and IEEE-754 bits avoid platform hashing and decimal rounding. Search-64's
/// scores are exact multiples of 1/128; no floating-point tolerance is needed.
fn digest_decision(digest: &mut blake3::Hasher, result: &FlatMcResult, best: usize) {
    digest.update(&(result.scores.len() as u64).to_le_bytes());
    digest.update(&result.simulations.to_le_bytes());
    digest.update(&(best as u64).to_le_bytes());
    for score in &result.scores {
        digest.update(&score.to_bits().to_le_bytes());
    }
}

fn scores_digest(digest: &blake3::Hasher) -> u64 {
    // Keep the budget's integer wire format, using the first 64 BLAKE3 bits.
    u64::from_le_bytes(digest.finalize().as_bytes()[..8].try_into().unwrap())
}

fn main() {
    let mut worst_overall: f64 = 0.0;
    let mut total_overall: f64 = 0.0;
    for seed in SEEDS {
        let mut env = Env::new(seed, true, false, false);
        env.reset(curated::players()).unwrap();
        let mut replies = Vec::new();
        let mut digest = blake3::Hasher::new();
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
            digest_decision(&mut digest, &result, best);
            let (_, _, terminated, truncated, _) = env.step(best as i64).unwrap();
            if terminated || truncated {
                break;
            }
        }
        let total: f64 = replies.iter().sum();
        replies.sort_by(f64::total_cmp);
        let worst = *replies.last().unwrap();
        worst_overall = worst_overall.max(worst);
        total_overall += total;
        let scores_digest = scores_digest(&digest);
        println!(
            "{{\"seed\":{seed},\"decisions\":{},\"worst_s\":{worst:.3},\"median_s\":{:.3},\"total_s\":{total:.3},\"digest\":{scores_digest}}}",
            replies.len(),
            replies[replies.len() / 2],
        );
    }
    println!("{{\"worst_overall_s\":{worst_overall:.3},\"total_overall_s\":{total_overall:.3}}}");
}

#[cfg(test)]
mod tests {
    use super::*;

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
