//! Shared action selection and score identity for bounded search benchmarks.

use managym::agent::env::FlatMcResult;

/// Select the first maximum, preserving legal-action order for ties.
pub fn best_action(result: &FlatMcResult) -> usize {
    result
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
        .0
}

/// Frame each decision with its action count, simulation count and chosen index,
/// then hash every score in legal-action order. Fixed-width little-endian words
/// and IEEE-754 bits avoid platform hashing and decimal rounding. Search-64's
/// scores are exact multiples of 1/128; no floating-point tolerance is needed.
pub fn digest_decision(digest: &mut blake3::Hasher, result: &FlatMcResult, best: usize) {
    digest.update(&(result.scores.len() as u64).to_le_bytes());
    digest.update(&result.simulations.to_le_bytes());
    digest.update(&(best as u64).to_le_bytes());
    for score in &result.scores {
        digest.update(&score.to_bits().to_le_bytes());
    }
}
