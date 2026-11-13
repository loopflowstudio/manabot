//! Opt-in counts of mana payment search work, never enabled in wheels.
//! A plan is one `PaymentModel::plan`/`plan_after_tap` request, whether the
//! search or the direct answer builds its witness; an affordability question
//! `PaymentModel::payable` answers directly builds no witness and is not
//! counted. A call is one `PaymentModel::witness` entry; a state is a call
//! that passed the bounds and was new to its plan's memo, so it expanded
//! successors. Counts are per-thread and deterministic for a fixed workload.

use crate::{
    agent::env::Env,
    state::player::{DecklistDisclosure, PlayerConfig},
};
use std::{cell::RefCell, collections::BTreeMap};

#[derive(Clone, Default, serde::Serialize)]
pub struct Metrics {
    /// Ordered first witnesses, including failed plans (FNV-1a over JSON).
    pub witness_digest: u64,
    pub plans: u64,
    pub calls: u64,
    pub states: u64,
    /// Most witness calls spent by a single plan.
    pub max_plan_calls: u64,
    /// Plans that spent at least 1,000 witness calls.
    pub plans_over_1k: u64,
    pub max_sources: u64,
}

thread_local! {
    static METRICS: RefCell<(Metrics, u64)> = RefCell::new((Metrics::default(), 0));
}

pub fn reset() {
    METRICS.with(|m| *m.borrow_mut() = (Metrics::default(), 0));
}

pub fn snapshot() -> Metrics {
    METRICS.with(|m| m.borrow().0.clone())
}

pub(crate) fn begin_plan(sources: usize) {
    METRICS.with(|m| {
        let (metrics, start) = &mut *m.borrow_mut();
        metrics.plans += 1;
        metrics.max_sources = metrics.max_sources.max(sources as u64);
        *start = metrics.calls;
    });
}

pub(crate) fn end_plan(witness: &impl serde::Serialize) {
    METRICS.with(|m| {
        let (metrics, start) = &mut *m.borrow_mut();
        let spent = metrics.calls - *start;
        for byte in serde_json::to_vec(witness).unwrap() {
            metrics.witness_digest =
                (metrics.witness_digest ^ u64::from(byte)).wrapping_mul(1099511628211);
        }
        metrics.max_plan_calls = metrics.max_plan_calls.max(spent);
        metrics.plans_over_1k += u64::from(spent >= 1000);
    });
}

pub(crate) fn call() {
    METRICS.with(|m| m.borrow_mut().0.calls += 1);
}

pub(crate) fn state() {
    METRICS.with(|m| m.borrow_mut().0.states += 1);
}

fn literal<'a>(source: &'a str, name: &str, close: &str) -> &'a str {
    let (_, value) = source.split_once(&format!("\n{name} = ")).unwrap();
    value[..value.find(close).unwrap() + close.len()].trim()
}

/// The game of a reporter script, reset and unplayed, with its actions: the
/// script's `SEED`, `DECKS` and `ACTIONS` literals, closed decklists and
/// submitted-deck admission, as the reporter's own run had them.
pub fn reporter_game(script: &str) -> (Env, Vec<i64>) {
    let seed: u64 = literal(script, "SEED", "\n").parse().unwrap();
    let decks: Vec<BTreeMap<String, usize>> =
        serde_json::from_str(literal(script, "DECKS", "\n]\n")).unwrap();
    let actions = serde_json::from_str(literal(script, "ACTIONS", "]\n")).unwrap();
    let players = decks
        .into_iter()
        .enumerate()
        .map(|(seat, deck)| {
            let mut player = PlayerConfig::new(format!("seat {seat}"), deck).submit_to_world();
            player.disclosure = DecklistDisclosure::Closed;
            player
        })
        .collect();
    let mut env = Env::new(seed, true, false, false);
    env.reset(players).unwrap();
    (env, actions)
}
