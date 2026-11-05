//! Opt-in layer attribution for bounded fixtures, never enabled in wheels.
//! Timers are inclusive and overlap (board includes dependency and clone time).
//! Allocation traffic is measured separately by `benchmark_allocations`.

use std::{cell::RefCell, time::Instant};

#[derive(Clone, Default, serde::Serialize)]
pub struct Metrics {
    pub boards: u64,
    pub source_plans: u64,
    pub board_ns: u64,
    pub contexts_cloned: u64,
    pub clone_ns: u64,
    pub dependency_passes: u64,
    pub dependency_ns: u64,
    pub comparisons: u64,
    pub edges: u64,
    pub mana_invalidations: u64,
}

thread_local! {
    static METRICS: RefCell<Metrics> = RefCell::new(Metrics::default());
}

pub fn reset() {
    METRICS.with(|m| *m.borrow_mut() = Metrics::default());
}

pub fn snapshot() -> Metrics {
    METRICS.with(|m| m.borrow().clone())
}

/// Eight isolated pool writes and taps on cloned, warmed fixture boards.
/// This attributes cache work; whole-action costs belong to the step trace.
pub fn probe(game: &mut crate::Game, id: crate::state::game_object::PermanentId) {
    use crate::state::{game_object::PlayerId, mana::Mana};
    for operation in ["mana", "tap"] {
        for sample in 0..8 {
            let mut branch = game.clone();
            branch.characteristics(id);
            reset();
            crate::benchmark_allocations::start();
            let start = Instant::now();
            if operation == "mana" {
                *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("RR");
            } else {
                branch.state.tap(id);
            }
            branch.characteristics(id);
            let elapsed = start.elapsed().as_nanos();
            let traffic = crate::benchmark_allocations::stop();
            println!(
                "{}",
                serde_json::json!({"probe":operation, "sample":sample,
                "elapsed_ns":elapsed, "metrics":snapshot(),
                "allocation_count":traffic.0, "allocated_bytes":traffic.1})
            );
        }
    }
}

pub(crate) fn record(f: impl FnOnce(&mut Metrics)) {
    METRICS.with(|m| f(&mut m.borrow_mut()));
}

pub(crate) struct Timer(Instant, fn(&mut Metrics, u64));

impl Timer {
    pub(crate) fn new(record: fn(&mut Metrics, u64)) -> Self {
        Self(Instant::now(), record)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        record(|m| self.1(m, self.0.elapsed().as_nanos() as u64));
    }
}
