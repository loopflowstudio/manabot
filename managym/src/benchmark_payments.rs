//! Opt-in counts of mana payment search work, never enabled in wheels.
//! A plan is one `PaymentModel::plan`/`plan_after_tap` request, whether the
//! search or the direct answer builds its witness; an affordability question
//! `PaymentModel::payable` answers directly builds no witness and is not
//! counted. A call is one `PaymentModel::witness` entry; a state is a call
//! that passed the bounds and was new to its plan's memo, so it expanded
//! successors. Counts are per-thread and deterministic for a fixed workload.

use std::cell::RefCell;

#[derive(Clone, Default, serde::Serialize)]
pub struct Metrics {
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

pub(crate) fn end_plan() {
    METRICS.with(|m| {
        let (metrics, start) = &mut *m.borrow_mut();
        let spent = metrics.calls - *start;
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
