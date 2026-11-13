//! Counts of mana payment search work in debug and opt-in benchmark builds.
//! Release wheels contain neither counters nor the scoped test work limit.
//! A plan is one `PaymentModel::plan`/`plan_after_tap` request; a call is one
//! `PaymentModel::witness` entry; a state is a call that passed the bounds and
//! was new to its plan's memo, so it expanded successors. Counts are
//! per-thread and deterministic for a fixed workload.

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
    static COUNTERS: RefCell<Counters> = RefCell::new(Counters::default());
}

#[derive(Default)]
struct Counters {
    metrics: Metrics,
    plan_start: u64,
    plan_sources: usize,
    limit: Option<CallLimit>,
}

struct CallLimit {
    calls: u64,
    context: String,
}

/// Stop a regressed search at its first over-budget call. The limit applies
/// to each plan on this thread and is restored even when gameplay unwinds.
/// A vector isolated step contains panics, so callers must also check its
/// error receipts rather than treating a returned batch as success.
pub fn with_call_limit<T>(calls: u64, context: &str, work: impl FnOnce() -> T) -> T {
    struct Restore(Option<CallLimit>);
    impl Drop for Restore {
        fn drop(&mut self) {
            COUNTERS.with(|counters| counters.borrow_mut().limit = self.0.take());
        }
    }
    let _restore = Restore(COUNTERS.with(|counters| {
        counters.borrow_mut().limit.replace(CallLimit {
            calls,
            context: context.into(),
        })
    }));
    work()
}

/// Reset counts without removing a scoped call limit.
pub fn reset() {
    COUNTERS.with(|counters| {
        let mut counters = counters.borrow_mut();
        counters.metrics = Metrics::default();
        counters.plan_start = 0;
    });
}

pub fn snapshot() -> Metrics {
    COUNTERS.with(|counters| counters.borrow().metrics.clone())
}

pub(crate) fn begin_plan(sources: usize) {
    COUNTERS.with(|counters| {
        let mut counters = counters.borrow_mut();
        counters.metrics.plans += 1;
        counters.metrics.max_sources = counters.metrics.max_sources.max(sources as u64);
        counters.plan_start = counters.metrics.calls;
        counters.plan_sources = sources;
    });
}

pub(crate) fn end_plan() {
    COUNTERS.with(|counters| {
        let mut counters = counters.borrow_mut();
        let spent = counters.metrics.calls - counters.plan_start;
        counters.metrics.max_plan_calls = counters.metrics.max_plan_calls.max(spent);
        counters.metrics.plans_over_1k += u64::from(spent >= 1000);
    });
}

pub(crate) fn call() {
    COUNTERS.with(|counters| {
        let mut counters = counters.borrow_mut();
        counters.metrics.calls += 1;
        if let Some(limit) = &counters.limit {
            assert!(
                counters.metrics.calls - counters.plan_start <= limit.calls,
                "payment search passed {} calls in one plan ({} sources)\n  {}",
                limit.calls,
                counters.plan_sources,
                limit.context
            );
        }
    });
}

pub(crate) fn state() {
    COUNTERS.with(|counters| counters.borrow_mut().metrics.states += 1);
}
