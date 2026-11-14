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

/// MTG-347's smart source choice (`flow/mana/tap_choice.rs`), process-wide:
/// how many payments ended at each stage, what they read and the wall time
/// of each phase. Timing is only taken under this feature.
#[derive(Clone, Default, serde::Serialize)]
pub struct SmartTapMetrics {
    pub payments: u64,
    /// Ended after pruning: saved obligations.
    pub tail: u64,
    /// Ended after pruning: nothing tapped, or no source to tap instead.
    pub no_replacement: u64,
    /// Read the follow-ups and found none.
    pub no_followups: u64,
    /// The kind-preferred payment, or the pruned witness when none is less
    /// reluctant, keeps every follow-up any payment could.
    pub witness_at_bound: u64,
    /// Ran the class search; `exhausted` of them spent its whole budget.
    pub class_search: u64,
    pub exhausted: u64,
    /// Read the source kinds and found a payment less reluctant than the
    /// pruned witness.
    pub kind_preferred: u64,
    pub changed: u64,
    pub followup_reads: u64,
    pub followups: u64,
    pub checks: u64,
    pub searches: u64,
    pub prune_ns: u64,
    pub kinds_ns: u64,
    pub followups_ns: u64,
    pub score_ns: u64,
    pub search_ns: u64,
    /// The whole switch-on block of `produce_mana`, the phases included.
    pub total_ns: u64,
}

static SMART: std::sync::Mutex<Option<SmartTapMetrics>> = std::sync::Mutex::new(None);

pub fn reset_smart_tap() {
    *SMART.lock().unwrap() = None;
}

pub fn smart_tap() -> SmartTapMetrics {
    SMART.lock().unwrap().clone().unwrap_or_default()
}

pub(crate) fn record_smart_tap(update: impl FnOnce(&mut SmartTapMetrics)) {
    update(SMART.lock().unwrap().get_or_insert_with(Default::default));
}

/// Wall time since the previous lap, added to one phase.
pub(crate) struct SmartTapClock(std::time::Instant);

impl SmartTapClock {
    pub(crate) fn start() -> Self {
        Self(std::time::Instant::now())
    }

    pub(crate) fn lap(&mut self, phase: impl FnOnce(&mut SmartTapMetrics) -> &mut u64) {
        let now = std::time::Instant::now();
        let spent = now.duration_since(self.0).as_nanos() as u64;
        self.0 = now;
        record_smart_tap(|metrics| *phase(metrics) += spent);
    }
}
