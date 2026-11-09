//! Opt-in attribution for bounded search fixtures. No timers in ordinary wheels.
//! Categories are inclusive; witness subcategories overlap witness construction.

use std::{cell::RefCell, collections::BTreeMap, time::Instant};

#[derive(Clone, Default, serde::Serialize)]
pub struct Cost {
    calls: u64,
    seconds: f64,
}

thread_local! {
    static COSTS: RefCell<Option<BTreeMap<&'static str, Cost>>> = const { RefCell::new(None) };
}

pub fn start() {
    COSTS.with(|costs| *costs.borrow_mut() = Some(BTreeMap::new()));
}

pub fn finish() -> BTreeMap<&'static str, Cost> {
    COSTS.with(|costs| costs.borrow_mut().take().unwrap_or_default())
}

pub(crate) struct Timer(Option<(Instant, &'static str)>);

impl Timer {
    pub(crate) fn new(category: &'static str) -> Self {
        Self(COSTS.with(|costs| costs.borrow().is_some().then(|| (Instant::now(), category))))
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((start, category)) = self.0 {
            COSTS.with(|costs| {
                if let Some(costs) = costs.borrow_mut().as_mut() {
                    let cost = costs.entry(category).or_default();
                    cost.calls += 1;
                    cost.seconds += start.elapsed().as_secs_f64();
                }
            });
        }
    }
}
