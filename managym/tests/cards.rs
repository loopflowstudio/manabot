//! Per-card behavior tests, one module per printed set, mirroring
//! `src/cardsets/`. Engine-rule tests that are not about one card stay in
//! `tests/rules/`. Tests witness behavior; they never gate card support,
//! which comes only from the set file defining the card.
// Shared with the rules tests; this crate uses only part of it.
#[allow(dead_code)]
#[path = "rules/helpers.rs"]
mod helpers;
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/support.rs"]
mod support;
#[path = "cards/tla.rs"]
mod tla;
