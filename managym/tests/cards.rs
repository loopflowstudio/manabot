//! Per-card behavior tests, one module per printed set, mirroring
//! `src/cardsets/`. Engine-rule tests that are not about one card stay in
//! `tests/rules/`. Tests witness behavior; they never gate card support,
//! which comes only from the set file defining the card.
// Shared with the rules tests; this crate uses only part of it.
#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/arb.rs"]
mod arb;
#[path = "cards/blb.rs"]
mod blb;
#[path = "cards/eld.rs"]
mod eld;
#[path = "cards/fin.rs"]
mod fin;
#[allow(dead_code)]
#[path = "rules/helpers.rs"]
mod helpers;
#[path = "cards/ice.rs"]
mod ice;
#[path = "cards/iko.rs"]
mod iko;
#[path = "cards/ktk.rs"]
mod ktk;
#[path = "cards/lea.rs"]
mod lea;
#[path = "cards/m20.rs"]
mod m20;
#[path = "cards/mkm.rs"]
mod mkm;
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/stx.rs"]
mod stx;
#[path = "cards/support.rs"]
mod support;
#[path = "cards/tla.rs"]
mod tla;
#[path = "cards/tokens.rs"]
mod tokens;
#[path = "cards/war.rs"]
mod war;
