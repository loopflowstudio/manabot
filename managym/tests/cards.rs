//! Card behavior tests, one module per set file in `src/cardsets/`: a test
//! lives with the set that first printed the card it is about. Engine rules
//! that are not about one card stay in `rules_tests`. Tests never decide
//! whether a card is supported; see `cards/README.md` for coverage.
//! `observability` checks that every viewer fact reaches the policy tensor.

mod support;

#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/arb.rs"]
mod arb;
#[path = "cards/blb.rs"]
mod blb;
#[path = "cards/eld.rs"]
mod eld;
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
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/soi.rs"]
mod soi;
#[path = "cards/stx.rs"]
mod stx;
#[path = "cards/tla.rs"]
mod tla;
#[path = "cards/war.rs"]
mod war;
#[path = "cards/zen.rs"]
mod zen;
