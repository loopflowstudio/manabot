//! Card set files: one module per printed set, each card filed once under
//! the set of its first printing.
//!
//! Each card is one function returning the engine's own [`CardDefinition`]
//! (written in the `managym-state` rules types), listed in its set module's
//! `cards()`; [`catalog`] gathers every set and is the only way the engine
//! reaches them. New behavior is a new `Effect` (or other rules-type) variant,
//! never card-specific logic. Every token lives in `tokens.rs`. Defined
//! non-token cards are supported; known rules bugs are tracked as Tasks,
//! not deck-entry exclusions. Card behavior
//! tests live in `tests/cards/<set>.rs` beside the sets.

mod afr;
mod blb;
mod bro;
mod eld;
mod fin;
mod grn;
mod ice;
mod iko;
mod isd;
mod ktk;
mod lea;
mod m20;
mod mh1;
mod mh2;
mod mkm;
mod ons;
mod ori;
mod por;
mod ptk;
mod roe;
mod rtr;
mod soi;
mod sth;
mod stx;
mod ths;
mod tla;
mod tokens;
mod vis;
mod war;
mod zen;

use std::sync::OnceLock;

use managym_state::card::CardDefinition;

/// Every card and token definition from every set, sorted by key.
pub fn catalog() -> &'static [CardDefinition] {
    static CATALOG: OnceLock<Vec<CardDefinition>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut definitions = Vec::new();
        for cards in SETS {
            definitions.extend(cards());
        }
        definitions.sort_by_key(|definition| definition.key);
        definitions
    })
}

const SETS: [fn() -> Vec<CardDefinition>; 30] = [
    afr::cards,
    blb::cards,
    bro::cards,
    eld::cards,
    fin::cards,
    grn::cards,
    ice::cards,
    iko::cards,
    isd::cards,
    ktk::cards,
    lea::cards,
    m20::cards,
    mh1::cards,
    mh2::cards,
    mkm::cards,
    ons::cards,
    ori::cards,
    por::cards,
    ptk::cards,
    roe::cards,
    rtr::cards,
    soi::cards,
    sth::cards,
    stx::cards,
    ths::cards,
    tla::cards,
    tokens::cards,
    vis::cards,
    war::cards,
    zen::cards,
];
