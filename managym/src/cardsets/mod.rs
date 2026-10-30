//! Card set files: one file per printed set, each card filed once under the
//! set of its first printing.
//!
//! Each card is one function returning the engine's own [`CardDefinition`],
//! listed in its set file's `cards()`; [`catalog`] gathers every set. New
//! behavior is a new `Effect` (or other rules-type) variant, never card-specific
//! logic. Every token lives in `tokens.rs`. A set file may mark a defined card
//! incomplete beside its definitions ([`incomplete`]); every other defined card
//! is supported.

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

use crate::state::card::CardDefinition;

/// A defined card whose behavior is known to be incomplete: `(key, missing
/// behavior)`. It still runs in the fixed setups that name it, but the world
/// does not support it for deck entry.
pub type Incomplete = (&'static str, &'static str);

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

/// Defined cards whose behavior is incomplete, with the missing behavior.
pub fn incomplete() -> &'static [Incomplete] {
    tla::INCOMPLETE
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
