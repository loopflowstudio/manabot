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

mod aer;
mod afr;
mod blb;
mod bng;
mod bro;
mod c19;
mod cmr;
mod csp;
mod dft;
mod dom;
mod dtk;
mod eld;
mod emn;
mod fdn;
mod fin;
mod frf;
mod fut;
mod gpt;
mod grn;
mod hob;
mod ice;
mod iko;
mod inv;
mod isd;
mod khm;
mod kld;
mod ktk;
mod lea;
mod m12;
mod m15;
mod m20;
mod mbs;
mod mh1;
mod mh2;
mod mid;
mod mkm;
mod ody;
mod ons;
mod ori;
mod pls;
mod por;
mod psal;
mod ptk;
mod rav;
mod roe;
mod rtr;
mod shm;
mod soi;
mod sth;
mod stx;
mod ths;
mod tla;
mod tle;
mod tokens;
mod ulg;
mod vis;
mod war;
mod wth;
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

const SETS: [fn() -> Vec<CardDefinition>; 59] = [
    aer::cards,
    afr::cards,
    blb::cards,
    bng::cards,
    bro::cards,
    c19::cards,
    cmr::cards,
    csp::cards,
    dft::cards,
    dom::cards,
    dtk::cards,
    eld::cards,
    emn::cards,
    fdn::cards,
    fin::cards,
    frf::cards,
    fut::cards,
    gpt::cards,
    grn::cards,
    hob::cards,
    ice::cards,
    iko::cards,
    inv::cards,
    isd::cards,
    khm::cards,
    kld::cards,
    ktk::cards,
    lea::cards,
    m12::cards,
    m15::cards,
    m20::cards,
    mbs::cards,
    mh1::cards,
    mh2::cards,
    mid::cards,
    mkm::cards,
    ody::cards,
    ons::cards,
    ori::cards,
    pls::cards,
    por::cards,
    psal::cards,
    ptk::cards,
    rav::cards,
    roe::cards,
    rtr::cards,
    shm::cards,
    soi::cards,
    sth::cards,
    stx::cards,
    ths::cards,
    tla::cards,
    tle::cards,
    tokens::cards,
    ulg::cards,
    vis::cards,
    war::cards,
    wth::cards,
    zen::cards,
];
