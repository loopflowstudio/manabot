//! Card set files: one module per printed set, each card filed once under
//! the set of its first printing.
//!
//! Each card is one function returning the engine's own [`CardDefinition`]
//! (a key, a Scryfall id and its `PrintedCharacteristics`, written in the
//! `managym-state` rules types), listed in its set module's
//! `cards()`; [`catalog`] gathers every set and is the only way the engine
//! reaches them. New behavior is a new `Effect` (or other rules-type) variant,
//! never card-specific logic. Tokens are not cards: each is a
//! [`TokenDefinition`] in `tokens.rs`, reached through [`token_catalog`] by
//! its `token.*` key. Defined cards are supported; known rules bugs are
//! tracked as Tasks, not deck-entry exclusions. Card behavior tests live in
//! `tests/cards/<set>.rs` beside the sets.

mod aer;
mod afr;
mod akh;
mod ala;
mod atq;
mod avr;
mod bfz;
mod blb;
mod bng;
mod bro;
mod c17;
mod c19;
mod chk;
mod clb;
mod cmr;
mod csp;
mod dft;
mod dka;
mod dmu;
mod dom;
mod dtk;
mod ecl;
mod eld;
mod emn;
mod eoe;
mod fdn;
mod fin;
mod frf;
mod fut;
mod gpt;
mod grn;
mod gtc;
mod hob;
mod hou;
mod ice;
mod iko;
mod inv;
mod isd;
mod j22;
mod jmp;
mod jou;
mod khm;
mod kld;
mod ktk;
mod lea;
mod lrw;
mod m10;
mod m11;
mod m12;
mod m14;
mod m15;
mod m19;
mod m20;
mod mbs;
mod mh1;
mod mh2;
mod mid;
mod mkm;
mod mmq;
mod mor;
mod mrd;
mod msh;
mod nec;
mod ody;
mod ogw;
mod ons;
mod ori;
mod p02;
mod pls;
mod por;
mod psal;
mod ptk;
mod rav;
mod rix;
mod roe;
mod rtr;
mod shm;
mod soi;
mod spm;
mod sth;
mod stx;
mod tdm;
mod ths;
mod tla;
mod tle;
mod tmp;
mod tokens;
mod tsp;
mod ulg;
mod usg;
mod vis;
mod war;
mod woe;
mod wth;
mod xln;
mod zen;
mod znr;

use std::sync::OnceLock;

use managym_state::card::{CardDefinition, TokenDefinition};

/// Every card definition from every set, sorted by key. Tokens are in
/// [`token_catalog`].
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

/// Every token definition, sorted by key.
pub fn token_catalog() -> &'static [TokenDefinition] {
    static TOKENS: OnceLock<Vec<TokenDefinition>> = OnceLock::new();
    TOKENS.get_or_init(|| {
        let mut definitions = tokens::tokens();
        definitions.sort_by_key(|definition| definition.key);
        definitions
    })
}

const SETS: &[fn() -> Vec<CardDefinition>] = &[
    aer::cards,
    afr::cards,
    akh::cards,
    ala::cards,
    atq::cards,
    avr::cards,
    bfz::cards,
    blb::cards,
    bng::cards,
    bro::cards,
    c17::cards,
    c19::cards,
    chk::cards,
    clb::cards,
    cmr::cards,
    csp::cards,
    dft::cards,
    dka::cards,
    dmu::cards,
    dom::cards,
    dtk::cards,
    ecl::cards,
    eld::cards,
    emn::cards,
    eoe::cards,
    fdn::cards,
    fin::cards,
    frf::cards,
    fut::cards,
    gpt::cards,
    grn::cards,
    gtc::cards,
    hob::cards,
    hou::cards,
    ice::cards,
    iko::cards,
    inv::cards,
    isd::cards,
    j22::cards,
    jmp::cards,
    jou::cards,
    khm::cards,
    kld::cards,
    ktk::cards,
    lea::cards,
    lrw::cards,
    m10::cards,
    m11::cards,
    m12::cards,
    m14::cards,
    m15::cards,
    m19::cards,
    m20::cards,
    mbs::cards,
    mh1::cards,
    mh2::cards,
    mid::cards,
    mkm::cards,
    mmq::cards,
    mor::cards,
    mrd::cards,
    msh::cards,
    nec::cards,
    ody::cards,
    ogw::cards,
    ons::cards,
    ori::cards,
    p02::cards,
    pls::cards,
    por::cards,
    psal::cards,
    ptk::cards,
    rav::cards,
    rix::cards,
    roe::cards,
    rtr::cards,
    shm::cards,
    soi::cards,
    spm::cards,
    sth::cards,
    stx::cards,
    tdm::cards,
    ths::cards,
    tla::cards,
    tle::cards,
    tmp::cards,
    tsp::cards,
    ulg::cards,
    usg::cards,
    vis::cards,
    war::cards,
    woe::cards,
    wth::cards,
    xln::cards,
    zen::cards,
    znr::cards,
];
