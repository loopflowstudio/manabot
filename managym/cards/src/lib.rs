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
mod apc;
mod arb;
mod atq;
mod avr;
mod bfz;
mod blb;
mod bng;
mod bro;
mod c13;
mod c14;
mod c15;
mod c17;
mod c19;
mod chk;
mod clb;
mod cmd;
mod cmr;
mod csp;
mod dft;
mod dgm;
mod dis;
mod dka;
mod dmu;
mod dom;
mod drk;
mod dtk;
mod ecl;
mod eld;
mod emn;
mod eoe;
mod fdn;
mod fin;
pub use fin::ragnarok;
mod dsk;
mod fra;
mod frf;
mod fut;
mod gpt;
mod grn;
mod gs1;
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
mod khc;
mod khm;
mod kld;
mod ktk;
mod lci;
mod lea;
mod lrw;
mod m10;
mod m11;
mod m12;
mod m13;
mod m14;
mod m15;
mod m19;
mod m20;
mod m21;
mod mbs;
mod mh1;
mod mh2;
mod mid;
mod mir;
mod mkm;
mod mmq;
mod mom;
mod mor;
mod mrd;
mod msh;
mod ncc;
mod nec;
mod ody;
mod ogw;
mod one;
mod ons;
mod ori;
mod p02;
mod pls;
mod pm15;
mod por;
mod psal;
mod ptk;
mod rav;
mod rix;
mod rna;
mod roe;
mod rtr;
mod scg;
mod shm;
mod snc;
mod soi;
mod sok;
mod som;
mod sos;
mod spm;
mod sth;
mod stx;
mod tdm;
mod thb;
mod ths;
mod tla;
mod tle;
mod tmp;
mod tmt;
mod tokens;
mod emblems;
pub use emblems::emblem_catalog;
mod tsp;
mod ulg;
mod usg;
mod vis;
mod vow;
mod war;
mod woe;
mod wth;
mod wwk;
mod xln;
mod zen;
mod znr;

use std::sync::OnceLock;

use managym_state::card::{
    CardDefinition, CardType, CardTypes, InsetSpell, PrintedCharacteristics, ReplacementEffect,
    TokenDefinition,
};

fn enters_prepared(spell: PrintedCharacteristics) -> PrintedCharacteristics {
    PrintedCharacteristics {
        types: CardTypes::new([CardType::Creature]),
        replacement_effects: vec![ReplacementEffect::EntersPrepared],
        inset: Some(InsetSpell::Prepared(spell.into())),
        ..Default::default()
    }
}

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
    arb::cards,
    atq::cards,
    ala::cards,
    apc::cards,
    avr::cards,
    bfz::cards,
    blb::cards,
    bng::cards,
    bro::cards,
    c14::cards,
    c15::cards,
    c13::cards,
    c17::cards,
    c19::cards,
    chk::cards,
    clb::cards,
    cmd::cards,
    cmr::cards,
    csp::cards,
    dft::cards,
    dis::cards,
    dka::cards,
    dgm::cards,
    dmu::cards,
    dom::cards,
    drk::cards,
    dsk::cards,
    dtk::cards,
    ecl::cards,
    eld::cards,
    emn::cards,
    eoe::cards,
    fdn::cards,
    fin::cards,
    fra::cards,
    frf::cards,
    fut::cards,
    gpt::cards,
    grn::cards,
    gs1::cards,
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
    khc::cards,
    khm::cards,
    kld::cards,
    ktk::cards,
    lci::cards,
    lea::cards,
    m10::cards,
    m12::cards,
    m13::cards,
    lrw::cards,
    m11::cards,
    m14::cards,
    m15::cards,
    m19::cards,
    m20::cards,
    m21::cards,
    mbs::cards,
    mh1::cards,
    mh2::cards,
    mid::cards,
    mir::cards,
    mkm::cards,
    mmq::cards,
    mom::cards,
    mor::cards,
    mrd::cards,
    msh::cards,
    ncc::cards,
    nec::cards,
    ody::cards,
    ogw::cards,
    one::cards,
    ons::cards,
    ori::cards,
    p02::cards,
    pls::cards,
    pm15::cards,
    por::cards,
    psal::cards,
    ptk::cards,
    rav::cards,
    rix::cards,
    rna::cards,
    roe::cards,
    rtr::cards,
    scg::cards,
    shm::cards,
    snc::cards,
    soi::cards,
    sok::cards,
    som::cards,
    sos::cards,
    spm::cards,
    sth::cards,
    stx::cards,
    tdm::cards,
    thb::cards,
    ths::cards,
    tla::cards,
    tle::cards,
    tmt::cards,
    tmp::cards,
    tsp::cards,
    ulg::cards,
    usg::cards,
    vis::cards,
    vow::cards,
    war::cards,
    woe::cards,
    wth::cards,
    wwk::cards,
    xln::cards,
    zen::cards,
    znr::cards,
];
