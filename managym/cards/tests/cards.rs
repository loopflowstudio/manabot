//! Card behavior tests, one module per set module in `src/`: a test lives
//! with the set that first printed the card it is about. Engine rules that
//! are not about one card stay in the engine's `rules_tests`. Tests never
//! decide whether a card is supported; see `cards/README.md` for coverage.
//! The scenario helpers are the engine's `tests/support`, mounted by path so
//! each primitive stays defined once.

#[path = "../../tests/support/mod.rs"]
mod support;

#[path = "cards/aer.rs"]
mod aer;
#[path = "cards/afr.rs"]
mod afr;
#[path = "cards/blb.rs"]
mod blb;
#[path = "cards/bng.rs"]
mod bng;
#[path = "cards/bro.rs"]
mod bro;
#[path = "cards/c17.rs"]
mod c17;
#[path = "cards/c19.rs"]
mod c19;
#[path = "cards/chk.rs"]
mod chk;
#[path = "cards/cmr.rs"]
mod cmr;
#[path = "cards/csp.rs"]
mod csp;
#[path = "cards/definitions.rs"]
mod definitions;
#[path = "cards/dft.rs"]
mod dft;
#[path = "cards/dom.rs"]
mod dom;
#[path = "cards/dtk.rs"]
mod dtk;
#[path = "cards/eld.rs"]
mod eld;
#[path = "cards/emn.rs"]
mod emn;
#[path = "cards/fdn.rs"]
mod fdn;
#[path = "cards/frf.rs"]
mod frf;
#[path = "cards/fut.rs"]
mod fut;
#[path = "cards/gpt.rs"]
mod gpt;
#[path = "cards/hob.rs"]
mod hob;
#[path = "cards/ice.rs"]
mod ice;
#[path = "cards/iko.rs"]
mod iko;
#[path = "cards/inv.rs"]
mod inv;
#[path = "cards/isd.rs"]
mod isd;
#[path = "cards/khm.rs"]
mod khm;
#[path = "cards/kld.rs"]
mod kld;
#[path = "cards/ktk.rs"]
mod ktk;
#[path = "cards/lea.rs"]
mod lea;
#[path = "cards/m12.rs"]
mod m12;
#[path = "cards/m15.rs"]
mod m15;
#[path = "cards/m20.rs"]
mod m20;
#[path = "cards/mbs.rs"]
mod mbs;
#[path = "cards/mh1.rs"]
mod mh1;
#[path = "cards/mid.rs"]
mod mid;
#[path = "cards/mkm.rs"]
mod mkm;
#[path = "cards/mmq.rs"]
mod mmq;
#[path = "cards/observability.rs"]
mod observability;
#[path = "cards/ody.rs"]
mod ody;
#[path = "cards/ons.rs"]
mod ons;
#[path = "cards/ori.rs"]
mod ori;
#[path = "cards/pls.rs"]
mod pls;
#[path = "cards/por.rs"]
mod por;
#[path = "cards/psal.rs"]
mod psal;
#[path = "cards/ptk.rs"]
mod ptk;
#[path = "cards/rav.rs"]
mod rav;
#[path = "cards/roe.rs"]
mod roe;
#[path = "cards/scryfall.rs"]
mod scryfall;
#[path = "cards/shm.rs"]
mod shm;
#[path = "cards/soi.rs"]
mod soi;
#[path = "cards/sth.rs"]
mod sth;
#[path = "cards/stx.rs"]
mod stx;
#[path = "cards/ths.rs"]
mod ths;
#[path = "cards/tla.rs"]
mod tla;
#[path = "cards/tle.rs"]
mod tle;
#[path = "cards/ulg.rs"]
mod ulg;
#[path = "cards/vis.rs"]
mod vis;
#[path = "cards/war.rs"]
mod war;
#[path = "cards/wth.rs"]
mod wth;
#[path = "cards/zen.rs"]
mod zen;
