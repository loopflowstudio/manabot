//! Card set files: one file per printed set, each card defined once.
//!
//! Every card is a typed Rust function in its set file, listed in that file's
//! `cards()`. [`catalog`] gathers them for `semantic::compile`, which lowers
//! them in process; there is no other card registration. Tokens made by TLA
//! cards live in `tla.rs`; other tokens live in `tokens.rs`.

mod afr;
mod arb;
mod blb;
mod bro;
mod eld;
mod fin;
mod iko;
mod ktk;
mod lci;
mod lea;
mod m20;
mod mh1;
mod ons;
mod soi;
mod stx;
mod tla;
mod tokens;
mod war;
mod zen;

use crate::cards::authoring::Catalog;

pub fn catalog() -> Catalog {
    let mut definitions = tla::cards();
    let token_templates = definitions
        .iter()
        .filter(|card| card.characteristics.token == Some(true))
        .map(|card| card.key)
        .collect();
    definitions.extend(afr::cards());
    definitions.extend(arb::cards());
    definitions.extend(blb::cards());
    definitions.extend(bro::cards());
    definitions.extend(eld::cards());
    definitions.extend(fin::cards());
    definitions.extend(iko::cards());
    definitions.extend(ktk::cards());
    definitions.extend(lci::cards());
    definitions.extend(lea::cards());
    definitions.extend(m20::cards());
    definitions.extend(mh1::cards());
    definitions.extend(ons::cards());
    definitions.extend(soi::cards());
    definitions.extend(stx::cards());
    definitions.extend(tokens::cards());
    definitions.extend(war::cards());
    definitions.extend(zen::cards());
    definitions.sort_by_key(|card| card.key);
    Catalog {
        definitions,
        token_templates,
    }
}
