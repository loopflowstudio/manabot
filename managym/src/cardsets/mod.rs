//! Card set files: one file per printed set, each card filed once under the
//! set of its first printing.
//!
//! A set file lists typed Rust definitions in `cards()`; [`catalog`] gathers
//! them for `semantic::compile`, which lowers them in process. A few cards the
//! typed language cannot yet express are native [`CardDefinition`]s in their
//! set file's `native_cards()`; they run in the default pack but have no
//! semantic program, so the world does not support them for deck entry. Tokens
//! made by TLA cards live in `tla.rs`; other tokens live in `tokens.rs`.

mod afr;
mod arb;
mod blb;
mod bro;
mod eld;
mod fin;
mod grn;
mod ice;
mod iko;
mod isd;
mod ktk;
mod lci;
mod lea;
mod m20;
mod mh1;
mod mkm;
mod ons;
mod ori;
mod por;
mod rtr;
mod soi;
mod sth;
mod stx;
mod tla;
mod tokens;
mod vis;
mod war;
mod zen;

use crate::{
    cards::authoring::Catalog,
    state::{
        card::{CardDefinition, CardType, CardTypes, Keywords},
        mana::ManaCost,
    },
};

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
        incomplete: tla::INCOMPLETE.to_vec(),
    }
}

/// Every native definition, in no particular order.
pub fn native_cards() -> Vec<CardDefinition> {
    let mut definitions = grn::native_cards();
    definitions.extend(ice::native_cards());
    definitions.extend(isd::native_cards());
    definitions.extend(lea::native_cards());
    definitions.extend(mkm::native_cards());
    definitions.extend(ori::native_cards());
    definitions.extend(por::native_cards());
    definitions.extend(rtr::native_cards());
    definitions.extend(sth::native_cards());
    definitions.extend(stx::native_cards());
    definitions.extend(vis::native_cards());
    definitions
}

/// A native creature with only keyword abilities.
fn native_creature(
    name: &str,
    mana_cost: &str,
    subtypes: &[&str],
    power: i32,
    toughness: i32,
    keywords: Keywords,
    text_box: &str,
) -> CardDefinition {
    CardDefinition {
        name: name.to_string(),
        mana_cost: Some(ManaCost::parse(mana_cost)),
        types: CardTypes::new([CardType::Creature]),
        subtypes: subtypes
            .iter()
            .map(|subtype| (*subtype).to_string())
            .collect(),
        keywords,
        text_box: text_box.to_string(),
        power: Some(power),
        toughness: Some(toughness),
        ..Default::default()
    }
}
