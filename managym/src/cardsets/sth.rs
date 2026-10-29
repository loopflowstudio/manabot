//! Stronghold.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![youthful_knight()]
}

fn youthful_knight() -> CardDefinition {
    let text_box = "First strike";

    CardDefinition {
        key: "sth.youthful_knight",
        scryfall_id: "fcd3b7a7-c5e8-4179-ade3-31625620f3a9",
        name: "Youthful Knight".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Knight".into()],
        power: Some(2),
        toughness: Some(1),
        keywords: Keywords {
            first_strike: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
