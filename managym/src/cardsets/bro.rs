//! The Brothers' War.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![giant_cindermaw()]
}

fn giant_cindermaw() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.";

    CardDefinition {
        key: "bro.giant_cindermaw",
        scryfall_id: "1349465f-d29f-4d4b-a653-f4388574c336",
        name: "Giant Cindermaw".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dinosaur".into(), "Beast".into()],
        power: Some(4),
        toughness: Some(3),
        keywords: Keywords {
            trample: true,
            ..Default::default()
        },
        statics: vec![StaticAbility::PlayersCantGainLife],
        text_box: text_box.into(),
        ..Default::default()
    }
}
