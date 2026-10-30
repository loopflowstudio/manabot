//! Magic Origins.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![boggart_brute()]
}

fn boggart_brute() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)";

    CardDefinition {
        key: "ori.boggart_brute",
        scryfall_id: "9d735ebf-61a4-4507-9399-6d32c8903ded",
        name: "Boggart Brute".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Goblin".into(), "Warrior".into()],
        power: Some(3),
        toughness: Some(2),
        keywords: Keywords {
            menace: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
