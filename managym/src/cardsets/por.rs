//! Portal.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![raging_goblin(), wind_drake()]
}

fn raging_goblin() -> CardDefinition {
    let text_box =
        "Haste (This creature can attack and {T} as soon as it comes under your control.)";

    CardDefinition {
        key: "por.raging_goblin",
        scryfall_id: "fed57a17-7847-4e60-bc40-4452880f12a3",
        name: "Raging Goblin".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Goblin".into(), "Berserker".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            haste: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wind_drake() -> CardDefinition {
    let text_box = "Flying";

    CardDefinition {
        key: "por.wind_drake",
        scryfall_id: "5486d2dc-9a5d-4f58-a5ec-d94de54b852f",
        name: "Wind Drake".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Drake".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
