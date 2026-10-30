//! Guilds of Ravnica.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![healer_s_hawk()]
}

fn healer_s_hawk() -> CardDefinition {
    let text_box =
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)";

    CardDefinition {
        key: "grn.healer_s_hawk",
        scryfall_id: "3313bd5c-b657-47a3-822a-dd0d9165492a",
        name: "Healer's Hawk".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bird".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            lifelink: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
