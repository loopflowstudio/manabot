//! Innistrad.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![typhoid_rats()]
}

fn typhoid_rats() -> CardDefinition {
    let text_box =
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)";

    CardDefinition {
        key: "isd.typhoid_rats",
        scryfall_id: "4490ce65-c73a-4809-abd1-ccc3175bd2a4",
        name: "Typhoid Rats".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Rat".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords::of(&[Keyword::Deathtouch]),
        text_box: text_box.into(),
        ..Default::default()
    }
}
