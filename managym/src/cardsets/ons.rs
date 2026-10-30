//! Onslaught.
use crate::state::{
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![glory_seeker()]
}

fn glory_seeker() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "ons.glory_seeker",
        scryfall_id: "9047075e-9fca-484d-bb79-32c0d6821281",
        name: "Glory Seeker".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(2),
        toughness: Some(2),
        text_box: text_box.into(),
        ..Default::default()
    }
}
