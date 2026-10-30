//! Ice Age.
use crate::state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![pyroclasm()]
}

fn pyroclasm() -> CardDefinition {
    let text_box = "Pyroclasm deals 2 damage to each creature.";

    CardDefinition {
        key: "ice.pyroclasm",
        scryfall_id: "88040748-ad76-4b9a-bd4e-87e5980e9816",
        name: "Pyroclasm".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::MassDamage { amount: 2 }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
