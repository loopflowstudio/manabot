//! Ice Age.
use crate::state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![pyroclasm()]
}

fn pyroclasm() -> CardDefinition {
    CardDefinition {
        name: "Pyroclasm".to_string(),
        mana_cost: Some(ManaCost::parse("1R")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::MassDamage { amount: 2 }],
        text_box: "Pyroclasm deals 2 damage to each creature.".to_string(),
        ..Default::default()
    }
}
