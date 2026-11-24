//! Shards of Alara.
use managym_state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dragon_fodder()]
}

fn dragon_fodder() -> CardDefinition {
    CardDefinition {
        key: "ala.dragon_fodder",
        scryfall_id: "9eab4120-e7d8-4132-a304-30b88e3175e2",
        printed: PrintedCharacteristics {
            name: "Dragon Fodder".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens("token.goblin_1_1_r", 2)],
            text_box: "Create two 1/1 red Goblin creature tokens.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
