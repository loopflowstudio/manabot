//! Innistrad: Midnight Hunt.
use managym_state::{
    ability::{Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![consider()]
}

fn consider() -> CardDefinition {
    let text_box = "Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)\nDraw a card.";
    CardDefinition {
        key: "mid.consider",
        scryfall_id: "a211d505-4d40-4914-a9da-220770d6ddbc",
        name: "Consider".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::Surveil { count: 1 },
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
