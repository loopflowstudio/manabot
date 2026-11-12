//! Magic 2012.
use managym_state::{
    ability::{Condition, Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![visions_of_beyond()]
}

fn visions_of_beyond() -> CardDefinition {
    let text_box =
        "Draw a card. If a graveyard has twenty or more cards in it, draw three cards instead.";
    let draw = |count| Effect::DrawCards {
        count: EffectValue::Fixed(count),
    };
    CardDefinition {
        key: "m12.visions_of_beyond",
        scryfall_id: "75657d26-b0f8-4892-8684-533c103c921d",
        name: "Visions of Beyond".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Instant]),
        // The condition is read once, before any card is drawn.
        spell_effects: vec![Effect::If {
            condition: Condition::AnyGraveyardAtLeast { count: 20 },
            then: vec![draw(3)],
            otherwise: vec![draw(1)],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
