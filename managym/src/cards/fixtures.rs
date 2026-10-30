//! Rules fixtures that are not printed cards.
//!
//! Integration tests and Python verification build games from the default
//! pack, so fixtures register there. They are never in a set file and the
//! world does not support them for deck entry.

use crate::state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

/// A one-mana modal sorcery that exercises mode choice.
pub(super) fn crossroads_of_destiny() -> CardDefinition {
    CardDefinition {
        name: "Crossroads of Destiny".to_string(),
        mana_cost: Some(ManaCost::parse("1")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::GainLife { amount: 3 }],
                vec![Effect::DrawCards { count: 1 }],
            ],
        }],
        text_box: "Choose one —\n• You gain 3 life.\n• Draw a card.".to_string(),
        ..Default::default()
    }
}
