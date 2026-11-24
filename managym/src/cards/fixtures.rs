//! Rules fixtures that are not printed cards.
//!
//! Integration tests and Python verification build games from the default
//! pack, so fixtures register there. They are never in a set file and the
//! world does not support them for deck entry.

use crate::state::{
    ability::{Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes, ManaAbility, PrintedCharacteristics},
    mana::{Mana, ManaCost},
};

/// A one-mana modal sorcery that exercises mode choice.
pub(super) fn crossroads_of_destiny() -> CardDefinition {
    CardDefinition {
        printed: PrintedCharacteristics {
            name: "Crossroads of Destiny".to_string(),
            mana_cost: Some(ManaCost::parse("1")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::gain_life(3)],
                    vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                ],
            }],
            text_box: "Choose one —\n• You gain 3 life.\n• Draw a card.".to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn island_face(name: &str) -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: name.to_string(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("U"),
        }],
        text_box: "{T}: Add {U}.".to_string(),
        ..Default::default()
    }
}

/// A modal double-faced card with a sorcery front and a land back
/// (CR 712.12): one land play or one cast, never both.
pub(super) fn tidal_errand() -> CardDefinition {
    CardDefinition {
        printed: PrintedCharacteristics {
            name: "Tidal Errand".to_string(),
            modal_back: true,
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
            text_box: "Draw a card.".to_string(),
            ..Default::default()
        },
        back_face: Some(Box::new(island_face("Tidal Haven"))),
        ..Default::default()
    }
}

/// A modal double-faced card with a land front and a creature back, so
/// creature permissions reach a card that is a land in every zone.
pub(super) fn kelp_hollow() -> CardDefinition {
    CardDefinition {
        printed: PrintedCharacteristics {
            modal_back: true,
            ..island_face("Kelp Hollow")
        },
        back_face: Some(Box::new(PrintedCharacteristics {
            name: "Kelp Warden".to_string(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            power: Some(2),
            toughness: Some(3),
            text_box: String::new(),
            ..Default::default()
        })),
        ..Default::default()
    }
}
