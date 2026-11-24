//! Journey into Nyx.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition, TurnStep},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};

use super::ths::temple;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        temple_of_epiphany(),
        temple_of_malady(),
        dictate_of_kruphix(),
    ]
}

fn temple_of_epiphany() -> CardDefinition {
    temple(
        "jou.temple_of_epiphany",
        "b882c6bf-b795-49fe-8242-a928aadb6f13",
        "Temple of Epiphany",
        ["U", "R"],
    )
}

fn temple_of_malady() -> CardDefinition {
    temple(
        "jou.temple_of_malady",
        "f30220f1-1992-4b5c-9e13-1762fb673155",
        "Temple of Malady",
        ["B", "G"],
    )
}

fn dictate_of_kruphix() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nAt the beginning of each player's draw step, that player draws an additional card.";
    CardDefinition {
        key: "jou.dictate_of_kruphix",
        scryfall_id: "e8e7916c-f39a-48a0-a47d-7e83ebf028fa",
        printed: PrintedCharacteristics {
            name: "Dictate of Kruphix".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Draw, Actor::AnyPlayer),
                intervening_if: None,
                effects: vec![Effect::ForPlayer {
                    player: PlayerSelector::TriggeringPlayer,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
