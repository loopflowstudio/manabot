//! Zendikar.
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![lotus_cobra()]
}

fn lotus_cobra() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, add one mana of any color.";

    CardDefinition {
        key: "zen.lotus_cobra",
        scryfall_id: "19adde22-e5eb-4815-beb6-c520b3274cc9",
        name: "Lotus Cobra".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Snake".into()],
        power: Some(2),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::AddMana {
                        mana: Mana::parse("W"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("U"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("B"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("R"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("G"),
                        until_end_of_combat: false,
                    }],
                ],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
