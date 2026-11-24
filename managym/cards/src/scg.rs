//! Scourge.

use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fierce_empath()]
}

fn fierce_empath() -> CardDefinition {
    let text_box = "When this creature enters, you may search your library for a creature card with mana value 6 or greater, reveal it, put it into your hand, then shuffle.";
    CardDefinition {
        key: "scg.fierce_empath",
        scryfall_id: "d237e169-f152-4ddf-a5a1-32ca46cfa16d",
        printed: PrintedCharacteristics {
            name: "Fierce Empath".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::search_library(
                            CardPredicate {
                                min_mana_value: Some(6),
                                ..CardPredicate::creature()
                            },
                            1.into(),
                            vec![Effect::return_to_hand(ObjectSelector::Selected)],
                        )],
                        vec![],
                    ],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
