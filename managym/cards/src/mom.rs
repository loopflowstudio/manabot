//! March of the Machine.

use managym_state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![wary_thespian()]
}

fn wary_thespian() -> CardDefinition {
    let text_box = "When this creature enters or dies, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "mom.wary_thespian",
        scryfall_id: "675b29bf-0b64-410f-9a92-c88e5615c27f",
        printed: PrintedCharacteristics {
            name: "Wary Thespian".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Druid".into()],
            power: Some(3),
            toughness: Some(1),
            abilities: vec![
                Ability::when_enters(TriggerSubject::This, vec![Effect::Surveil { count: 1 }]),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::Surveil { count: 1 }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
