//! Shadows over Innistrad.
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![tireless_tracker()]
}

fn tireless_tracker() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, investigate. (Create a Clue token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a Clue, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "soi.tireless_tracker",
        scryfall_id: "ee8e9928-d9b2-4570-adb8-44b34115decd",
        name: "Tireless Tracker".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::CreateToken {
                    token_name: "Clue".into(),
                    count: 1,
                    tapped_and_attacking: false,
                    attach_source: false,
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::Sacrificed {
                    predicate: CardPredicate {
                        subtype: Some("Clue".into()),
                        ..Default::default()
                    },
                    controller_only: true,
                },
                effects: vec![Effect::PutCountersOnSource {
                    count: 1,
                    kind: CounterKind::PlusOnePlusOne,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
