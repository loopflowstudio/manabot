//! Final Fantasy.
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![sazh_s_chocobo()]
}

fn sazh_s_chocobo() -> CardDefinition {
    let text_box =
        "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "fin.sazh_s_chocobo",
        scryfall_id: "dda6b4d0-1b60-46b0-b321-b9ffe15afff4",
        name: "Sazh's Chocobo".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bird".into()],
        power: Some(0),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnyYouControl(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::PutCountersOnSource {
                count: 1,
                kind: CounterKind::PlusOnePlusOne,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
