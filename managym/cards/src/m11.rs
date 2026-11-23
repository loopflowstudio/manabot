//! Magic 2011.
use managym_state::{
    ability::{Ability, Actor, Effect, ObjectSelector, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![CardDefinition {
        key: "m11.ajanis_pridemate",
        scryfall_id: "f70d1452-6b61-4c63-841f-4256ac498e9f",
        printed: PrintedCharacteristics {
            name: "Ajani's Pridemate".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::GainsLife { who: Actor::You },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: "Whenever you gain life, put a +1/+1 counter on this creature.".into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
