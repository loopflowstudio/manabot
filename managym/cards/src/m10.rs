//! Magic 2010.
use managym_state::{
    ability::{Ability, Actor, Effect, ObjectSelector, PaymentCost, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::{Color, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mold_adder()]
}

fn mold_adder() -> CardDefinition {
    let text_box = "Whenever an opponent casts a blue or black spell, you may put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "m10.mold_adder",
        scryfall_id: "a216a729-6283-4c2b-90fe-ec8f3b9c570f",
        printed: PrintedCharacteristics {
            name: "Mold Adder".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Fungus".into(), "Snake".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::Opponent,
                    CardPredicate {
                        colors_any: vec![Color::Blue, Color::Black],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
