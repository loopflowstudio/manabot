//! Theros Beyond Death.
use managym_state::{
    ability::{
        Ability, Actor, CompareOp, Condition, Effect, ObjectSelector, TriggerCondition, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![nessian_hornbeetle()]
}

fn nessian_hornbeetle() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, if you control another creature with power 4 or greater, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "thb.nessian_hornbeetle",
        scryfall_id: "8200fcda-e30c-460f-9964-47e657b7c758",
        printed: PrintedCharacteristics {
            name: "Nessian Hornbeetle".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Insect".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: Some(Condition::compare(
                    ObjectSelector::each_other_you_control(CardPredicate {
                        min_power: Some(4),
                        ..CardPredicate::creature()
                    })
                    .count(),
                    CompareOp::Ge,
                    1,
                )),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
