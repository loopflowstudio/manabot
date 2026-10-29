//! War of the Spark.
use crate::state::{
    ability::{Ability, Effect, TargetSpec, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mayhem_devil()]
}

fn mayhem_devil() -> CardDefinition {
    let text_box =
        "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.";

    CardDefinition {
        key: "war.mayhem_devil",
        scryfall_id: "17416926-168b-49b3-9231-acbb8f8a1d13",
        name: "Mayhem Devil".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Devil".into()],
        power: Some(3),
        toughness: Some(3),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Sacrificed {
                predicate: CardPredicate::default(),
                controller_only: false,
            },
            effects: vec![Effect::DealDamage {
                amount: 1,
                target: TargetSpec::CreatureOrPlayer,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
