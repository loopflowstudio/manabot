//! Visions.
use crate::state::{
    ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![man_o_war()]
}

fn man_o_war() -> CardDefinition {
    CardDefinition {
        name: "Man-o'-War".to_string(),
        mana_cost: Some(ManaCost::parse("2U")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Jellyfish".to_string()],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::ReturnToHand {
                target: TargetSpec::Creature,
            }],
        }],
        text_box: "When this creature enters, return target creature to its owner's hand."
            .to_string(),
        power: Some(2),
        toughness: Some(2),
        ..Default::default()
    }
}
