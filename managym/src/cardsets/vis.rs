//! Visions.
use crate::state::{
    ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![man_o_war()]
}

fn man_o_war() -> CardDefinition {
    let text_box = "When this creature enters, return target creature to its owner's hand.";

    CardDefinition {
        key: "vis.man_o_war",
        scryfall_id: "4dbf9bf9-75cd-4b25-a3a1-43b7e029700b",
        name: "Man-o'-War".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Jellyfish".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::ReturnToHand {
                target: TargetSpec::Creature,
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
