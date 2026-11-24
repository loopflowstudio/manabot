//! Ixalan.
use managym_state::{
    ability::{Ability, Condition, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![storm_fleet_spy()]
}

fn storm_fleet_spy() -> CardDefinition {
    let text_box = "Raid — When this creature enters, if you attacked this turn, draw a card.";
    CardDefinition {
        key: "xln.storm_fleet_spy",
        scryfall_id: "f7c33ef4-60bb-4f95-92a5-7abedaac6767",
        printed: PrintedCharacteristics {
            name: "Storm Fleet Spy".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
