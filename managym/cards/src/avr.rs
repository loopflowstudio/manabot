//! Avacyn Restored.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![driver_of_the_dead()]
}

fn driver_of_the_dead() -> CardDefinition {
    let text_box = "When this creature dies, return target creature card with mana value 2 or less from your graveyard to the battlefield.";
    CardDefinition {
        key: "avr.driver_of_the_dead",
        scryfall_id: "56113cde-4210-46be-bd53-8966c36ef2a3",
        printed: PrintedCharacteristics {
            name: "Driver of the Dead".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::return_to_battlefield(
                    ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            max_mana_value: Some(2),
                            ..CardPredicate::creature()
                        },
                    }),
                    false,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
