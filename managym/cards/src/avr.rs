//! Avacyn Restored.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![maalfeld_twins(), driver_of_the_dead()]
}

fn maalfeld_twins() -> CardDefinition {
    let text_box = "When this creature dies, create two 2/2 black Zombie creature tokens.";
    CardDefinition {
        key: "avr.maalfeld_twins",
        scryfall_id: "c63dd203-bce9-4ab7-8a0c-059d19d384e9",
        printed: PrintedCharacteristics {
            name: "Maalfeld Twins".into(),
            mana_cost: Some(ManaCost::parse("{5}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(4),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::create_tokens("token.zombie_2_2_b", 2)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
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
