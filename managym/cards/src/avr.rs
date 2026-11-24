//! Avacyn Restored.
use managym_state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![maalfeld_twins()]
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
