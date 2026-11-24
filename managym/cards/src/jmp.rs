//! Jumpstart.
use managym_state::{
    ability::{Ability, Effect, TriggerSubject},
    card::{
        CardDefinition, CardType, CardTypes, LayerValue, PrintedCharacteristics, StaticAbility,
        StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![corsair_captain(), release_the_dogs()]
}

fn release_the_dogs() -> CardDefinition {
    let text_box = "Create four 1/1 white Dog creature tokens.";
    CardDefinition {
        key: "jmp.release_the_dogs",
        scryfall_id: "7df3cd89-02c9-4a1c-9a8a-d17a0b1030c9",
        printed: PrintedCharacteristics {
            name: "Release the Dogs".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens("token.dog_1_1_w", 4)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn corsair_captain() -> CardDefinition {
    let text_box = "When this creature enters, create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")\nOther Pirates you control get +1/+1.";
    CardDefinition {
        key: "jmp.corsair_captain",
        scryfall_id: "a9b016d4-ddf6-47d6-b934-a0b979b60680",
        printed: PrintedCharacteristics {
            name: "Corsair Captain".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.treasure", 1)],
            )],
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Pirate")),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
