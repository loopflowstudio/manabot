//! Dark Ascension.
use managym_state::{
    ability::{Ability, Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![CardDefinition {
        key: "dka.drogskol_reaver",
        scryfall_id: "af2d9e0b-6433-40a2-9847-9fa4e3c008c4",
        printed: PrintedCharacteristics {
            name: "Drogskol Reaver".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            keywords: Keywords::of(&[Keyword::Flying, Keyword::DoubleStrike, Keyword::Lifelink]),
            power: Some(3),
            toughness: Some(5),
            abilities: vec![Ability::when_you_gain_life(Effect::DrawCards { count: EffectValue::Fixed(1) })],
            text_box: "Flying\nDouble strike (This creature deals both first-strike and regular combat damage.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhenever you gain life, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
