//! The Dark.
use managym_state::{
    ability::{Ability, Actor, Effect, TriggerCondition, TurnStep},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ball_lightning()]
}

fn ball_lightning() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nHaste (This creature can attack and {T} as soon as it comes under your control.)\nAt the beginning of the end step, sacrifice this creature.";
    CardDefinition {
        key: "drk.ball_lightning",
        scryfall_id: "c1ba83ab-83f5-421d-bba1-0f925870b5c8",
        printed: PrintedCharacteristics {
            name: "Ball Lightning".into(),
            mana_cost: Some(ManaCost::parse("{R}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elemental".into()],
            power: Some(6),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Trample, Keyword::Haste]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                effects: vec![Effect::sacrifice_this()],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
