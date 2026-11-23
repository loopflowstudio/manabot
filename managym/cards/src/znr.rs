//! Zendikar Rising.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![marauding_blight_priest()]
}

fn marauding_blight_priest() -> CardDefinition {
    CardDefinition {
        key: "znr.marauding_blight_priest",
        scryfall_id: "730ddbcd-0814-4e22-85e9-78b0878324b6",
        printed: PrintedCharacteristics {
            name: "Marauding Blight-Priest".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::GainsLife { who: Actor::You },
                intervening_if: None,
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                }],
            }],
            text_box: "Whenever you gain life, each opponent loses 1 life.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
