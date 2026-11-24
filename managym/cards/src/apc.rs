//! Apocalypse.
use managym_state::{
    ability::{Ability, Actor, Effect, PlayerSelector, TriggerCondition, TurnStep},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![phyrexian_arena()]
}

fn phyrexian_arena() -> CardDefinition {
    CardDefinition {
        key: "apc.phyrexian_arena",
        scryfall_id: "84e19975-e3e1-453b-b902-a1b1fc1d8504",
        printed: PrintedCharacteristics {
            name: "Phyrexian Arena".into(),
            types: CardTypes::new([CardType::Enchantment]),
            mana_cost: Some(ManaCost::parse("{1}{B}{B}")),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::BeginningOf {
                    step: TurnStep::Upkeep,
                    whose: Actor::You,
                },
                intervening_if: None,
                effects: vec![
                    Effect::DrawCards { count: 1.into() },
                    Effect::LoseLife {
                        player: PlayerSelector::You,
                        amount: 1.into(),
                    },
                ],
            }],
            text_box: "At the beginning of your upkeep, you draw a card and lose 1 life.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
