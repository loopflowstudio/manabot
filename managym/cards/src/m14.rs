//! Magic 2014.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::{Color, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mindsparker()]
}

fn mindsparker() -> CardDefinition {
    let text_box = "First strike (This creature deals combat damage before creatures without first strike.)\nWhenever an opponent casts a white or blue instant or sorcery spell, this creature deals 2 damage to that player.";
    CardDefinition {
        key: "m14.mindsparker",
        scryfall_id: "a94295dc-d078-4f3f-9856-bd0a1899a9ca",
        printed: PrintedCharacteristics {
            name: "Mindsparker".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elemental".into()],
            power: Some(3),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::Opponent,
                    CardPredicate {
                        colors_any: vec![Color::White, Color::Blue],
                        card_types_any: vec![CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::TriggeringPlayer,
                    amount: EffectValue::Fixed(2),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
