//! Zendikar Rising.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        broken_wings(),
        highborn_vampire(),
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
                abilities: vec![Ability::when_you_gain_life(Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                })],
                text_box: "Whenever you gain life, each opponent loses 1 life.".into(),
                ..Default::default()
            },
            ..Default::default()
        },
    ]
}

fn highborn_vampire() -> CardDefinition {
    CardDefinition {
        key: "znr.highborn_vampire",
        scryfall_id: "24c40082-516e-4381-a4cc-e61c5a9a6cac",
        printed: PrintedCharacteristics {
            name: "Highborn Vampire".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(3),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn broken_wings() -> CardDefinition {
    CardDefinition {
        key: "znr.broken_wings",
        scryfall_id: "c0fc2dfd-85b0-4add-be18-b39549235921",
        printed: PrintedCharacteristics {
            name: "Broken Wings".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                                ..Default::default()
                            },
                            CardPredicate {
                                flying: true,
                                ..CardPredicate::creature()
                            },
                        ],
                        ..Default::default()
                    },
                },
            ))],
            text_box: "Destroy target artifact, enchantment, or creature with flying.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
