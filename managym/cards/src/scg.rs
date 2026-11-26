//! Scourge.

use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, Per, PlayerSelector, TriggerCondition,
        TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dragon_mage(), fierce_empath()]
}

fn dragon_mage() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature deals combat damage to a player, each player discards their hand, then draws seven cards.";
    CardDefinition {
        key: "scg.dragon_mage",
        scryfall_id: "7687a201-0ecc-4739-86e3-3b4090d345a8",
        printed: PrintedCharacteristics {
            name: "Dragon Mage".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into(), "Wizard".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Some(Actor::AnyPlayer),
                    per: Per::Object,
                },
                intervening_if: None,
                effects: vec![
                    Effect::ForPlayer {
                        player: PlayerSelector::EachPlayer,
                        effects: vec![Effect::move_to(
                            ObjectSelector::Zone(ZoneType::Hand),
                            ZoneType::Graveyard,
                        )],
                    },
                    Effect::ForPlayer {
                        player: PlayerSelector::EachPlayer,
                        effects: vec![Effect::DrawCards { count: 7.into() }],
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fierce_empath() -> CardDefinition {
    let text_box = "When this creature enters, you may search your library for a creature card with mana value 6 or greater, reveal it, put it into your hand, then shuffle.";
    CardDefinition {
        key: "scg.fierce_empath",
        scryfall_id: "d237e169-f152-4ddf-a5a1-32ca46cfa16d",
        printed: PrintedCharacteristics {
            name: "Fierce Empath".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::search_library(
                            CardPredicate {
                                min_mana_value: Some(6),
                                ..CardPredicate::creature()
                            },
                            1.into(),
                            vec![Effect::return_to_hand(ObjectSelector::Selected)],
                        )],
                        vec![],
                    ],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
