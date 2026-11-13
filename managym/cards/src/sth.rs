//! Stronghold.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, Scope, TriggerCondition, TriggerSubject},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        Restriction, StaticAbility, StaticRestriction, StaticScope,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![intruder_alarm(), youthful_knight()]
}

fn youthful_knight() -> CardDefinition {
    let text_box = "First strike";

    CardDefinition {
        key: "sth.youthful_knight",
        scryfall_id: "fcd3b7a7-c5e8-4179-ade3-31625620f3a9",
        printed: PrintedCharacteristics {
            name: "Youthful Knight".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Knight".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn intruder_alarm() -> CardDefinition {
    let text_box = "Creatures don't untap during their controllers' untap steps.\nWhenever a creature enters, untap all creatures.";
    CardDefinition {
        key: "sth.intruder_alarm",
        scryfall_id: "7e55f653-0952-4713-8d43-ca50acff9e3b",
        printed: PrintedCharacteristics {
            name: "Intruder Alarm".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::All(CardPredicate::creature()),
                kind: Restriction::DoesntUntap,
                unless: None,
            })],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another(CardPredicate::creature()),
                },
                effects: vec![Effect::untap(ObjectSelector::each(
                    Scope::Battlefield,
                    CardPredicate::creature(),
                ))],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
