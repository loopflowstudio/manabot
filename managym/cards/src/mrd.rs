//! Mirrodin.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        solemn_simulacrum(),
        fireshrieker(),
        leonin_skyhunter(),
        gilded_lotus(),
    ]
}

fn gilded_lotus() -> CardDefinition {
    let text_box = "{T}: Add three mana of any one color.";
    CardDefinition {
        key: "mrd.gilded_lotus",
        scryfall_id: "a1d5e4c8-dfd0-45bc-8000-ebfaccfefec3",
        printed: PrintedCharacteristics {
            name: "Gilded Lotus".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(
                    ["WWW", "UUU", "BBB", "RRR", "GGG"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn leonin_skyhunter() -> CardDefinition {
    let text_box = "Flying";
    CardDefinition {
        key: "mrd.leonin_skyhunter",
        scryfall_id: "275a47e1-816c-44f9-bd05-b8b56410436f",
        printed: PrintedCharacteristics {
            name: "Leonin Skyhunter".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Knight".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fireshrieker() -> CardDefinition {
    let text_box = "Equipped creature has double strike. (It deals both first-strike and regular combat damage.)\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "mrd.fireshrieker",
        scryfall_id: "8da0fcc6-6209-4b8e-997d-ad3cc4ff0856",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::Attached,
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::DoubleStrike,
                ]))],
            })],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            name: "Fireshrieker".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            mana_cost: Some(ManaCost::parse("{3}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn solemn_simulacrum() -> CardDefinition {
    let text_box = "When this creature enters, you may search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.\nWhen this creature dies, you may draw a card.";
    CardDefinition {
        key: "mrd.solemn_simulacrum",
        scryfall_id: "00f9955f-a522-47bf-b064-92dd21a76b18",
        printed: PrintedCharacteristics {
            name: "Solemn Simulacrum".into(),
            mana_cost: Some(ManaCost::parse("{4}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Golem".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![
                Ability::when_enters(
                    TriggerSubject::This,
                    vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::search_library(
                                CardPredicate::basic_land(),
                                1.into(),
                                vec![Effect::return_to_battlefield(
                                    ObjectSelector::Selected,
                                    true,
                                )],
                            )],
                            vec![],
                        ],
                    }],
                ),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::Modal {
                        modes: vec![vec![Effect::DrawCards { count: 1.into() }], vec![]],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
