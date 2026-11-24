//! Gatecrash.
use managym_state::{
    ability::{Ability, AttackSize, Effect, ObjectSelector, Per, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

use super::rtr::guildgate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        aurelia_the_warleader(),
        boros_guildgate(),
        dimir_guildgate(),
        gruul_guildgate(),
        orzhov_guildgate(),
        simic_guildgate(),
    ]
}

fn boros_guildgate() -> CardDefinition {
    guildgate(
        "gtc.boros_guildgate",
        "a0b447a8-524b-4bda-b975-7e194fd741fb",
        "Boros Guildgate",
        ["R", "W"],
    )
}

fn dimir_guildgate() -> CardDefinition {
    guildgate(
        "gtc.dimir_guildgate",
        "4951bf75-1a88-4c85-b4e9-063d84f1dabf",
        "Dimir Guildgate",
        ["U", "B"],
    )
}

fn gruul_guildgate() -> CardDefinition {
    guildgate(
        "gtc.gruul_guildgate",
        "99c54269-8798-4023-836f-640103e08ce0",
        "Gruul Guildgate",
        ["R", "G"],
    )
}

fn orzhov_guildgate() -> CardDefinition {
    guildgate(
        "gtc.orzhov_guildgate",
        "000d609c-deb7-4bd7-9c1d-e20fb3ed4f5f",
        "Orzhov Guildgate",
        ["W", "B"],
    )
}

fn simic_guildgate() -> CardDefinition {
    guildgate(
        "gtc.simic_guildgate",
        "1ce3f6f2-c52c-4fb8-afa0-b1ea723bb4c6",
        "Simic Guildgate",
        ["G", "U"],
    )
}

fn aurelia_the_warleader() -> CardDefinition {
    let text_box = "Flying, vigilance, haste\nWhenever Aurelia attacks for the first time each turn, untap all creatures you control. After this phase, there is an additional combat phase.";
    CardDefinition {
        key: "gtc.aurelia_the_warleader",
        scryfall_id: "4ec18e35-05e4-4bfc-b32b-c3e71c95a71d",
        printed: PrintedCharacteristics {
            name: "Aurelia, the Warleader".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance, Keyword::Haste]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                    per: Per::Object,
                    attackers: AttackSize::Any,
                    nth: Some(1),
                },
                intervening_if: None,
                effects: vec![
                    Effect::Untap {
                        what: ObjectSelector::each_you_control(CardPredicate::creature()),
                    },
                    Effect::AdditionalCombatPhase { attackers: None },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
