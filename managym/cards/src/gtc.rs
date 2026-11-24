//! Gatecrash.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

use super::rtr::guildgate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        boros_charm(),
        aetherize(),
        angelic_edict(),
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

fn angelic_edict() -> CardDefinition {
    CardDefinition {
        key: "gtc.angelic_edict",
        scryfall_id: "e24b62d3-c200-4330-a255-92d77f01ba44",
        printed: PrintedCharacteristics {
            name: "Angelic Edict".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::exile(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Enchantment],
                        ..Default::default()
                    },
                },
            ))],
            text_box: "Exile target creature or enchantment.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn aetherize() -> CardDefinition {
    CardDefinition {
        key: "gtc.aetherize",
        scryfall_id: "33303859-c6e0-4ebd-bb5f-44be7f5d7459",
        printed: PrintedCharacteristics {
            name: "Aetherize".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::return_to_hand(ObjectSelector::each(
                Scope::Battlefield,
                CardPredicate {
                    attacking: true,
                    ..CardPredicate::creature()
                },
            ))],
            text_box: "Return all attacking creatures to their owner's hand.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn boros_charm() -> CardDefinition {
    let text_box = "Choose one —\n• Boros Charm deals 4 damage to target player or planeswalker.\n• Permanents you control gain indestructible until end of turn.\n• Target creature gains double strike until end of turn.";
    CardDefinition {
        key: "gtc.boros_charm",
        scryfall_id: "d4ddf9cc-40a7-4b4f-bb51-b08171453c9a",
        printed: PrintedCharacteristics {
            name: "Boros Charm".into(),
            mana_cost: Some(ManaCost::parse("{R}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        4,
                        ObjectSelector::Target(TargetSpec::PlayerOrPlaneswalker),
                    )],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each_you_control(CardPredicate::permanent()),
                        Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                    )],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::Creature),
                        Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
