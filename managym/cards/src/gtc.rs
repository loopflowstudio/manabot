//! Gatecrash.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

use super::rtr::guildgate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
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
