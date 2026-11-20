//! Gatecrash.
use managym_state::card::CardDefinition;

use super::rtr::guildgate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
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
