//! Portal.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![wind_drake(), raging_goblin()]
}

fn wind_drake() -> CardDefinition {
    native_creature(
        "Wind Drake",
        "2U",
        &["Drake"],
        2,
        2,
        Keywords {
            flying: true,
            ..Default::default()
        },
        "Flying",
    )
}

fn raging_goblin() -> CardDefinition {
    native_creature(
        "Raging Goblin",
        "R",
        &["Goblin", "Berserker"],
        1,
        1,
        Keywords {
            haste: true,
            ..Default::default()
        },
        "Haste (This creature can attack and {T} as soon as it comes under your control.)",
    )
}
