//! Magic Origins.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![boggart_brute()]
}

fn boggart_brute() -> CardDefinition {
    native_creature(
        "Boggart Brute",
        "2R",
        &["Goblin", "Warrior"],
        3,
        2,
        Keywords {
            menace: true,
            ..Default::default()
        },
        "Menace (This creature can't be blocked except by two or more creatures.)",
    )
}
