//! Innistrad.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![typhoid_rats()]
}

fn typhoid_rats() -> CardDefinition {
    native_creature(
        "Typhoid Rats",
        "B",
        &["Rat"],
        1,
        1,
        Keywords {
            deathtouch: true,
            ..Default::default()
        },
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)",
    )
}
