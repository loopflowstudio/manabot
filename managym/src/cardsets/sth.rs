//! Stronghold.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![youthful_knight()]
}

fn youthful_knight() -> CardDefinition {
    native_creature(
        "Youthful Knight",
        "1W",
        &["Human", "Knight"],
        2,
        1,
        Keywords {
            first_strike: true,
            ..Default::default()
        },
        "First strike",
    )
}
