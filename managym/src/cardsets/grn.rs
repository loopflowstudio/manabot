//! Guilds of Ravnica.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![healer_s_hawk()]
}

fn healer_s_hawk() -> CardDefinition {
    native_creature(
        "Healer's Hawk",
        "W",
        &["Bird"],
        1,
        1,
        Keywords {
            flying: true,
            lifelink: true,
            ..Default::default()
        },
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)",
    )
}
