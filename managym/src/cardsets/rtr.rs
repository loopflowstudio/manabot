//! Return to Ravnica.
use super::native_creature;
use crate::state::card::{CardDefinition, Keywords};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![fencing_ace()]
}

fn fencing_ace() -> CardDefinition {
    native_creature(
        "Fencing Ace",
        "1W",
        &["Human", "Soldier"],
        1,
        1,
        Keywords {
            double_strike: true,
            ..Default::default()
        },
        "Double strike (This creature deals both first-strike and regular combat damage.)",
    )
}
