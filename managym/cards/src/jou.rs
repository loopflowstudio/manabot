//! Journey into Nyx.
use managym_state::card::CardDefinition;

use super::ths::temple;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![temple_of_epiphany(), temple_of_malady()]
}

fn temple_of_epiphany() -> CardDefinition {
    temple(
        "jou.temple_of_epiphany",
        "b882c6bf-b795-49fe-8242-a928aadb6f13",
        "Temple of Epiphany",
        ["U", "R"],
    )
}

fn temple_of_malady() -> CardDefinition {
    temple(
        "jou.temple_of_malady",
        "f30220f1-1992-4b5c-9e13-1762fb673155",
        "Temple of Malady",
        ["B", "G"],
    )
}
