//! The one rules world this engine implements.
//!
//! A world names rules meaning plus observation/action shape (see `WORLDS.md`).
//! Any running engine knows its world: [`WORLD_VERSION`]. The world's cards are
//! the typed definitions in `cardsets/`; a card is *supported* when it has a
//! current qualification there (independent witnesses bound to its exact
//! authored closure). Native legacy cards and preserved-but-unqualified
//! definitions still run in authored setups but are not supported for
//! arbitrary decks.
//!
//! This is an eligibility check only. Deck construction policy (counts,
//! formats, sideboards) belongs to the deck-admission caller.
//!
//! Older worlds run from the git revision that recorded them; this engine
//! carries no rules compatibility shims for earlier worlds.

use std::collections::BTreeSet;

use crate::semantic::{content_pool, Eligibility, IrError, PoolDefinition};

/// Rules and observation/action meaning; see `WORLDS.md` for compatibility.
pub const WORLD_VERSION: &str = "w16";

/// The pool holding every card this world defines.
pub const WORLD_POOL: &str = "tla-pool-v1";

/// Whether this world supports `name` for deck entry. Unknown names are
/// unsupported rather than errors.
pub fn supports_card(name: &str) -> Result<bool, IrError> {
    Ok(card(name)?.is_some_and(|row| row.eligibility == Eligibility::Eligible))
}

/// The distinct card names in `names` this world does not support, sorted.
/// A deck is supported exactly when this is empty.
pub fn unsupported_cards<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<String>, IrError> {
    let mut unsupported = BTreeSet::new();
    for name in names {
        if !supports_card(name)? {
            unsupported.insert(name.to_owned());
        }
    }
    Ok(unsupported.into_iter().collect())
}

/// This world's descriptor row for `name`, including the reason for an
/// unsupported card.
pub fn card(name: &str) -> Result<Option<PoolDefinition>, IrError> {
    Ok(content_pool(WORLD_POOL)?
        .definitions
        .into_iter()
        .find(|row| row.registry_name == name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_cards_are_qualified_and_decks_report_the_rest() {
        assert!(supports_card("Avatar Enthusiasts").unwrap());
        assert!(supports_card("Forest").unwrap());
        // Authored but awaiting an independent prerequisite.
        assert!(!supports_card("Water Tribe Rallier").unwrap());
        // Native legacy card with no typed definition.
        assert!(!supports_card("Counterspell").unwrap());
        assert!(!supports_card("Not A Card").unwrap());
        let deck = [
            "Forest",
            "Avatar Enthusiasts",
            "Water Tribe Rallier",
            "Forest",
        ];
        assert_eq!(unsupported_cards(deck).unwrap(), ["Water Tribe Rallier"]);
        assert!(card("Water Tribe Rallier")
            .unwrap()
            .unwrap()
            .reason
            .is_some());
    }
}
