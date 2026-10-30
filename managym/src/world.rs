//! The one rules world this engine implements.
//!
//! A world names rules meaning plus observation/action shape (see `WORLDS.md`).
//! Any running engine knows its world: [`WORLD_VERSION`]. The world's cards are
//! every definition in `cardsets/`, whatever its set; a card is *supported*
//! when its set file defines it and does not mark it incomplete. Incomplete
//! definitions still run in the fixed setups that name them but are not
//! supported for arbitrary decks. [`set_cards`] narrows the pool to chosen
//! sets for deck generation or validation.
//!
//! This is an eligibility check only. Deck construction policy (counts,
//! formats, sideboards) belongs to the deck-admission caller.
//!
//! Older worlds run from the git revision that recorded them; this engine
//! carries no rules compatibility shims for earlier worlds.

use std::collections::BTreeSet;

use crate::cards::{content_pool, ContentError, Eligibility, PoolDefinition};

/// Rules and observation/action meaning; see `WORLDS.md` for compatibility.
pub const WORLD_VERSION: &str = "w18";

/// The pool describing the default pack: every card this world defines, from
/// every set, with the default pack's `CardDefId`s and digest.
pub const WORLD_POOL: &str = "world";

/// Whether this world supports `name` for deck entry. Unknown names are
/// unsupported rather than errors.
pub fn supports_card(name: &str) -> Result<bool, ContentError> {
    Ok(unsupported_cards([name])?.is_empty())
}

/// The distinct card names in `names` this world does not support, sorted.
/// A deck is supported exactly when this is empty; the pool row of an
/// unsupported definition gives the reason.
pub fn unsupported_cards<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<String>, ContentError> {
    let pool = content_pool(WORLD_POOL)?;
    let supported = pool
        .definitions
        .iter()
        .filter(|row| row.eligibility == Eligibility::Eligible)
        .map(|row| row.registry_name.as_str())
        .collect::<BTreeSet<_>>();
    Ok(names
        .into_iter()
        .filter(|name| !supported.contains(name))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect())
}

/// This world's descriptor rows for cards from `sets` (key prefixes such
/// as `"tla"`), sorted by name. Deck generators and validators apply this
/// filter; the pool itself spans every set.
pub fn set_cards(sets: &[&str]) -> Result<Vec<PoolDefinition>, ContentError> {
    Ok(content_pool(WORLD_POOL)?
        .definitions
        .into_iter()
        .filter(|row| {
            row.key.as_deref().is_some_and(|key| {
                key.split_once('.')
                    .is_some_and(|(set, _)| sets.contains(&set))
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defined_cards_are_supported_and_decks_report_the_rest() {
        assert!(supports_card("Avatar Enthusiasts").unwrap());
        assert!(supports_card("Forest").unwrap());
        assert!(supports_card("Lightning Bolt").unwrap());
        // Defined, but marked incomplete beside its definition.
        assert!(!supports_card("Water Tribe Rallier").unwrap());
        assert!(supports_card("Counterspell").unwrap());
        assert!(!supports_card("Crossroads of Destiny").unwrap());
        assert!(!supports_card("Not A Card").unwrap());
        let deck = [
            "Forest",
            "Avatar Enthusiasts",
            "Water Tribe Rallier",
            "Forest",
        ];
        assert_eq!(unsupported_cards(deck).unwrap(), ["Water Tribe Rallier"]);
    }

    #[test]
    fn set_filter_selects_cards_by_key_prefix() {
        let names = |sets: &[&str]| {
            set_cards(sets)
                .unwrap()
                .into_iter()
                .map(|row| row.registry_name)
                .collect::<Vec<_>>()
        };
        let tla = names(&["tla"]);
        assert!(tla.contains(&"Avatar Enthusiasts".to_owned()));
        assert!(!tla.contains(&"Forest".to_owned()));
        assert!(!tla.contains(&"Pop Quiz".to_owned()));
        let with_basics = names(&["tla", "lea"]);
        assert!(with_basics.contains(&"Forest".to_owned()));
        assert!(with_basics.contains(&"Avatar Enthusiasts".to_owned()));
        assert!(names(&["zzz"]).is_empty());
    }
}
