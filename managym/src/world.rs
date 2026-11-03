//! The one rules world this engine implements.
//!
//! A world names rules meaning plus observation/action shape (see `WORLDS.md`).
//! Any running engine knows its world: [`WORLD_VERSION`]. The world's cards are
//! the one catalog ([`default_content_pack`]): every definition in `cardsets/`,
//! whatever its set. A card is *supported* when its set file defines it and
//! does not mark it incomplete. Incomplete definitions still run in the fixed
//! setups that name them but are not supported for arbitrary decks.
//! [`set_cards`] narrows the pool to chosen sets for deck generation or
//! validation.
//!
//! This is an eligibility check only. Deck construction policy (counts,
//! formats, sideboards) belongs to the deck-admission caller.
//!
//! Older worlds run from the git revision that recorded them; this engine
//! carries no rules compatibility shims for earlier worlds.

use std::collections::BTreeSet;

use crate::cards::{content_pack::default_content_pack, world_pool, Eligibility, PoolDefinition};

/// Rules and observation/action meaning; see `WORLDS.md` for compatibility.
pub const WORLD_VERSION: &str = "w17";

/// The public name of the world pool ([`world_pool`]) and of a deck submitted
/// to the world for admission (`PlayerConfig::submitted`); frozen evidence
/// and the Python API (`WORLD_SETUP`, `content_pool("world")`) read it.
pub const WORLD_POOL: &str = "world";

/// What decks and checkpoints bind: the rules world and the digest of its
/// cards, which changes whenever any card definition changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldIdentity {
    pub world_version: String,
    pub card_digest: String,
}

/// The installed world's identity: its version and the catalog's digest.
pub fn world_identity() -> WorldIdentity {
    WorldIdentity {
        world_version: WORLD_VERSION.to_owned(),
        card_digest: default_content_pack().content_digest(),
    }
}

/// Whether this world supports `name` for deck entry. Unknown names are
/// unsupported rather than errors.
pub fn supports_card(name: &str) -> bool {
    unsupported_cards([name]).is_empty()
}

/// The distinct card names in `names` this world does not support, sorted.
/// A deck is supported exactly when this is empty; the pool row of an
/// unsupported definition gives the reason.
pub fn unsupported_cards<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let supported = world_pool()
        .definitions
        .iter()
        .filter(|row| row.eligibility == Eligibility::Eligible)
        .map(|row| row.registry_name.as_str())
        .collect::<BTreeSet<_>>();
    names
        .into_iter()
        .filter(|name| !supported.contains(name))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// This world's descriptor rows for cards from `sets` (key prefixes such
/// as `"tla"`), sorted by name. Deck generators and validators apply this
/// filter; the pool itself spans every set.
pub fn set_cards(sets: &[&str]) -> Vec<PoolDefinition> {
    world_pool()
        .definitions
        .iter()
        .filter(|row| {
            row.key.as_deref().is_some_and(|key| {
                key.split_once('.')
                    .is_some_and(|(set, _)| sets.contains(&set))
            })
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defined_cards_are_supported_and_decks_report_the_rest() {
        assert!(supports_card("Avatar Enthusiasts"));
        assert!(supports_card("Forest"));
        assert!(supports_card("Lightning Bolt"));
        // Defined, but marked incomplete beside its definition.
        assert!(!supports_card("Water Tribe Rallier"));
        assert!(supports_card("Counterspell"));
        assert!(!supports_card("Crossroads of Destiny"));
        assert!(!supports_card("Not A Card"));
        let deck = [
            "Forest",
            "Avatar Enthusiasts",
            "Water Tribe Rallier",
            "Forest",
        ];
        assert_eq!(unsupported_cards(deck), ["Water Tribe Rallier"]);
    }

    #[test]
    fn world_identity_names_the_version_and_card_digest() {
        let identity = world_identity();
        assert_eq!(identity.world_version, WORLD_VERSION);
        assert_eq!(
            identity.card_digest,
            default_content_pack().content_digest()
        );
        assert_eq!(identity.card_digest, world_pool().content_digest);
    }

    #[test]
    fn set_filter_selects_cards_by_key_prefix() {
        let names = |sets: &[&str]| {
            set_cards(sets)
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
