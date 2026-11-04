//! The world pool: one descriptor row per catalog definition, with its support.
//!
//! The pool ([`world_pool`], named [`crate::world::WORLD_POOL`]) describes the
//! catalog ([`default_content_pack`]), so it shares the catalog's `CardDefId`s
//! and digest; deck import and admission read only it. A defined card is
//! eligible regardless of known rules bugs;
//! tokens, back faces and rules fixtures are never deck entries. Setup
//! construction policy belongs to the admission caller, not this catalog.

use std::sync::OnceLock;

use super::ContentError;
use crate::{cards::content_pack::default_content_pack, state::card::CardDefId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eligibility {
    Eligible,
    Unsupported,
}

#[derive(Clone, Debug)]
pub struct PoolDefinition {
    pub card_def_id: CardDefId,
    /// Set-file key (`tla.avatar_enthusiasts`); `None` for rules fixtures.
    pub key: Option<String>,
    /// Scryfall card-object id of the definition's printing.
    pub scryfall_id: Option<String>,
    pub registry_name: String,
    pub eligibility: Eligibility,
    pub reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ContentPool {
    pub world_version: String,
    pub content_digest: String,
    /// Sorted by registry name.
    pub definitions: Vec<PoolDefinition>,
}

fn describe() -> ContentPool {
    let content = default_content_pack();
    let mut definitions = content
        .definition_entries()
        .map(|(card_def_id, definition)| {
            let reason = if definition.is_token {
                Some("A token, created by its card; not a deck entry".to_owned())
            } else if content.front_face(card_def_id).is_some() {
                Some("The back face of a double-faced card; not a deck entry".to_owned())
            } else if definition.key.is_empty() {
                Some("A rules fixture, not a printed card".to_owned())
            } else {
                None
            };
            PoolDefinition {
                card_def_id,
                key: (!definition.key.is_empty()).then(|| definition.key.to_owned()),
                scryfall_id: (!definition.scryfall_id.is_empty())
                    .then(|| definition.scryfall_id.to_owned()),
                registry_name: definition.name.clone(),
                eligibility: if reason.is_none() {
                    Eligibility::Eligible
                } else {
                    Eligibility::Unsupported
                },
                reason,
            }
        })
        .collect::<Vec<_>>();
    definitions.sort_by(|a, b| a.registry_name.cmp(&b.registry_name));
    ContentPool {
        world_version: crate::WORLD_VERSION.to_owned(),
        content_digest: content.content_digest(),
        definitions,
    }
}

/// Every catalog card and whether the world supports it for deck entry.
pub fn world_pool() -> &'static ContentPool {
    static POOL: OnceLock<ContentPool> = OnceLock::new();
    POOL.get_or_init(describe)
}

/// The pool by its public name; only [`crate::world::WORLD_POOL`] exists.
pub fn content_pool(name: &str) -> Result<&'static ContentPool, ContentError> {
    if name == crate::world::WORLD_POOL {
        Ok(world_pool())
    } else {
        Err(ContentError(format!("unknown content pool {name:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row<'a>(pool: &'a ContentPool, name: &str) -> &'a PoolDefinition {
        pool.definitions
            .iter()
            .find(|d| d.registry_name == name)
            .unwrap()
    }

    #[test]
    fn descriptor_joins_catalog_ids_and_rejects_unknown_pool() {
        let pool = content_pool(crate::world::WORLD_POOL).unwrap();
        let content = default_content_pack();
        assert_eq!(pool.world_version, crate::WORLD_VERSION);
        assert_eq!(pool.content_digest, content.content_digest());
        assert_eq!(pool.definitions.len(), content.len());
        let avatar = row(pool, "Avatar Enthusiasts");
        assert_eq!(avatar.eligibility, Eligibility::Eligible);
        assert_eq!(
            Some(avatar.card_def_id),
            content.definition_id("Avatar Enthusiasts")
        );
        // Every defined card is in the world pool, whatever its set.
        assert_eq!(row(pool, "Pop Quiz").eligibility, Eligibility::Eligible);
        assert_eq!(row(pool, "Counterspell").eligibility, Eligibility::Eligible);
        let cat_gator = row(pool, "Cat-Gator");
        assert_eq!(cat_gator.eligibility, Eligibility::Eligible);
        assert!(cat_gator.reason.is_none());
        assert_eq!(row(pool, "Ally").eligibility, Eligibility::Unsupported);
        let fixture = row(pool, "Crossroads of Destiny");
        assert_eq!(fixture.eligibility, Eligibility::Unsupported);
        assert!(fixture.key.is_none());
        assert!(content_pool("missing-pool").is_err());
        assert!(content_pool("tla-baseline-v1").is_err());
    }
}
