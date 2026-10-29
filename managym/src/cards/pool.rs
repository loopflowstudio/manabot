//! Content eligibility shared by deck import and native admission.
//!
//! Every pack key names one [`ContentPack`] with one descriptor row per
//! definition. The world pool ([`crate::world::WORLD_POOL`]) describes the
//! default pack, so it shares that pack's `CardDefId`s and digest; authored
//! setups hold their decks' cards. A defined card is eligible unless its set
//! file marks it incomplete ([`cardsets::incomplete`]); tokens and rules
//! fixtures are never deck entries. Setup construction policy belongs to the
//! admission caller, not this catalog.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use super::{setups::authored_packs, ContentError};
use crate::{
    cards::content_pack::{default_content_pack, ContentPack},
    cardsets,
    state::card::CardDefId,
};

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
    pub pack_key: String,
    pub world_version: String,
    pub content_digest: String,
    pub definitions: Vec<PoolDefinition>,
}

struct PoolRuntime {
    content: Arc<ContentPack>,
    descriptor: ContentPool,
}

impl PoolRuntime {
    fn describe(pack_key: &str, content: Arc<ContentPack>) -> Self {
        let incomplete = cardsets::incomplete()
            .iter()
            .copied()
            .collect::<BTreeMap<_, _>>();
        let mut definitions = content
            .definition_entries()
            .map(|(card_def_id, definition)| {
                let reason = if definition.is_token {
                    Some("A token, created by its card; not a deck entry".to_owned())
                } else if definition.key.is_empty() {
                    Some("A rules fixture, not a printed card".to_owned())
                } else {
                    incomplete
                        .get(definition.key)
                        .map(|reason| reason.to_string())
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
        let manifest = content.manifest();
        Self {
            descriptor: ContentPool {
                pack_key: pack_key.to_owned(),
                world_version: manifest.world_version,
                content_digest: manifest.content_digest,
                definitions,
            },
            content,
        }
    }
}

fn pools() -> Result<&'static BTreeMap<String, PoolRuntime>, ContentError> {
    static POOLS: OnceLock<Result<BTreeMap<String, PoolRuntime>, ContentError>> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            let world = crate::world::WORLD_POOL;
            let mut pools = BTreeMap::new();
            pools.insert(
                world.to_owned(),
                PoolRuntime::describe(world, default_content_pack()),
            );
            for pack in authored_packs()? {
                pools.insert(
                    pack.pack_key.clone(),
                    PoolRuntime::describe(&pack.pack_key, Arc::clone(&pack.content)),
                );
            }
            Ok(pools)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Describe every card of a pack and whether it is supported, without admitting a deck.
pub fn content_pool(pack_key: &str) -> Result<ContentPool, ContentError> {
    Ok(pool_runtime(pack_key)?.descriptor.clone())
}

/// The same content whose IDs and digest the descriptor exposes. Callers must
/// check every submitted definition's eligibility before use.
pub fn pool_content(pack_key: &str) -> Result<Arc<ContentPack>, ContentError> {
    Ok(Arc::clone(&pool_runtime(pack_key)?.content))
}

fn pool_runtime(pack_key: &str) -> Result<&'static PoolRuntime, ContentError> {
    pools()?
        .get(pack_key)
        .ok_or_else(|| ContentError(format!("unknown content pool {pack_key:?}")))
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
    fn descriptor_joins_runtime_ids_and_rejects_unknown_pool() {
        let pool = content_pool(crate::world::WORLD_POOL).unwrap();
        let content = pool_content(&pool.pack_key).unwrap();
        assert_eq!(pool.world_version, crate::WORLD_VERSION);
        assert_eq!(pool.content_digest, content.content_digest());
        let avatar = row(&pool, "Avatar Enthusiasts");
        assert_eq!(avatar.eligibility, Eligibility::Eligible);
        assert_eq!(
            Some(avatar.card_def_id),
            content.definition_id("Avatar Enthusiasts")
        );
        // Every defined card is in the world pool, whatever its set.
        assert_eq!(row(&pool, "Pop Quiz").eligibility, Eligibility::Eligible);
        assert_eq!(
            row(&pool, "Counterspell").eligibility,
            Eligibility::Eligible
        );
        let marked = row(&pool, "Cat-Gator");
        assert_eq!(marked.eligibility, Eligibility::Unsupported);
        assert!(marked.reason.is_some());
        assert_eq!(row(&pool, "Ally").eligibility, Eligibility::Unsupported);
        let fixture = row(&pool, "Crossroads of Destiny");
        assert_eq!(fixture.eligibility, Eligibility::Unsupported);
        assert!(fixture.key.is_none());
        // The world pool is the default pack: same ids, same digest.
        let default = default_content_pack();
        assert!(Arc::ptr_eq(&content, &default));
        assert_eq!(
            Some(avatar.card_def_id),
            default.definition_id("Avatar Enthusiasts")
        );
        assert!(content_pool("missing-pool").is_err());

        let baseline = content_pool("tla-baseline-v1").unwrap();
        assert!(baseline
            .definitions
            .iter()
            .all(|d| d.registry_name != "Avatar Enthusiasts"));
        assert_eq!(
            row(&baseline, "Treetop Freedom Fighters").eligibility,
            Eligibility::Eligible
        );
    }
}
