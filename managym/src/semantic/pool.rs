//! Versioned content eligibility shared by deck import and native admission.
//!
//! Every pack key names compiled content with one descriptor row per
//! definition. A defined card is eligible unless its set file marks it
//! incomplete ([`compile::incomplete`]); tokens are never deck entries. The
//! world pool also lists the native set-file cards, which have no semantic
//! program, as unsupported. Setup construction policy belongs to the admission
//! caller, not this catalog.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};

use super::{compile, IrError, SemanticPack};
use crate::{
    cards::content_pack::{CompiledSemanticManifest, ContentPack},
    cardsets,
    state::card::CardDefId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Eligibility {
    Eligible,
    Unsupported,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolDefinition {
    #[serde(skip_deserializing)]
    pub card_def_id: Option<CardDefId>,
    pub semantic_key: Option<String>,
    pub registry_name: String,
    pub eligibility: Eligibility,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContentPool {
    pub pack_key: String,
    pub world_version: String,
    pub content_digest: String,
    pub compiled_semantics: CompiledSemanticManifest,
    pub definitions: Vec<PoolDefinition>,
}

struct PoolRuntime {
    content: Arc<ContentPack>,
    descriptor: ContentPool,
}

impl PoolRuntime {
    fn compile(
        semantic: &SemanticPack,
        incomplete: &BTreeMap<String, String>,
        native: &[String],
    ) -> Result<Self, IrError> {
        let content = Arc::new(semantic.compile_content_pack()?);
        let mut definitions = BTreeMap::new();
        for definition in &semantic.definitions {
            let reason = if definition.characteristics["token"] == true {
                Some("A token, created by its card; not a deck entry".to_owned())
            } else {
                incomplete.get(&definition.semantic_key).cloned()
            };
            definitions.insert(
                definition.registry_name.clone(),
                PoolDefinition {
                    card_def_id: content.definition_id(&definition.registry_name),
                    semantic_key: Some(definition.semantic_key.clone()),
                    registry_name: definition.registry_name.clone(),
                    eligibility: if reason.is_none() {
                        Eligibility::Eligible
                    } else {
                        Eligibility::Unsupported
                    },
                    reason,
                },
            );
        }
        for name in native {
            definitions.insert(
                name.clone(),
                PoolDefinition {
                    card_def_id: None,
                    semantic_key: None,
                    registry_name: name.clone(),
                    eligibility: Eligibility::Unsupported,
                    reason: Some("Native definition without a semantic program".into()),
                },
            );
        }
        let manifest = content.manifest();
        let descriptor = ContentPool {
            pack_key: semantic.pack_key.clone(),
            world_version: manifest.world_version,
            content_digest: manifest.content_digest,
            compiled_semantics: manifest.compiled_semantics.expect("compiled pool"),
            definitions: definitions.into_values().collect(),
        };
        Ok(Self {
            content,
            descriptor,
        })
    }
}

fn pools() -> Result<&'static BTreeMap<String, PoolRuntime>, IrError> {
    static POOLS: OnceLock<Result<BTreeMap<String, PoolRuntime>, IrError>> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            let incomplete = compile::incomplete()?;
            let native = cardsets::native_cards()
                .into_iter()
                .map(|definition| definition.name)
                .collect::<Vec<_>>();
            let world = SemanticPack::current("pool")?;
            let world_runtime = PoolRuntime::compile(&world, incomplete, &native)?;
            SemanticPack::catalog()?
                .iter()
                .map(|semantic| {
                    let runtime = PoolRuntime::compile(semantic, incomplete, &[])?;
                    Ok((semantic.pack_key.clone(), runtime))
                })
                .chain([Ok((world.pack_key.clone(), world_runtime))])
                .collect()
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Describe every card of a pack and whether it is supported, without admitting a deck.
pub fn content_pool(pack_key: &str) -> Result<ContentPool, IrError> {
    Ok(pool_runtime(pack_key)?.descriptor.clone())
}

/// The same compiled content whose IDs and digest the descriptor exposes.
/// Callers must check every submitted definition's eligibility before use.
pub fn pool_content(pack_key: &str) -> Result<Arc<ContentPack>, IrError> {
    Ok(Arc::clone(&pool_runtime(pack_key)?.content))
}

fn pool_runtime(pack_key: &str) -> Result<&'static PoolRuntime, IrError> {
    pools()?
        .get(pack_key)
        .ok_or_else(|| IrError::Malformed(format!("unknown content pool {pack_key:?}")))
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
        let pool = content_pool("tla-pool-v1").unwrap();
        let content = pool_content(&pool.pack_key).unwrap();
        assert_eq!(pool.world_version, crate::WORLD_VERSION);
        assert_eq!(pool.content_digest, content.content_digest());
        let avatar = row(&pool, "Avatar Enthusiasts");
        assert_eq!(avatar.eligibility, Eligibility::Eligible);
        assert_eq!(
            avatar.card_def_id,
            content.definition_id("Avatar Enthusiasts")
        );
        // Every defined card is in the world pool, whatever its set.
        assert_eq!(row(&pool, "Pop Quiz").eligibility, Eligibility::Eligible);
        let marked = row(&pool, "Cat-Gator");
        assert_eq!(marked.eligibility, Eligibility::Unsupported);
        assert!(marked.card_def_id.is_some() && marked.reason.is_some());
        assert_eq!(row(&pool, "Ally").eligibility, Eligibility::Unsupported);
        let native = row(&pool, "Counterspell");
        assert_eq!(native.eligibility, Eligibility::Unsupported);
        assert!(native.card_def_id.is_none());
        assert!(pool
            .definitions
            .iter()
            .all(|d| d.registry_name != "Crossroads of Destiny"));
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
