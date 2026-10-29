//! Versioned content eligibility shared by deck import and native admission.
//!
//! A compiled definition is not automatically eligible. Reviewed qualifications
//! bind independent witnesses to the complete authored closure (see
//! [`compile::qualified`]). Setup construction policy belongs to the admission
//! caller, not this catalog.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};

use super::{compile, IrError, SemanticPack};
use crate::{
    cardsets::alpha::{CompiledSemanticManifest, ContentPack},
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
        qualified: &BTreeSet<String>,
        inventory: &BTreeSet<String>,
    ) -> Result<Self, IrError> {
        let content = Arc::new(semantic.compile_content_pack()?);
        // IR owns identity and pack membership; qualifications only decide
        // eligibility. Inventory-only cards remain explicit.
        let mut definitions = BTreeMap::new();
        for definition in &semantic.definitions {
            let eligible = qualified.contains(&definition.semantic_key);
            definitions.insert(
                definition.registry_name.clone(),
                PoolDefinition {
                    card_def_id: content.definition_id(&definition.registry_name),
                    semantic_key: Some(definition.semantic_key.clone()),
                    registry_name: definition.registry_name.clone(),
                    eligibility: if eligible {
                        Eligibility::Eligible
                    } else {
                        Eligibility::Unsupported
                    },
                    reason: if eligible {
                        None
                    } else {
                        Some("Preserved authored behavior; complete independent qualification pending".into())
                    },
                },
            );
        }
        for name in inventory {
            definitions
                .entry(name.clone())
                .or_insert_with(|| PoolDefinition {
                    card_def_id: None,
                    semantic_key: None,
                    registry_name: name.clone(),
                    eligibility: Eligibility::Unsupported,
                    reason: Some("Not implemented in this content pack".into()),
                });
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
            let (qualified, inventory) = (compile::qualified()?, compile::inventory()?);
            let pool = SemanticPack::current("pool")?;
            SemanticPack::catalog()?
                .iter()
                .chain([&pool])
                .map(|semantic| {
                    let runtime = PoolRuntime::compile(semantic, qualified, inventory)?;
                    Ok((semantic.pack_key.clone(), runtime))
                })
                .collect()
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Describe supported and known unsupported cards without admitting a deck.
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

    #[test]
    fn descriptor_joins_runtime_ids_and_rejects_unknown_pool() {
        let pool = content_pool("tla-pool-v1").unwrap();
        let content = pool_content(&pool.pack_key).unwrap();
        assert_eq!(pool.world_version, crate::WORLD_VERSION);
        assert_eq!(pool.content_digest, content.content_digest());
        let avatar = pool
            .definitions
            .iter()
            .find(|d| d.registry_name == "Avatar Enthusiasts")
            .unwrap();
        assert_eq!(avatar.eligibility, Eligibility::Eligible);
        assert_eq!(
            avatar.card_def_id,
            content.definition_id("Avatar Enthusiasts")
        );
        let missing = pool
            .definitions
            .iter()
            .find(|d| d.registry_name == "Airbending Lesson")
            .unwrap();
        assert_eq!(missing.eligibility, Eligibility::Unsupported);
        assert!(missing.card_def_id.is_none());
        assert!(missing.reason.is_some());
        assert!(content_pool("missing-pool").is_err());

        let baseline = content_pool("tla-baseline-v1").unwrap();
        let excluded = baseline
            .definitions
            .iter()
            .find(|d| d.registry_name == "Avatar Enthusiasts")
            .unwrap();
        assert_eq!(excluded.eligibility, Eligibility::Unsupported);
        assert!(excluded.card_def_id.is_none());
        assert!(excluded.semantic_key.is_none());
    }
}
