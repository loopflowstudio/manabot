//! Versioned content eligibility shared by deck import and native admission.
//!
//! A compiled definition is not automatically eligible. Offline qualifications
//! bind independent witnesses to the complete authored definition; generated
//! descriptors bind those decisions to the exact IR hash. Setup construction
//! policy belongs to the admission caller, not this catalog.

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};

use super::{IrError, SemanticPack};
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

#[derive(Deserialize)]
struct QualificationManifest {
    ir_hash: String,
    definitions: Vec<PoolDefinition>,
}

struct PoolRuntime {
    content: Arc<ContentPack>,
    descriptor: ContentPool,
}

fn pools() -> Result<&'static BTreeMap<String, PoolRuntime>, IrError> {
    static POOLS: OnceLock<Result<BTreeMap<String, PoolRuntime>, IrError>> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            let mut qualifications: BTreeMap<String, QualificationManifest> = serde_json::from_str(
                include_str!("../../../content/semantic/current/generated/eligibility.json"),
            )
            .map_err(|e| IrError::Malformed(e.to_string()))?;
            let mut pools = BTreeMap::new();
            for semantic in SemanticPack::catalog()?
                .iter()
                .cloned()
                .chain([SemanticPack::pool()?])
            {
                let qualification = qualifications.remove(&semantic.pack_key).ok_or_else(|| {
                    IrError::Malformed(format!("missing eligibility for {}", semantic.pack_key))
                })?;
                if qualification.ir_hash != semantic.ir_hash {
                    return Err(IrError::Malformed(format!(
                        "stale eligibility for {}",
                        semantic.pack_key
                    )));
                }
                let content = Arc::new(semantic.compile_content_pack()?);
                let mut definitions = qualification.definitions;
                for row in &mut definitions {
                    row.card_def_id = content.definition_id(&row.registry_name);
                    let actual = semantic
                        .definitions
                        .iter()
                        .find(|d| d.registry_name == row.registry_name);
                    if actual.map(|d| &d.semantic_key) != row.semantic_key.as_ref()
                        || (row.eligibility == Eligibility::Eligible && row.card_def_id.is_none())
                    {
                        return Err(IrError::Malformed(format!(
                            "incomplete pool definition {}",
                            row.registry_name
                        )));
                    }
                }
                let manifest = content.manifest();
                let descriptor = ContentPool {
                    pack_key: semantic.pack_key.clone(),
                    world_version: manifest.world_version,
                    content_digest: manifest.content_digest,
                    compiled_semantics: manifest.compiled_semantics.expect("compiled pool"),
                    definitions,
                };
                pools.insert(
                    semantic.pack_key,
                    PoolRuntime {
                        content,
                        descriptor,
                    },
                );
            }
            Ok(pools)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Describe supported and known unsupported cards without admitting a deck.
pub fn content_pool(pack_key: &str) -> Result<ContentPool, IrError> {
    pools()?
        .get(pack_key)
        .map(|p| p.descriptor.clone())
        .ok_or_else(|| IrError::Malformed(format!("unknown content pool {pack_key:?}")))
}

/// The same compiled content whose IDs and digest the descriptor exposes.
/// Callers must check every submitted definition's eligibility before use.
pub fn pool_content(pack_key: &str) -> Result<Arc<ContentPack>, IrError> {
    pools()?
        .get(pack_key)
        .map(|p| Arc::clone(&p.content))
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
    }
}
