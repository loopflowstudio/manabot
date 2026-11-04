//! Immutable public naming vocabulary, independent of deck admission.
//!
//! Installed entries come from real catalog definitions. A validated vocabulary
//! can also contain names without executable definitions; a learning join never
//! manufactures a card. Decisions and forks share entries and serialize only
//! their digest, so witness generation never copies the whole vocabulary.

use crate::state::card::CardDefId;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen))]
pub struct NameCandidate {
    pub name: String,
    pub definition: Option<CardDefId>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct NameVocabulary {
    digest: String,
    #[serde(skip)]
    entries: Vec<NameCandidate>,
}

impl NameVocabulary {
    /// Bind an already validated source. This enforces representation invariants,
    /// not Oracle legality; provenance and the accepted universe belong to callers.
    pub fn new(mut entries: Vec<NameCandidate>) -> Arc<Self> {
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        assert!(!entries.is_empty(), "empty naming vocabulary");
        assert!(
            entries.iter().all(|entry| !entry.name.is_empty()),
            "empty card name"
        );
        assert!(
            entries.windows(2).all(|pair| pair[0].name != pair[1].name),
            "duplicate card name"
        );
        let digest = blake3::hash(&serde_json::to_vec(&entries).expect("names serialize"))
            .to_hex()
            .to_string();
        Arc::new(Self { digest, entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn get(&self, index: usize) -> Option<&NameCandidate> {
        self.entries.get(index)
    }
    pub fn iter(&self) -> impl Iterator<Item = &NameCandidate> {
        self.entries.iter()
    }
}
