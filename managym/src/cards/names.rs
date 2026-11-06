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

/// Most options a naming prompt offers. Larger scopes publish groups first.
pub const NAME_BRANCHING: usize = 64;

/// A contiguous span of vocabulary rows, `start..end` in sorted name order.
/// Sorting makes every span a public, viewer-safe set: its first and last names
/// identify all members.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct NameScope {
    pub start: usize,
    pub end: usize,
}

/// One option of a naming prompt: a narrower group, or a vocabulary row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameOption {
    Group(NameScope),
    Name(usize),
}

impl NameScope {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Whether this scope's prompt offers names rather than groups.
    pub fn is_leaf(&self) -> bool {
        self.len() <= NAME_BRANCHING
    }

    /// Number of options published for this scope: names for a leaf, groups
    /// otherwise.
    pub fn option_count(&self) -> usize {
        self.option_count_with(NAME_BRANCHING)
    }

    /// Option `index` of this scope's prompt; `None` when out of range.
    pub fn option(&self, index: usize) -> Option<NameOption> {
        self.option_with(index, NAME_BRANCHING)
    }

    /// Children form a deterministic, balanced, complete partition: the depth
    /// needed is the smallest `d` with `len <= b^d`; the scope splits into
    /// `k = ceil(len / b^(d-1))` children whose sizes differ by at most one, so
    /// every child fits in `d - 1` further levels and every name is reachable
    /// through exactly one path. `b` is a parameter only so tests can exercise
    /// deep trees with real vocabularies; play always uses `NAME_BRANCHING`.
    pub(crate) fn option_count_with(&self, branching: usize) -> usize {
        let len = self.len();
        if len <= branching {
            return len;
        }
        let mut capacity = branching;
        while capacity * branching < len {
            capacity *= branching;
        }
        len.div_ceil(capacity)
    }

    pub(crate) fn option_with(&self, index: usize, branching: usize) -> Option<NameOption> {
        let count = self.option_count_with(branching);
        if index >= count {
            return None;
        }
        let len = self.len();
        Some(if len <= branching {
            NameOption::Name(self.start + index)
        } else {
            NameOption::Group(NameScope {
                start: self.start + index * len / count,
                end: self.start + (index + 1) * len / count,
            })
        })
    }
}

impl NameVocabulary {
    /// The whole vocabulary, the root naming prompt.
    pub fn root(&self) -> NameScope {
        NameScope {
            start: 0,
            end: self.len(),
        }
    }
}
