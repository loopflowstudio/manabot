//! Exact, viewer-scoped facts about library positions, counted from the top.
//!
//! A private permutation or an unspecified departure forgets positions. It
//! never replaces them with claims about a set of possible places. Hand
//! minima remain on Player; both kinds of fact name definitions, not objects.

use std::collections::BTreeMap;

use super::card::CardDefId;

pub const KNOWN_LIBRARY_CAPACITY: usize = 10;
/// The effective owner view can contain all ten public and ten private facts.
pub const KNOWN_LIBRARY_ROWS: usize = 2 * KNOWN_LIBRARY_CAPACITY;
const TOP_CAPACITY: usize = 8;

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct LibraryKnowledge {
    pub public: BTreeMap<usize, CardDefId>,
    pub own: BTreeMap<usize, CardDefId>,
}

impl LibraryKnowledge {
    pub fn is_empty(&self) -> bool {
        self.public.is_empty() && self.own.is_empty()
    }

    pub fn forget_order(&mut self) {
        self.public.clear();
        self.own.clear();
    }

    /// A top departure has a publicly identified position. Return the
    /// definition the opponent already knew there, if any.
    pub fn remove_top(&mut self) -> Option<CardDefId> {
        self.remove_position(0)
    }

    /// Public departure at an identified position shifts only the facts below it.
    pub fn remove_position(&mut self, removed: usize) -> Option<CardDefId> {
        let known = self.public.get(&removed).copied();
        for view in [&mut self.public, &mut self.own] {
            *view = std::mem::take(view)
                .into_iter()
                .filter_map(|(position, definition)| {
                    if position == removed {
                        None
                    } else {
                        Some((position - usize::from(position > removed), definition))
                    }
                })
                .collect();
        }
        known
    }

    /// Insertion at the top shifts all older facts. The caller supplies only
    /// definitions seen by the respective viewer, never the hidden entrant.
    pub fn insert_top(&mut self, public: Option<CardDefId>, own: Option<CardDefId>) {
        for (view, seen) in [(&mut self.public, public), (&mut self.own, own)] {
            if !view.is_empty() {
                *view = std::mem::take(view)
                    .into_iter()
                    .map(|(position, definition)| (position + 1, definition))
                    .collect();
            }
            if let Some(definition) = seen {
                view.insert(0, definition);
            }
            Self::bound(view);
        }
    }

    /// Keep eight nearest-top and two nearest-bottom facts. Dropping a fact
    /// is the only overflow behavior and cannot disclose private information.
    pub fn bound(view: &mut BTreeMap<usize, CardDefId>) {
        if view.len() <= KNOWN_LIBRARY_CAPACITY {
            return;
        }
        let top_end = *view.keys().nth(TOP_CAPACITY - 1).unwrap();
        let bottom_start = *view
            .keys()
            .rev()
            .nth(KNOWN_LIBRARY_CAPACITY - TOP_CAPACITY - 1)
            .unwrap();
        view.retain(|position, _| *position <= top_end || *position >= bottom_start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_knowledge_position_updates_never_expose_an_unknown_entrant() {
        let mut knowledge = LibraryKnowledge::default();
        knowledge.public.insert(0, CardDefId(1));
        knowledge.own = knowledge.public.clone();
        knowledge.insert_top(None, Some(CardDefId(2)));
        assert_eq!(knowledge.public, BTreeMap::from([(1, CardDefId(1))]));
        assert_eq!(knowledge.remove_top(), None);
        assert_eq!(knowledge.public, BTreeMap::from([(0, CardDefId(1))]));
        assert_eq!(knowledge.remove_top(), Some(CardDefId(1)));
        assert!(knowledge.is_empty());
    }

    #[test]
    fn hidden_knowledge_cap_keeps_both_ends_without_replacing_facts() {
        let mut view: BTreeMap<_, _> = (0..20).map(|i| (i, CardDefId(i as u32))).collect();
        LibraryKnowledge::bound(&mut view);
        assert_eq!(
            view.keys().copied().collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 18, 19]
        );
    }
}
