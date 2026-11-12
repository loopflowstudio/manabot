//! Limited construction and free-order card selection before a game.
//!
//! `CardPool` holds only the cards available; `Limited` holds only the rules.
//! `Limited::selection` is the one stateless authority over a pool and deck
//! counts: the cards that may be selected next and whether stopping is legal.
//! `LimitedBuild` is one deck being built from a pool, committing each
//! selection. World admission owns card eligibility; this module adds pool
//! multiplicity and the 40-card minimum. A build's size cap bounds an episode,
//! not Limited deck legality.

use std::collections::{BTreeMap, BTreeSet};

use crate::cards::{
    content_pack::default_content_pack,
    deck_admission::{admit_deck, DeckIssue, MAX_SETUP_CARDS},
    world_pool, ContentError, Eligibility,
};
use crate::state::deck::Deck;

pub const LIMITED_MIN_CARDS: usize = 40;
pub const DEFAULT_BUILD_CAP: usize = 60;

/// Cards available to a build: supported names with positive counts.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen, module = "managym"))]
pub struct CardPool {
    cards: BTreeMap<String, usize>,
}

/// The Limited construction rules, independent of any pool.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, module = "managym.format")
)]
pub struct Limited;

/// What a deck may do next: the selectable cards in name order, and whether
/// the deck as it stands passes the Limited rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub cards: Vec<String>,
    pub can_stop: bool,
}

/// One deck being built from one pool; selections commit.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "managym.format"))]
pub struct LimitedBuild {
    pool: CardPool,
    max_cards: usize,
    deck: BTreeMap<String, usize>,
}

/// Eligible basic lands from printed characteristics, never a name allowlist.
pub fn basic_lands() -> BTreeSet<String> {
    let catalog = default_content_pack();
    world_pool()
        .definitions
        .iter()
        .filter(|row| {
            row.eligibility == Eligibility::Eligible
                && catalog.definition(row.card_def_id).is_some_and(|card| {
                    card.supertypes.iter().any(|s| s == "basic") && card.types.is_land()
                })
        })
        .map(|row| row.registry_name.clone())
        .collect()
}

fn admitted_counts(cards: &BTreeMap<String, usize>) -> Vec<DeckIssue> {
    // An empty pool or deck so far is legitimate; ordinary game admission
    // deliberately continues to reject an empty main deck.
    if cards.is_empty() {
        Vec::new()
    } else {
        admit_deck(&Deck::new(cards.clone(), BTreeMap::new()))
    }
}

fn reject(issues: &[DeckIssue]) -> Result<(), ContentError> {
    if issues.is_empty() {
        Ok(())
    } else {
        Err(ContentError(
            issues
                .iter()
                .map(|i| i.message.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        ))
    }
}

fn size(deck: &BTreeMap<String, usize>) -> usize {
    deck.values().copied().fold(0usize, usize::saturating_add)
}

impl CardPool {
    pub fn new(cards: BTreeMap<String, usize>) -> Result<Self, ContentError> {
        reject(&admitted_counts(&cards))?;
        Ok(Self { cards })
    }

    pub fn cards(&self) -> &BTreeMap<String, usize> {
        &self.cards
    }
}

impl Limited {
    /// Admission and pool counts, without the minimum size.
    fn partial_issues(
        pool: &CardPool,
        deck: &BTreeMap<String, usize>,
        basics: &BTreeSet<String>,
    ) -> Vec<DeckIssue> {
        let mut issues = admitted_counts(deck);
        for (name, count) in deck {
            let available = pool.cards.get(name).copied().unwrap_or(0);
            if !basics.contains(name) && *count > available {
                let message = if available == 0 {
                    format!("{name}: not in the pool")
                } else {
                    format!("{name}: deck exceeds the pool count of {available}")
                };
                issues.push(DeckIssue {
                    card_name: Some(name.clone()),
                    code: "pool_count".into(),
                    ..DeckIssue::format(message)
                });
            }
        }
        issues
    }

    /// Construction issues of a finished main deck, independent of any cap.
    pub fn issues(&self, pool: &CardPool, deck: &BTreeMap<String, usize>) -> Vec<DeckIssue> {
        let mut issues = Self::partial_issues(pool, deck, &basic_lands());
        if size(deck) < LIMITED_MIN_CARDS {
            issues.push(DeckIssue::format(
                "Limited requires at least 40 main-deck cards",
            ));
        }
        issues
    }

    /// Every next card in name order; an invalid deck so far fails rather
    /// than truncates. At `max_cards` nothing is selectable.
    pub fn selection(
        &self,
        pool: &CardPool,
        deck: &BTreeMap<String, usize>,
        max_cards: usize,
    ) -> Result<Selection, ContentError> {
        let mut cards = basic_lands();
        reject(&Self::partial_issues(pool, deck, &cards))?;
        let size = size(deck);
        if size > max_cards {
            return Err(ContentError("deck exceeds the build's size cap".into()));
        }
        cards.extend(
            pool.cards
                .iter()
                .filter(|&(name, count)| deck.get(name).copied().unwrap_or(0) < *count)
                .map(|(name, _)| name.clone()),
        );
        Ok(Selection {
            cards: if size == max_cards {
                Vec::new()
            } else {
                cards.into_iter().collect()
            },
            // Admission and pool counts passed above; only the minimum remains.
            can_stop: size >= LIMITED_MIN_CARDS,
        })
    }

    pub fn build(&self, pool: CardPool, max_cards: usize) -> Result<LimitedBuild, ContentError> {
        if !(LIMITED_MIN_CARDS..=MAX_SETUP_CARDS).contains(&max_cards) {
            return Err(ContentError(format!(
                "build size cap must be in {LIMITED_MIN_CARDS}..={MAX_SETUP_CARDS}"
            )));
        }
        Ok(LimitedBuild {
            pool,
            max_cards,
            deck: BTreeMap::new(),
        })
    }
}

impl LimitedBuild {
    pub fn selection(&self) -> Selection {
        Limited
            .selection(&self.pool, &self.deck, self.max_cards)
            .expect("a build only commits selectable cards")
    }

    /// Commit one copy of a selectable card; otherwise leave the deck unchanged.
    pub fn select(&mut self, card: &str) -> Result<(), ContentError> {
        if !self.selection().cards.iter().any(|name| name == card) {
            return Err(ContentError(format!("{card}: not selectable")));
        }
        *self.deck.entry(card.to_owned()).or_insert(0) += 1;
        Ok(())
    }

    pub fn deck(&self) -> &BTreeMap<String, usize> {
        &self.deck
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(cards: &[(&str, usize)]) -> CardPool {
        CardPool::new(cards.iter().map(|&(n, c)| (n.into(), c)).collect()).unwrap()
    }

    #[test]
    fn builds_use_pool_counts_and_stop_only_at_legal_decks() {
        let elves = pool(&[("Llanowar Elves", 6)]);
        let mut build = Limited.build(elves.clone(), 60).unwrap();
        for _ in 0..6 {
            build.select("Llanowar Elves").unwrap();
        }
        assert!(build.select("Llanowar Elves").is_err());
        assert_eq!(build.deck()["Llanowar Elves"], 6);
        for _ in 0..33 {
            build.select("Forest").unwrap();
        }
        assert!(!build.selection().can_stop);
        build.select("Forest").unwrap();
        let next = build.selection();
        assert!(next.can_stop && next.cards.contains(&"Forest".into()));
        assert!(Limited.issues(&elves, build.deck()).is_empty());
        for _ in 0..20 {
            build.select("Forest").unwrap();
        }
        assert_eq!(
            build.selection(),
            Selection {
                cards: vec![],
                can_stop: true
            }
        );
        assert!(build.select("Forest").is_err());
        assert_eq!(build.deck()["Forest"], 54);
        // The cap bounds a build, not legality.
        let large = BTreeMap::from([("Forest".into(), 61)]);
        assert!(Limited.issues(&elves, &large).is_empty());
        assert!(Limited.selection(&elves, &large, 60).is_err());
        let over = BTreeMap::from([("Llanowar Elves".into(), 7), ("Forest".into(), 40)]);
        assert!(!Limited.issues(&elves, &over).is_empty());
        assert!(Limited.selection(&elves, &over, 60).is_err());
    }

    #[test]
    fn rejects_bad_pools_caps_and_decks_and_allows_an_empty_pool() {
        for name in ["Not a Card", "Ally"] {
            assert!(CardPool::new(BTreeMap::from([(name.into(), 1)])).is_err());
        }
        assert!(CardPool::new(BTreeMap::from([("Forest".into(), 0)])).is_err());
        let empty = pool(&[]);
        assert!(Limited.build(empty.clone(), 39).is_err());
        let selection = Limited.selection(&empty, &BTreeMap::new(), 60).unwrap();
        assert!(!selection.can_stop);
        assert!(!selection.cards.contains(&"Llanowar Elves".into()));
        for deck in [
            BTreeMap::from([("Llanowar Elves".into(), 1)]),
            BTreeMap::from([("Forest".into(), 0)]),
        ] {
            assert!(Limited.selection(&empty, &deck, 60).is_err());
        }
        let forty = BTreeMap::from([("Forest".into(), 40)]);
        assert!(Limited.selection(&empty, &forty, 60).unwrap().can_stop);
    }
}
