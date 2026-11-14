//! Limited construction and free-order card selection before a game.
//!
//! `CardPool` holds only the cards available; `Limited` holds only the rules.
//! Both read counts over the card axis: index `i` is `CardDefId(i)`, so a row
//! has one entry per catalog definition. `Limited::available` is the one
//! stateless authority over a pool row and a deck row: the copies of each card
//! that may still be added and whether stopping is legal. `Limited::legal` and
//! the name-keyed `Limited::issues` share its pool rule, and `LimitedBuild` is
//! one deck being built from a pool, committing each selection through it.
//! World admission owns card eligibility; this module adds pool multiplicity
//! and the 40-card minimum. A build's size cap bounds an episode, not Limited
//! deck legality.

use std::{collections::BTreeMap, sync::OnceLock};

use crate::cards::{
    content_pack::default_content_pack,
    deck_admission::{admit_deck, DeckIssue, MAX_SETUP_CARDS},
    world_pool, ContentError, Eligibility,
};
use crate::state::deck::Deck;

pub const LIMITED_MIN_CARDS: usize = 40;
pub const DEFAULT_BUILD_CAP: usize = 60;

/// Cards available to a build: supported names with positive counts, held
/// both by name and as a count row over the card axis.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen, module = "managym"))]
pub struct CardPool {
    cards: BTreeMap<String, usize>,
    counts: Vec<u32>,
}

/// The Limited construction rules, independent of any pool.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen, module = "managym.format"))]
pub struct Limited;

/// What a deck may do next: per card, the copies that can still be added
/// (all zero at the cap), and whether the deck as it stands is legal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Available {
    pub copies: Vec<u32>,
    pub can_stop: bool,
}

/// One deck being built from one pool; selections commit.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "managym.format"))]
pub struct LimitedBuild {
    pool: CardPool,
    max_cards: usize,
    deck: Vec<u32>,
}

// ---------------------------------------------------------------------------
// The card axis
// ---------------------------------------------------------------------------

struct Axis {
    names: Vec<String>,
    /// Deck entries the world supports; tokens, back faces and fixtures are not.
    eligible: Vec<bool>,
    /// Eligible basic lands from printed characteristics, never a name allowlist.
    basics: Vec<bool>,
}

fn axis() -> &'static Axis {
    static AXIS: OnceLock<Axis> = OnceLock::new();
    AXIS.get_or_init(|| {
        let catalog = default_content_pack();
        let mut eligible = vec![false; catalog.len()];
        for row in &world_pool().definitions {
            eligible[row.card_def_id.0 as usize] = row.eligibility == Eligibility::Eligible;
        }
        let (names, basics) = catalog
            .definition_entries()
            .map(|(id, card)| {
                let basic = eligible[id.0 as usize]
                    && card.supertypes.iter().any(|s| s == "basic")
                    && card.types.is_land();
                (card.name.clone(), basic)
            })
            .unzip();
        Axis {
            names,
            eligible,
            basics,
        }
    })
}

/// Entries on the card axis: one per catalog definition.
pub fn card_count() -> usize {
    axis().names.len()
}

/// Which cards a pool does not limit, over the card axis.
pub fn basics() -> &'static [bool] {
    &axis().basics
}

/// Names of the basic lands, in name order.
pub fn basic_lands() -> Vec<String> {
    let axis = axis();
    let mut names = (0..axis.names.len())
        .filter(|&i| axis.basics[i])
        .map(|i| axis.names[i].clone())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn card_name(id: usize) -> &'static str {
    &axis().names[id]
}

fn card_id(name: &str) -> Option<usize> {
    default_content_pack()
        .definition_id(name)
        .map(|id| id.0 as usize)
}

fn check_len(row: &[u32]) -> Result<(), ContentError> {
    if row.len() == card_count() {
        Ok(())
    } else {
        Err(ContentError(format!(
            "a count row has {} entries; the card axis has {}",
            row.len(),
            card_count()
        )))
    }
}

/// A row may hold only cards the world supports for deck entry.
fn check_supported(row: &[u32]) -> Result<(), ContentError> {
    check_len(row)?;
    match (0..row.len()).find(|&i| row[i] > 0 && !axis().eligible[i]) {
        Some(i) => Err(ContentError(format!(
            "{}: not supported for deck entry",
            card_name(i)
        ))),
        None => Ok(()),
    }
}

fn total(row: &[u32]) -> usize {
    row.iter().map(|&n| n as usize).sum()
}

/// The pool rule: non-basics the deck holds beyond their pool counts, with
/// each card's pool count.
fn over_pool<'a>(pool: &'a [u32], deck: &'a [u32]) -> impl Iterator<Item = (usize, u32)> + 'a {
    let basics = basics();
    (0..deck.len())
        .filter(move |&i| !basics[i] && deck[i] > pool[i])
        .map(move |i| (i, pool[i]))
}

fn pool_count_issue(id: usize, available: u32) -> DeckIssue {
    let name = card_name(id);
    let message = if available == 0 {
        format!("{name}: not in the pool")
    } else {
        format!("{name}: deck exceeds the pool count of {available}")
    };
    DeckIssue {
        card_name: Some(name.to_owned()),
        code: "pool_count".into(),
        ..DeckIssue::format(message)
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

fn names(row: &[u32]) -> BTreeMap<String, usize> {
    (0..row.len())
        .filter(|&i| row[i] > 0)
        .map(|i| (card_name(i).to_owned(), row[i] as usize))
        .collect()
}

// ---------------------------------------------------------------------------
// Pools and rules
// ---------------------------------------------------------------------------

impl CardPool {
    /// Admit named counts; an empty pool is legitimate.
    pub fn new(cards: BTreeMap<String, usize>) -> Result<Self, ContentError> {
        if !cards.is_empty() {
            reject(&admit_deck(&Deck::new(cards.clone(), BTreeMap::new())))?;
        }
        let mut counts = vec![0u32; card_count()];
        for (name, &count) in &cards {
            let id = card_id(name).expect("admitted names are defined");
            counts[id] = u32::try_from(count)
                .map_err(|_| ContentError(format!("{name}: count is too large")))?;
        }
        Ok(Self { cards, counts })
    }

    /// A pool from a count row over the card axis.
    pub fn from_counts(counts: &[u32]) -> Result<Self, ContentError> {
        check_supported(counts)?;
        Self::new(names(counts))
    }

    pub fn cards(&self) -> &BTreeMap<String, usize> {
        &self.cards
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }
}

impl Limited {
    /// Construction issues of a finished main deck, independent of any cap.
    pub fn issues(&self, pool: &CardPool, deck: &BTreeMap<String, usize>) -> Vec<DeckIssue> {
        // Ordinary admission names unknown, unsupported and zero-count cards.
        let mut issues = if deck.is_empty() {
            Vec::new()
        } else {
            admit_deck(&Deck::new(deck.clone(), BTreeMap::new()))
        };
        let mut row = vec![0u32; card_count()];
        for (name, &count) in deck {
            if let Some(id) = card_id(name) {
                row[id] = u32::try_from(count).unwrap_or(u32::MAX);
            }
        }
        issues.extend(over_pool(pool.counts(), &row).map(|(i, n)| pool_count_issue(i, n)));
        if total(&row) < LIMITED_MIN_CARDS {
            issues.push(DeckIssue::format(
                "Limited requires at least 40 main-deck cards",
            ));
        }
        issues
    }

    /// Whether a deck row passes the Limited rules, with no cap. Fails only
    /// for a malformed row or a pool holding an unsupported card.
    pub fn legal(&self, pool: &[u32], deck: &[u32]) -> Result<bool, ContentError> {
        check_supported(pool)?;
        check_len(deck)?;
        let size = total(deck);
        Ok(check_supported(deck).is_ok()
            && over_pool(pool, deck).next().is_none()
            && (LIMITED_MIN_CARDS..=MAX_SETUP_CARDS).contains(&size))
    }

    /// The copies of each card that can still be added to `deck`, and whether
    /// it can stop. Non-basics may add their pool count minus their deck
    /// count, unclipped by the cap; basics may fill the slots left. A deck no
    /// build could reach (beyond a pool count, unsupported, over the cap)
    /// fails rather than truncates.
    pub fn available(
        &self,
        pool: &[u32],
        deck: &[u32],
        max_cards: usize,
    ) -> Result<Available, ContentError> {
        check_cap(max_cards)?;
        check_supported(pool)?;
        check_supported(deck)?;
        if let Some((id, n)) = over_pool(pool, deck).next() {
            return Err(ContentError(pool_count_issue(id, n).message));
        }
        let size = total(deck);
        if size > max_cards {
            return Err(ContentError(format!(
                "deck of {size} cards exceeds the build's size cap of {max_cards}"
            )));
        }
        let slots = (max_cards - size) as u32;
        let basics = basics();
        let copies = (0..deck.len())
            .map(|i| {
                if slots == 0 {
                    0
                } else if basics[i] {
                    slots
                } else {
                    pool[i] - deck[i]
                }
            })
            .collect();
        Ok(Available {
            copies,
            // Support, pool counts and the cap passed above; only the minimum remains.
            can_stop: size >= LIMITED_MIN_CARDS,
        })
    }

    pub fn build(&self, pool: CardPool, max_cards: usize) -> Result<LimitedBuild, ContentError> {
        check_cap(max_cards)?;
        Ok(LimitedBuild {
            pool,
            max_cards,
            deck: vec![0; card_count()],
        })
    }
}

fn check_cap(max_cards: usize) -> Result<(), ContentError> {
    if (LIMITED_MIN_CARDS..=MAX_SETUP_CARDS).contains(&max_cards) {
        Ok(())
    } else {
        Err(ContentError(format!(
            "build size cap must be in {LIMITED_MIN_CARDS}..={MAX_SETUP_CARDS}"
        )))
    }
}

impl LimitedBuild {
    pub fn available(&self) -> Available {
        Limited
            .available(self.pool.counts(), &self.deck, self.max_cards)
            .expect("a build only commits available cards")
    }

    /// The cards that can still be selected, with their remaining copies.
    pub fn available_cards(&self) -> BTreeMap<String, usize> {
        names(&self.available().copies)
    }

    /// Commit one copy of an available card; otherwise leave the deck unchanged.
    pub fn select(&mut self, card: &str) -> Result<(), ContentError> {
        match card_id(card) {
            Some(id) if self.available().copies[id] > 0 => {
                self.deck[id] += 1;
                Ok(())
            }
            _ => Err(ContentError(format!("{card}: not available"))),
        }
    }

    pub fn deck(&self) -> BTreeMap<String, usize> {
        names(&self.deck)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(cards: &[(&str, usize)]) -> CardPool {
        CardPool::new(cards.iter().map(|&(n, c)| (n.into(), c)).collect()).unwrap()
    }

    fn row(cards: &[(&str, u32)]) -> Vec<u32> {
        let mut row = vec![0; card_count()];
        for &(name, count) in cards {
            row[card_id(name).unwrap()] = count;
        }
        row
    }

    fn id(name: &str) -> usize {
        card_id(name).unwrap()
    }

    #[test]
    fn builds_use_pool_counts_and_stop_only_at_legal_decks() {
        let elves = pool(&[("Llanowar Elves", 6)]);
        let mut build = Limited.build(elves.clone(), 60).unwrap();
        assert_eq!(build.available_cards()["Llanowar Elves"], 6);
        for _ in 0..6 {
            build.select("Llanowar Elves").unwrap();
        }
        assert!(build.select("Llanowar Elves").is_err());
        assert!(!build.available_cards().contains_key("Llanowar Elves"));
        assert_eq!(build.deck()["Llanowar Elves"], 6);
        for _ in 0..33 {
            build.select("Forest").unwrap();
        }
        assert!(!build.available().can_stop);
        build.select("Forest").unwrap();
        let next = build.available();
        assert!(next.can_stop && next.copies[id("Forest")] == 20);
        assert!(Limited.issues(&elves, &build.deck()).is_empty());
        for _ in 0..20 {
            build.select("Forest").unwrap();
        }
        let capped = build.available();
        assert!(capped.can_stop && capped.copies.iter().all(|&n| n == 0));
        assert!(build.select("Forest").is_err());
        assert_eq!(build.deck()["Forest"], 54);
        // The cap bounds a build, not legality.
        let large = BTreeMap::from([("Forest".into(), 61)]);
        assert!(Limited.issues(&elves, &large).is_empty());
        assert!(Limited
            .legal(elves.counts(), &row(&[("Forest", 61)]))
            .unwrap());
        assert!(Limited
            .available(elves.counts(), &row(&[("Forest", 61)]), 60)
            .is_err());
        let over = BTreeMap::from([("Llanowar Elves".into(), 7), ("Forest".into(), 40)]);
        assert!(!Limited.issues(&elves, &over).is_empty());
        let over = row(&[("Llanowar Elves", 7), ("Forest", 40)]);
        assert!(!Limited.legal(elves.counts(), &over).unwrap());
        assert!(Limited.available(elves.counts(), &over, 60).is_err());
    }

    #[test]
    fn rows_and_names_agree_on_every_step_of_a_build() {
        let pool = pool(&[("Llanowar Elves", 3), ("Island", 2)]);
        let mut build = Limited.build(pool.clone(), 45).unwrap();
        let picks = ["Llanowar Elves", "Island", "Island", "Llanowar Elves"]
            .into_iter()
            .chain(std::iter::repeat_n("Forest", 41));
        for pick in picks {
            let available = build.available();
            let by_name = build.available_cards();
            assert_eq!(names(&available.copies), by_name);
            assert_eq!(
                available.can_stop,
                Limited.legal(pool.counts(), &build.deck).unwrap()
            );
            assert_eq!(
                available.can_stop,
                Limited.issues(&pool, &build.deck()).is_empty()
            );
            build.select(pick).unwrap();
        }
        // A basic in the pool is still unlimited; the cap stops everything.
        assert_eq!(build.deck()["Island"], 2);
        assert!(build.available().copies.iter().all(|&n| n == 0));
        assert_eq!(
            CardPool::from_counts(pool.counts()).unwrap(),
            pool,
            "counts round-trip"
        );
    }

    #[test]
    fn rejects_bad_pools_caps_and_decks_and_allows_an_empty_pool() {
        for name in ["Not a Card", "Ally"] {
            assert!(CardPool::new(BTreeMap::from([(name.into(), 1)])).is_err());
        }
        assert!(CardPool::new(BTreeMap::from([("Forest".into(), 0)])).is_err());
        let unsupported = (0..card_count()).find(|&i| !axis().eligible[i]).unwrap();
        let mut token = vec![0; card_count()];
        token[unsupported] = 1;
        assert!(CardPool::from_counts(&token).is_err());
        assert!(CardPool::from_counts(&[1]).is_err());
        let empty = pool(&[]);
        assert!(Limited.build(empty.clone(), 39).is_err());
        assert!(Limited.available(empty.counts(), &row(&[]), 39).is_err());
        let start = Limited.available(empty.counts(), &row(&[]), 60).unwrap();
        assert!(!start.can_stop);
        assert_eq!(start.copies[id("Llanowar Elves")], 0);
        assert!(basics()[id("Forest")] && !basics()[id("Llanowar Elves")]);
        assert!(basic_lands().contains(&"Forest".to_owned()));
        for deck in [row(&[("Llanowar Elves", 1)]), token.clone(), vec![0; 3]] {
            assert!(Limited.available(empty.counts(), &deck, 60).is_err());
        }
        assert!(!Limited.legal(empty.counts(), &token).unwrap());
        assert!(Limited.legal(&token, &row(&[("Forest", 40)])).is_err());
        let forty = row(&[("Forest", 40)]);
        assert!(
            Limited
                .available(empty.counts(), &forty, 60)
                .unwrap()
                .can_stop
        );
    }
}
