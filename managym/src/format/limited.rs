//! Limited construction and free-order picks before a game. World admission
//! owns card eligibility; this module adds pool multiplicity and the 40-card
//! minimum. The picking cap bounds an episode, not Limited deck legality.

use std::collections::{BTreeMap, BTreeSet};

use crate::cards::{
    content_pack::default_content_pack,
    deck_admission::{admit_deck, DeckIssue, MAX_SETUP_CARDS},
    world_pool, ContentError, Eligibility,
};
use crate::state::deck::Deck;

pub const LIMITED_MIN_CARDS: usize = 40;
pub const DEFAULT_PICK_CAP: usize = 60;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen, module = "manabot.managym"))]
pub struct LimitedRules {
    pool: BTreeMap<String, usize>,
    max_cards: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, get_all, module = "manabot.managym")
)]
pub struct LimitedChoices {
    pub cards: Vec<String>,
    pub can_stop: bool,
}

/// Eligible basic lands from printed characteristics, never a name allowlist.
pub fn basic_lands() -> BTreeSet<String> {
    let catalog = default_content_pack();
    world_pool()
        .definitions
        .iter()
        .filter_map(|row| {
            let card = catalog.definition(row.card_def_id)?;
            (row.eligibility == Eligibility::Eligible
                && card.supertypes.iter().any(|s| s == "basic")
                && card.types.is_land())
            .then(|| row.registry_name.clone())
        })
        .collect()
}

fn admitted_counts(cards: &BTreeMap<String, usize>) -> Vec<DeckIssue> {
    // An empty partial build/pool is legitimate; ordinary game admission
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

impl LimitedRules {
    pub fn new(pool: BTreeMap<String, usize>, max_cards: usize) -> Result<Self, ContentError> {
        if !(LIMITED_MIN_CARDS..=MAX_SETUP_CARDS).contains(&max_cards) {
            return Err(ContentError(format!(
                "pick cap must be in {LIMITED_MIN_CARDS}..={MAX_SETUP_CARDS}"
            )));
        }
        reject(&admitted_counts(&pool))?;
        Ok(Self { pool, max_cards })
    }

    fn partial_issues(
        &self,
        picks: &BTreeMap<String, usize>,
        basics: &BTreeSet<String>,
    ) -> Vec<DeckIssue> {
        let mut issues = admitted_counts(picks);
        for (name, count) in picks {
            let available = self.pool.get(name).copied().unwrap_or(0);
            if !basics.contains(name) && *count > available {
                let message = if available == 0 {
                    format!("{name}: not in the pool")
                } else {
                    format!("{name}: picks exceed the pool count of {available}")
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

    /// Full construction check, independent of the picking episode's cap.
    pub fn check(&self, picks: &BTreeMap<String, usize>) -> Vec<DeckIssue> {
        let mut issues = self.partial_issues(picks, &basic_lands());
        if picks.values().copied().fold(0usize, usize::saturating_add) < LIMITED_MIN_CARDS {
            issues.push(DeckIssue::format(
                "Limited requires at least 40 main-deck cards",
            ));
        }
        issues
    }

    /// Every next card in name order; invalid prefixes fail rather than truncate.
    pub fn choices(&self, picks: &BTreeMap<String, usize>) -> Result<LimitedChoices, ContentError> {
        let mut cards = basic_lands();
        reject(&self.partial_issues(picks, &cards))?;
        let size = picks.values().copied().fold(0usize, usize::saturating_add);
        if size > self.max_cards {
            return Err(ContentError("picks exceed the episode cap".into()));
        }
        cards.extend(self.pool.iter().filter_map(|(name, count)| {
            (picks.get(name).copied().unwrap_or(0) < *count).then(|| name.clone())
        }));
        Ok(LimitedChoices {
            cards: if size == self.max_cards {
                Vec::new()
            } else {
                cards.into_iter().collect()
            },
            // Admission and pool counts passed above; only the minimum remains.
            can_stop: size >= LIMITED_MIN_CARDS,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_picks_use_pool_counts_and_stop_only_at_legal_builds() {
        let rules = LimitedRules::new(BTreeMap::from([("Llanowar Elves".into(), 6)]), 60).unwrap();
        let mut picks = BTreeMap::from([("Llanowar Elves".into(), 6), ("Forest".into(), 33)]);
        let choices = rules.choices(&picks).unwrap();
        assert!(!choices.can_stop);
        assert!(!choices.cards.contains(&"Llanowar Elves".into()));
        assert!(choices.cards.contains(&"Forest".into()));
        picks.insert("Forest".into(), 34);
        assert!(rules.choices(&picks).unwrap().can_stop);
        picks.insert("Forest".into(), 54);
        assert_eq!(
            rules.choices(&picks).unwrap(),
            LimitedChoices {
                cards: vec![],
                can_stop: true
            }
        );
        picks.insert("Forest".into(), 55);
        assert!(rules.check(&picks).is_empty());
        assert!(rules.choices(&picks).is_err());
        picks.insert("Llanowar Elves".into(), 7);
        assert!(!rules.check(&picks).is_empty());
        assert!(rules.choices(&picks).is_err());
    }

    #[test]
    fn limited_rejects_bad_inputs_and_allows_an_empty_pool() {
        assert!(LimitedRules::new(BTreeMap::new(), 39).is_err());
        for name in ["Not a Card", "Ally"] {
            assert!(LimitedRules::new(BTreeMap::from([(name.into(), 1)]), 60).is_err());
        }
        assert!(LimitedRules::new(BTreeMap::from([("Forest".into(), 0)]), 60).is_err());
        let rules = LimitedRules::new(BTreeMap::new(), 60).unwrap();
        assert!(!rules.choices(&BTreeMap::new()).unwrap().can_stop);
        assert!(rules
            .choices(&BTreeMap::from([("Llanowar Elves".into(), 1)]))
            .is_err());
        assert!(rules
            .choices(&BTreeMap::from([("Forest".into(), 0)]))
            .is_err());
        assert!(
            rules
                .choices(&BTreeMap::from([("Forest".into(), 40)]))
                .unwrap()
                .can_stop
        );
    }
}
