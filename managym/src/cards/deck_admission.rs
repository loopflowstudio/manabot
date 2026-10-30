//! Construction admission for submitted development decks. A submitted deck
//! needs only the installed world: its cards are supported when the world
//! defines them ([`crate::world`]). Set restrictions belong to deck making, not
//! admission. Content eligibility comes from the world's descriptor rows; this
//! module never infers rules from names.

use serde::{Deserialize, Serialize};

use super::{content_pool, pool_content, ContentError, ContentPool, Eligibility};
use crate::{
    cards::content_pack::ContentPack,
    state::{deck::Deck, player::PlayerConfig},
    world::WORLD_POOL,
};

pub const DEVELOPMENT_RULESET: &str = "development-v1";
/// Operational allocation bound, not a tournament construction rule.
pub const MAX_SETUP_CARDS: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeckSection {
    Main,
    Sideboard,
}

impl DeckSection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Sideboard => "sideboard",
        }
    }
}

/// One actionable problem with a submitted deck. `line` is the 1-based source
/// line when the deck came from a decklist; admission of counts alone has none.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct DeckIssue {
    pub line: Option<usize>,
    pub section: Option<DeckSection>,
    pub card_name: Option<String>,
    pub code: String,
    pub message: String,
}

impl DeckIssue {
    pub(crate) fn format(message: impl Into<String>) -> Self {
        Self {
            line: None,
            section: None,
            card_name: None,
            code: "format".into(),
            message: message.into(),
        }
    }
}

/// The world's construction rules ([`DEVELOPMENT_RULESET`]).
fn construction_issues(deck: &Deck) -> Vec<DeckIssue> {
    let mut issues = Vec::new();
    if deck.main.is_empty() {
        issues.push(DeckIssue {
            section: Some(DeckSection::Main),
            ..DeckIssue::format("development-v1 requires a nonempty main deck")
        });
    }
    let mut total = 0usize;
    for (section, cards) in [
        (DeckSection::Main, &deck.main),
        (DeckSection::Sideboard, &deck.sideboard),
    ] {
        for (name, count) in cards {
            if *count == 0 {
                issues.push(DeckIssue {
                    line: None,
                    section: Some(section),
                    card_name: Some(name.clone()),
                    code: "count".into(),
                    message: "quantity must be positive".into(),
                });
            }
            total = total.saturating_add(*count);
        }
    }
    if total > MAX_SETUP_CARDS {
        issues.push(DeckIssue::format(format!(
            "setup exceeds the {MAX_SETUP_CARDS}-card resource limit (main plus sideboard)"
        )));
    }
    issues
}

/// Return actionable construction/content issues without allocating a game.
/// Env setup invokes this same admission for every deck submitted to the world.
pub fn admit_deck(deck: &Deck) -> Result<Vec<DeckIssue>, ContentError> {
    Ok(check_decks(std::slice::from_ref(deck))?.remove(0))
}

/// Each deck's issues in this world, in input order; an empty entry admits.
/// One cheap upfront check for a batch: the world is read once, no game is
/// allocated, and game creation still admits every deck again.
pub fn check_decks(decks: &[Deck]) -> Result<Vec<Vec<DeckIssue>>, ContentError> {
    let descriptor = content_pool(WORLD_POOL)?;
    let content = pool_content(WORLD_POOL)?;
    Ok(decks
        .iter()
        .map(|deck| admit_in_world(&descriptor, &content, deck))
        .collect())
}

/// Admission against the world's descriptor and the runtime content it
/// describes; tests substitute descriptors to reach ambiguous names.
pub(crate) fn admit_in_world(
    pool: &ContentPool,
    content: &ContentPack,
    deck: &Deck,
) -> Vec<DeckIssue> {
    let mut issues = construction_issues(deck);
    for (section, cards) in [
        (DeckSection::Main, &deck.main),
        (DeckSection::Sideboard, &deck.sideboard),
    ] {
        for name in cards.keys() {
            let mut rows = pool
                .definitions
                .iter()
                .filter(|row| &row.registry_name == name);
            let (code, reason) = match (rows.next(), rows.next()) {
                (None, _) => ("unknown", "card is not defined in this world"),
                (Some(row), None) if row.eligibility != Eligibility::Eligible => (
                    "unsupported",
                    row.reason
                        .as_deref()
                        .unwrap_or("card is not supported in this world"),
                ),
                (Some(row), None) => {
                    let definition = content.definition(row.card_def_id);
                    match definition {
                        Some(definition) if !definition.is_token => continue,
                        Some(_) => ("format", "tokens cannot be submitted in main or sideboard"),
                        None => (
                            "unsupported",
                            "supported definition is missing from runtime content",
                        ),
                    }
                }
                _ => ("ambiguous", "card name identifies multiple definitions"),
            };
            issues.push(DeckIssue {
                line: None,
                section: Some(section),
                card_name: Some(name.clone()),
                code: code.into(),
                message: format!("{name}: {reason}"),
            });
        }
    }
    issues
}

/// Why a player's submitted deck fails development admission, or `None` when it
/// passes. Game creation and training-deck replacement share this rejection.
pub fn development_rejection(player: &PlayerConfig) -> Result<Option<String>, ContentError> {
    let issues = admit_deck(&player.deck)?;
    Ok((!issues.is_empty()).then(|| {
        issues
            .iter()
            .map(|issue| issue.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn novel_decks_admit_against_the_world_without_an_authored_setup() {
        let main = BTreeMap::from([("Forest".into(), 10), ("Mountain".into(), 10)]);
        let sideboard = BTreeMap::from([("Island".into(), 1)]);
        let deck = Deck::new(main.clone(), sideboard);
        assert!(admit_deck(&deck).unwrap().is_empty());
        let unsupported = Deck::new(main, BTreeMap::from([("Accumulate Wisdom".into(), 1)]));
        let checked = check_decks(&[deck, unsupported.clone()]).unwrap();
        assert!(checked[0].is_empty());
        assert_eq!(checked[1], admit_deck(&unsupported).unwrap());
        let issues = &checked[1];
        assert!(issues
            .iter()
            .any(|issue| issue.code == "unsupported"
                && issue.section == Some(DeckSection::Sideboard)));
    }

    #[test]
    fn development_construction_is_not_tournament_legality() {
        let deck = Deck::new(
            BTreeMap::from([("Forest".into(), 8)]),
            BTreeMap::from([("Island".into(), 20)]),
        );
        assert!(construction_issues(&deck).is_empty());
    }

    #[test]
    fn empty_zero_and_overflowing_allocations_reject() {
        assert!(!construction_issues(&Deck::default()).is_empty());
        let deck = Deck::new(
            BTreeMap::from([("Forest".into(), 0)]),
            BTreeMap::from([("Island".into(), usize::MAX), ("Mountain".into(), 2)]),
        );
        let issues = construction_issues(&deck);
        assert!(issues.iter().any(|issue| issue.code == "count"));
        assert!(issues
            .iter()
            .any(|issue| issue.message.contains("resource limit")));
    }
}
