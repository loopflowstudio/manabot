//! Construction admission for submitted development decks. Content eligibility
//! belongs to the pool descriptor; this module never infers rules from names.

use serde::{Deserialize, Serialize};

use super::{content_pool, pool_content, ContentPool, Eligibility, IrError};
use crate::{
    cards::content_pack::ContentPack,
    state::{deck::Deck, player::PlayerConfig},
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

fn construction_issues(deck: &Deck, ruleset: &str) -> Vec<DeckIssue> {
    let mut issues = Vec::new();
    if ruleset != DEVELOPMENT_RULESET {
        issues.push(DeckIssue::format(format!(
            "unknown construction ruleset {ruleset:?}; choose {DEVELOPMENT_RULESET}"
        )));
    }
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
/// Env setup invokes this same admission for every submitted non-preset deck.
pub fn admit_deck(pack_key: &str, deck: &Deck, ruleset: &str) -> Result<Vec<DeckIssue>, IrError> {
    Ok(admit_in_pool(
        &content_pool(pack_key)?,
        &*pool_content(pack_key)?,
        deck,
        ruleset,
    ))
}

/// Admission against one pool descriptor and the runtime content it describes.
pub(crate) fn admit_in_pool(
    pool: &ContentPool,
    content: &ContentPack,
    deck: &Deck,
    ruleset: &str,
) -> Vec<DeckIssue> {
    let mut issues = construction_issues(deck, ruleset);
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
                (None, _) => ("unknown", "card is not in the selected content pool"),
                (Some(row), None) if row.eligibility != Eligibility::Eligible => (
                    "unsupported",
                    row.reason
                        .as_deref()
                        .unwrap_or("card is not supported in this pool"),
                ),
                (Some(row), None) => {
                    let definition = row.card_def_id.and_then(|id| content.definition(id));
                    match definition {
                        Some(definition) if !definition.is_token => continue,
                        Some(_) => ("format", "tokens cannot be submitted in main or sideboard"),
                        None => (
                            "unsupported",
                            "supported definition is missing from runtime content",
                        ),
                    }
                }
                _ => (
                    "ambiguous",
                    "card name identifies multiple pool definitions",
                ),
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
pub fn development_rejection(
    pack_key: &str,
    player: &PlayerConfig,
) -> Result<Option<String>, IrError> {
    let issues = admit_deck(pack_key, &player.deck, DEVELOPMENT_RULESET)?;
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
    fn novel_pool_decks_admit_without_an_authored_setup() {
        let main = BTreeMap::from([("Forest".into(), 10), ("Mountain".into(), 10)]);
        let sideboard = BTreeMap::from([("Island".into(), 1)]);
        let deck = Deck::new(main.clone(), sideboard);
        assert!(admit_deck("tla-pool-v1", &deck, DEVELOPMENT_RULESET)
            .unwrap()
            .is_empty());
        let unsupported = Deck::new(main, BTreeMap::from([("Accumulate Wisdom".into(), 1)]));
        let issues = admit_deck("tla-pool-v1", &unsupported, DEVELOPMENT_RULESET).unwrap();
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
        assert!(construction_issues(&deck, DEVELOPMENT_RULESET).is_empty());
        assert!(construction_issues(&deck, "standard")
            .iter()
            .any(|issue| issue.code == "format"));
    }

    #[test]
    fn empty_zero_and_overflowing_allocations_reject() {
        assert!(!construction_issues(&Deck::default(), DEVELOPMENT_RULESET).is_empty());
        let deck = Deck::new(
            BTreeMap::from([("Forest".into(), 0)]),
            BTreeMap::from([("Island".into(), usize::MAX), ("Mountain".into(), 2)]),
        );
        let issues = construction_issues(&deck, DEVELOPMENT_RULESET);
        assert!(issues.iter().any(|issue| issue.code == "count"));
        assert!(issues
            .iter()
            .any(|issue| issue.message.contains("resource limit")));
    }
}
