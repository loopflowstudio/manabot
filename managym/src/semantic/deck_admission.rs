//! Construction admission for submitted development decks. Content eligibility
//! belongs to the pool descriptor; this module never infers rules from names.

use std::collections::BTreeMap;

use serde::Serialize;

use super::{content_pool, pool_content, Eligibility, IrError};

pub const DEVELOPMENT_RULESET: &str = "development-v1";
/// Operational allocation bound, not a tournament construction rule.
pub const MAX_SETUP_CARDS: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeckSection {
    Main,
    Sideboard,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DeckIssue {
    pub section: Option<DeckSection>,
    pub card_name: Option<String>,
    pub code: String,
    pub message: String,
}

impl DeckIssue {
    fn format(message: impl Into<String>) -> Self {
        Self {
            section: None,
            card_name: None,
            code: "format".into(),
            message: message.into(),
        }
    }
}

fn construction_issues(
    main: &BTreeMap<String, usize>,
    sideboard: &BTreeMap<String, usize>,
    ruleset: &str,
) -> Vec<DeckIssue> {
    let mut issues = Vec::new();
    if ruleset != DEVELOPMENT_RULESET {
        issues.push(DeckIssue::format(format!(
            "unknown construction ruleset {ruleset:?}; choose {DEVELOPMENT_RULESET}"
        )));
    }
    if main.is_empty() {
        issues.push(DeckIssue {
            section: Some(DeckSection::Main),
            ..DeckIssue::format("development-v1 requires a nonempty main deck")
        });
    }
    let mut total = 0usize;
    for (section, cards) in [
        (DeckSection::Main, main),
        (DeckSection::Sideboard, sideboard),
    ] {
        for (name, count) in cards {
            if *count == 0 {
                issues.push(DeckIssue {
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
pub fn admit_deck(
    pack_key: &str,
    main: &BTreeMap<String, usize>,
    sideboard: &BTreeMap<String, usize>,
    ruleset: &str,
) -> Result<Vec<DeckIssue>, IrError> {
    let pool = content_pool(pack_key)?;
    let content = pool_content(pack_key)?;
    let mut issues = construction_issues(main, sideboard, ruleset);
    for (section, cards) in [
        (DeckSection::Main, main),
        (DeckSection::Sideboard, sideboard),
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
                        .unwrap_or("card has no complete qualification in this pool"),
                ),
                (Some(row), None) => {
                    let definition = row.card_def_id.and_then(|id| content.definition(id));
                    match definition {
                        Some(definition) if !definition.is_token => continue,
                        Some(_) => ("format", "tokens cannot be submitted in main or sideboard"),
                        None => (
                            "unsupported",
                            "qualified definition is missing from runtime content",
                        ),
                    }
                }
                _ => (
                    "ambiguous",
                    "card name identifies multiple pool definitions",
                ),
            };
            issues.push(DeckIssue {
                section: Some(section),
                card_name: Some(name.clone()),
                code: code.into(),
                message: format!("{name}: {reason}"),
            });
        }
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn novel_pool_decks_admit_without_an_authored_setup() {
        let main = BTreeMap::from([("Forest".into(), 10), ("Mountain".into(), 10)]);
        let sideboard = BTreeMap::from([("Island".into(), 1)]);
        assert!(
            admit_deck("tla-pool-v1", &main, &sideboard, DEVELOPMENT_RULESET)
                .unwrap()
                .is_empty()
        );
        let unsupported = BTreeMap::from([("Airbending Lesson".into(), 1)]);
        let issues = admit_deck("tla-pool-v1", &main, &unsupported, DEVELOPMENT_RULESET).unwrap();
        assert!(issues
            .iter()
            .any(|issue| issue.code == "unsupported"
                && issue.section == Some(DeckSection::Sideboard)));
    }

    #[test]
    fn development_construction_is_not_tournament_legality() {
        let main = BTreeMap::from([("Forest".into(), 8)]);
        let sideboard = BTreeMap::from([("Island".into(), 20)]);
        assert!(construction_issues(&main, &sideboard, DEVELOPMENT_RULESET).is_empty());
        assert!(construction_issues(&main, &sideboard, "standard")
            .iter()
            .any(|issue| issue.code == "format"));
    }

    #[test]
    fn empty_zero_and_overflowing_allocations_reject() {
        assert!(
            !construction_issues(&BTreeMap::new(), &BTreeMap::new(), DEVELOPMENT_RULESET)
                .is_empty()
        );
        let main = BTreeMap::from([("Forest".into(), 0)]);
        let sideboard = BTreeMap::from([("Island".into(), usize::MAX), ("Mountain".into(), 2)]);
        let issues = construction_issues(&main, &sideboard, DEVELOPMENT_RULESET);
        assert!(issues.iter().any(|issue| issue.code == "count"));
        assert!(issues
            .iter()
            .any(|issue| issue.message.contains("resource limit")));
    }
}
