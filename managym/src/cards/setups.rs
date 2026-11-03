//! Authored setups: the fixed decklists in `content/semantic/current/setups.json`.
//!
//! Each setup names its decks by card key and becomes one [`AuthoredSetup`]:
//! its decks as player configurations over the one catalog
//! ([`crate::cards::content_pack::default_content_pack`]), like any other
//! match. A setup is named by its preset (`two_deck`) or by the key frozen
//! evidence records for it (`ur-lessons-vs-gw-allies`, `pack_key` in
//! `setups.json`). Card behavior comes only from the set files; this file is
//! decklists.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

use serde::Deserialize;

use super::ContentError;
use crate::{cardsets, state::player::PlayerConfig};

const SETUPS: &str = include_str!("../../../content/semantic/current/setups.json");

#[derive(Deserialize)]
struct SetupSource {
    pack_key: String,
    decks: Vec<DeckSource>,
}

#[derive(Deserialize)]
struct DeckSource {
    key: String,
    card_count: usize,
    cards: Vec<EntrySource>,
    #[serde(default)]
    sideboard: Vec<EntrySource>,
}

#[derive(Deserialize)]
struct EntrySource {
    definition: String,
    count: usize,
}

/// One authored deck, by card name.
#[derive(Clone, Debug)]
pub struct AuthoredDeck {
    pub key: String,
    pub cards: BTreeMap<String, usize>,
    pub sideboard: BTreeMap<String, usize>,
}

/// One setup's decks.
#[derive(Clone, Debug)]
pub struct AuthoredSetup {
    /// The setup's name in `setups.json` (for example `two_deck`).
    pub preset: String,
    /// The name frozen evidence records for this setup (`pack_key` in
    /// `setups.json` and in retained manifests, receipts and Academy packs).
    pub pack_key: String,
    pub decks: Vec<AuthoredDeck>,
}

impl AuthoredSetup {
    pub fn deck(&self, key: &str) -> Result<&AuthoredDeck, ContentError> {
        self.decks
            .iter()
            .find(|deck| deck.key == key)
            .ok_or_else(|| ContentError(format!("unknown authored deck {key:?}")))
    }

    pub fn player_config(&self, name: &str, key: &str) -> Result<PlayerConfig, ContentError> {
        let deck = self.deck(key)?;
        Ok(PlayerConfig::new(name, deck.cards.clone()).with_sideboard(deck.sideboard.clone()))
    }

    pub fn decklist(&self, key: &str) -> Result<BTreeMap<String, usize>, ContentError> {
        Ok(self.deck(key)?.cards.clone())
    }
}

/// Every authored setup, in preset order.
pub fn authored_setups() -> Result<&'static [AuthoredSetup], ContentError> {
    static SETUPS_BY_PRESET: OnceLock<Result<Vec<AuthoredSetup>, ContentError>> = OnceLock::new();
    SETUPS_BY_PRESET
        .get_or_init(|| {
            let sources: BTreeMap<String, SetupSource> = serde_json::from_str(SETUPS)
                .map_err(|error| ContentError(format!("setups.json: {error}")))?;
            sources
                .into_iter()
                .map(|(preset, source)| build(preset, source))
                .collect()
        })
        .as_deref()
        .map_err(Clone::clone)
}

/// The authored setup named `name`: its preset or its evidence `pack_key`.
pub fn authored_setup(name: &str) -> Result<&'static AuthoredSetup, ContentError> {
    authored_setups()?
        .iter()
        .find(|setup| setup.preset == name || setup.pack_key == name)
        .ok_or_else(|| ContentError(format!("unknown authored setup {name:?}")))
}

fn build(preset: String, source: SetupSource) -> Result<AuthoredSetup, ContentError> {
    let by_key = cardsets::catalog()
        .iter()
        .map(|definition| (definition.key, definition))
        .collect::<BTreeMap<_, _>>();
    let context = |message: String| ContentError(format!("setup {preset}: {message}"));

    let mut decks = Vec::new();
    let mut keys = BTreeSet::new();
    for deck in source.decks {
        if !keys.insert(deck.key.clone()) {
            return Err(context(format!("duplicate deck {}", deck.key)));
        }
        let resolve = |entries: &[EntrySource], sideboard: bool| {
            let mut out = BTreeMap::new();
            for entry in entries {
                let definition = by_key
                    .get(entry.definition.as_str())
                    .ok_or_else(|| context(format!("unknown card {:?}", entry.definition)))?;
                if entry.count == 0 || (sideboard && definition.is_token) {
                    return Err(context(format!("invalid entry {:?}", entry.definition)));
                }
                if out.insert(definition.name.clone(), entry.count).is_some() {
                    return Err(context(format!("repeated entry {:?}", entry.definition)));
                }
            }
            Ok(out)
        };
        let cards = resolve(&deck.cards, false)?;
        let sideboard = resolve(&deck.sideboard, true)?;
        if cards.values().sum::<usize>() != deck.card_count {
            return Err(context(format!(
                "{} entries do not sum to {}",
                deck.key, deck.card_count
            )));
        }
        decks.push(AuthoredDeck {
            key: deck.key,
            cards,
            sideboard,
        });
    }
    Ok(AuthoredSetup {
        preset,
        pack_key: source.pack_key,
        decks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setups_resolve_by_preset_or_evidence_key() {
        let setups = authored_setups().unwrap();
        let names = setups
            .iter()
            .flat_map(|setup| [setup.preset.as_str(), setup.pack_key.as_str()])
            .collect::<Vec<_>>();
        assert_eq!(
            names.len(),
            names.iter().collect::<BTreeSet<_>>().len(),
            "setup names must be unique"
        );
        let by_preset = authored_setup("two_deck").unwrap();
        let by_key = authored_setup("ur-lessons-vs-gw-allies").unwrap();
        assert!(std::ptr::eq(by_preset, by_key));
        assert!(authored_setup("missing-setup").is_err());
        assert!(by_preset.deck("missing-deck").is_err());
    }

    #[test]
    fn complexity_roster_is_explicit_and_vanilla() {
        let setup = authored_setup("complexity").unwrap();
        assert_eq!(setup.pack_key, "mtg122-complexity-v1");
        assert_eq!(setup.decks.len(), 6);
        let catalog = crate::cards::content_pack::default_content_pack();
        for name in ["Savannah Lions", "Glory Seeker", "Hill Giant"] {
            let id = catalog.definition_id(name).unwrap();
            let definition = catalog.definition(id).unwrap();
            assert!(definition.abilities.is_empty() && definition.activated_abilities.is_empty());
        }
    }
}
