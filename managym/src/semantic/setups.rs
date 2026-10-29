//! Authored setups: the fixed decklists in `content/semantic/current/setups.json`.
//!
//! Each setup names its decks by card key and becomes one [`AuthoredPack`]: a
//! [`ContentPack`] holding exactly the deck cards and the tokens they create,
//! in key order, plus the decks as player configurations. A match whose two
//! decklists equal an authored setup's runs on that pack
//! ([`content_pack_for_authored_match`]); every other match uses the default
//! pack. Card behavior comes only from the set files; this file is decklists.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

use serde::Deserialize;

use super::ContentError;
use crate::{
    cards::content_pack::ContentPack,
    cardsets,
    state::{card::CardDefinition, player::PlayerConfig},
};

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

impl AuthoredDeck {
    /// Whether `config` plays exactly this deck and sideboard.
    fn matches(&self, config: &PlayerConfig) -> bool {
        self.cards == config.decklist && self.sideboard == config.sideboard
    }
}

/// One setup's pack and decks.
#[derive(Clone, Debug)]
pub struct AuthoredPack {
    /// The setup's name in `setups.json` (for example `two_deck`).
    pub preset: String,
    pub pack_key: String,
    pub content: Arc<ContentPack>,
    pub decks: Vec<AuthoredDeck>,
}

impl AuthoredPack {
    pub fn deck(&self, key: &str) -> Result<&AuthoredDeck, ContentError> {
        self.decks
            .iter()
            .find(|deck| deck.key == key)
            .ok_or_else(|| ContentError::Malformed(format!("unknown authored deck {key:?}")))
    }

    pub fn player_config(&self, name: &str, key: &str) -> Result<PlayerConfig, ContentError> {
        let deck = self.deck(key)?;
        Ok(PlayerConfig::new(name, deck.cards.clone()).with_sideboard(deck.sideboard.clone()))
    }

    pub fn decklist(&self, key: &str) -> Result<BTreeMap<String, usize>, ContentError> {
        Ok(self.deck(key)?.cards.clone())
    }
}

/// Every authored setup, in `setups.json` order.
pub fn authored_packs() -> Result<&'static [AuthoredPack], ContentError> {
    static PACKS: OnceLock<Result<Vec<AuthoredPack>, ContentError>> = OnceLock::new();
    PACKS
        .get_or_init(|| {
            let sources: BTreeMap<String, SetupSource> = serde_json::from_str(SETUPS)
                .map_err(|error| ContentError::Malformed(format!("setups.json: {error}")))?;
            sources
                .into_iter()
                .map(|(preset, source)| build(preset, source))
                .collect()
        })
        .as_deref()
        .map_err(Clone::clone)
}

/// The authored setup named `preset` in `setups.json`.
pub fn authored_pack(preset: &str) -> Result<&'static AuthoredPack, ContentError> {
    authored_packs()?
        .iter()
        .find(|pack| pack.preset == preset)
        .ok_or_else(|| ContentError::Malformed(format!("unknown authored setup {preset:?}")))
}

/// The authored setup whose pack key is `pack_key`.
pub fn authored_pack_for_key(pack_key: &str) -> Result<&'static AuthoredPack, ContentError> {
    authored_packs()?
        .iter()
        .find(|pack| pack.pack_key == pack_key)
        .ok_or_else(|| ContentError::Malformed(format!("unknown content pack {pack_key:?}")))
}

fn build(preset: String, source: SetupSource) -> Result<AuthoredPack, ContentError> {
    let catalog = cardsets::catalog();
    let by_key = catalog
        .iter()
        .map(|definition| (definition.key, definition))
        .collect::<BTreeMap<_, _>>();
    let by_name = catalog
        .iter()
        .map(|definition| (definition.name.as_str(), definition))
        .collect::<BTreeMap<_, _>>();
    let context = |message: String| ContentError::Malformed(format!("setup {preset}: {message}"));

    let mut members = BTreeSet::new();
    let mut decks = Vec::new();
    let mut keys = BTreeSet::new();
    for deck in source.decks {
        if !keys.insert(deck.key.clone()) {
            return Err(context(format!("duplicate deck {}", deck.key)));
        }
        let mut resolve = |entries: &[EntrySource], sideboard: bool| {
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
                members.insert(definition.key);
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

    // Close over the tokens the deck cards create.
    let mut pending = members.iter().copied().collect::<Vec<_>>();
    while let Some(key) = pending.pop() {
        for name in created_tokens(by_key[key]) {
            let token = by_name
                .get(name.as_str())
                .ok_or_else(|| context(format!("{key} creates unknown token {name:?}")))?;
            if members.insert(token.key) {
                pending.push(token.key);
            }
        }
    }
    let definitions = members.into_iter().map(|key| by_key[key].clone()).collect();
    Ok(AuthoredPack {
        preset,
        pack_key: source.pack_key.clone(),
        content: Arc::new(ContentPack::from_definitions(
            Some(source.pack_key),
            definitions,
        )),
        decks,
    })
}

/// Names of the token definitions `definition`'s effects create: every
/// `token_name` at any depth of its serialized effects, so an effect that nests
/// others (modes, branches, granted abilities) needs no visitor entry.
pub(crate) fn created_tokens(definition: &CardDefinition) -> BTreeSet<String> {
    fn visit(value: &serde_json::Value, out: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(fields) => {
                for (name, field) in fields {
                    match (name.as_str(), field) {
                        ("token_name", serde_json::Value::String(token)) => {
                            out.insert(token.clone());
                        }
                        _ => visit(field, out),
                    }
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|item| visit(item, out)),
            _ => {}
        }
    }
    let mut out = BTreeSet::new();
    visit(
        &serde_json::to_value(definition).expect("card definitions serialize"),
        &mut out,
    );
    out
}

/// The authored pack for an exact two-deck match, if one is authored.
///
/// An explicit `content_pack` on the configs must name an authored pack and
/// both decks must be its setups; otherwise configs whose decklists equal one
/// authored setup's run on it, and ambiguity fails closed.
pub(crate) fn content_pack_for_authored_match(
    player_configs: &[PlayerConfig],
) -> Result<Option<Arc<ContentPack>>, ContentError> {
    Ok(select(player_configs, authored_packs()?)?.map(|pack| Arc::clone(&pack.content)))
}

fn admits(pack: &AuthoredPack, config: &PlayerConfig) -> bool {
    pack.decks.iter().any(|deck| deck.matches(config))
}

fn select<'a>(
    player_configs: &[PlayerConfig],
    packs: &'a [AuthoredPack],
) -> Result<Option<&'a AuthoredPack>, ContentError> {
    if let Some(key) = player_configs.iter().find_map(|p| p.content_pack.as_ref()) {
        let pack = packs
            .iter()
            .find(|pack| &pack.pack_key == key)
            .ok_or_else(|| {
                ContentError::Malformed(format!("unknown explicit content pack {key}"))
            })?;
        if player_configs.len() != 2
            || player_configs
                .iter()
                .any(|p| p.content_pack.as_ref() != Some(key) || !admits(pack, p))
        {
            return Err(ContentError::Malformed(
                "explicit pack requires two admitted authored setups".into(),
            ));
        }
        return Ok(Some(pack));
    }
    let matches = packs
        .iter()
        .filter(|pack| exact(pack, player_configs))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [pack] => Ok(Some(pack)),
        _ => Err(ContentError::Malformed(format!(
            "authored content catalog is ambiguous for packs {:?}",
            matches
                .iter()
                .map(|pack| pack.pack_key.as_str())
                .collect::<Vec<_>>()
        ))),
    }
}

/// The configs are exactly the pack's setups, each used once.
fn exact(pack: &AuthoredPack, player_configs: &[PlayerConfig]) -> bool {
    if player_configs.len() != pack.decks.len() {
        return false;
    }
    let mut unmatched = pack.decks.iter().collect::<Vec<_>>();
    for config in player_configs {
        let Some(index) = unmatched.iter().position(|deck| deck.matches(config)) else {
            return false;
        };
        unmatched.remove(index);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configs(pack: &AuthoredPack, left: &str, right: &str) -> Vec<PlayerConfig> {
        vec![
            pack.player_config("left", left).unwrap(),
            pack.player_config("right", right).unwrap(),
        ]
    }

    #[test]
    fn explicit_mirrors_preserve_pack_and_reject_unadmitted_setups() {
        let packs = authored_packs().unwrap();
        let pack = authored_pack_for_key("ur-lessons-vs-gw-allies").unwrap();
        for deck in &pack.decks {
            let mut config = pack.player_config(&deck.key, &deck.key).unwrap();
            config.content_pack = Some(pack.pack_key.clone());
            let pair = vec![config.clone(), config.clone()];
            assert_eq!(
                select(&pair, packs).unwrap().unwrap().pack_key,
                pack.pack_key
            );
            let mut wrong = pair.clone();
            wrong[0].content_pack = None;
            assert!(select(&wrong, packs).is_err());
            wrong = pair.clone();
            wrong[0].decklist.clear();
            assert!(select(&wrong, packs).is_err());
            wrong = pair.clone();
            wrong[0].content_pack = Some("missing-pack".into());
            assert!(select(&wrong, packs).is_err());
        }
    }

    #[test]
    fn complexity_roster_is_explicit_and_vanilla() {
        let packs = authored_packs().unwrap();
        let pack = authored_pack("complexity").unwrap();
        assert_eq!(pack.pack_key, "mtg122-complexity-v1");
        assert_eq!(pack.decks.len(), 6);
        for name in ["Savannah Lions", "Glory Seeker", "Hill Giant"] {
            let id = pack.content.definition_id(name).unwrap();
            let definition = pack.content.definition(id).unwrap();
            assert!(definition.abilities.is_empty() && definition.activated_abilities.is_empty());
        }
        for deck in &pack.decks {
            let mut config = pack.player_config(&deck.key, &deck.key).unwrap();
            config.content_pack = Some(pack.pack_key.clone());
            let pair = [config.clone(), config];
            assert_eq!(
                select(&pair, packs).unwrap().unwrap().pack_key,
                "mtg122-complexity-v1"
            );
        }
        // An ordinary product pair still chooses the original product pack.
        let product = authored_pack("two_deck").unwrap();
        let pair = configs(product, "ur_lessons", "gw_allies");
        assert_eq!(
            select(&pair, packs).unwrap().unwrap().pack_key,
            "ur-lessons-vs-gw-allies"
        );
    }

    #[test]
    fn duplicate_exact_catalog_entries_fail_closed() {
        let pack = authored_pack("jeong_increment").unwrap();
        let player_configs = configs(pack, "ur_lessons", "gw_allies_jeong");
        let error = select(&player_configs, &[pack.clone(), pack.clone()])
            .expect_err("ambiguous catalog must fail closed");
        assert!(error.to_string().contains("catalog is ambiguous"));
    }

    #[test]
    fn packs_close_over_created_tokens() {
        let pack = authored_pack("two_deck").unwrap();
        assert!(pack.content.definition_id("Ally").is_some());
        for (_, definition) in pack.content.definition_entries() {
            for token in created_tokens(definition) {
                assert!(pack.content.definition_id(&token).is_some(), "{token}");
            }
        }
    }
}
