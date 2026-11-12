//! Every card names the Scryfall printing of the set file it lives in, and
//! every token its first token printing unless it was never printed.
//! `check_scryfall.py` beside these tests confirms names and sets online; this
//! offline check keeps the ids well formed and unique and the keys in shape.

use std::collections::BTreeSet;

use managym_cards::{catalog, token_catalog};

fn well_formed(id: &str) -> bool {
    let groups = id.split('-').collect::<Vec<_>>();
    groups.iter().map(|group| group.len()).eq([8, 4, 4, 4, 12])
        && groups.iter().all(|group| {
            group
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
}

/// Every printed id, cards' then tokens', with the key that carries it.
fn printed_ids() -> impl Iterator<Item = (&'static str, &'static str)> {
    catalog()
        .iter()
        .map(|object| (object.key, object.scryfall_id))
        .chain(
            token_catalog()
                .iter()
                .filter_map(|token| Some((token.key, token.scryfall_id?))),
        )
}

#[test]
fn every_printed_definition_has_a_well_formed_unique_scryfall_id() {
    let mut seen = BTreeSet::new();
    for (key, id) in printed_ids() {
        assert!(well_formed(id), "{key}: malformed Scryfall id {id:?}");
        assert!(seen.insert(id), "{key}: duplicate Scryfall id");
    }
}

#[test]
fn every_card_key_names_its_set_and_is_unique() {
    let mut keys = BTreeSet::new();
    for definition in catalog() {
        let (set, object) = definition.key.split_once('.').expect("<set>.<card> key");
        assert!(!set.is_empty() && !object.is_empty(), "{}", definition.key);
        assert_ne!(
            set, "token",
            "{}: tokens are TokenDefinitions",
            definition.key
        );
        assert!(
            keys.insert(definition.key),
            "duplicate key {}",
            definition.key
        );
    }
}

#[test]
fn every_token_key_is_a_unique_token_key() {
    let mut keys = BTreeSet::new();
    for token in token_catalog() {
        let name = token.key.strip_prefix("token.").expect("token.<name> key");
        assert!(!name.is_empty(), "{}", token.key);
        assert!(keys.insert(token.key), "duplicate key {}", token.key);
    }
}
