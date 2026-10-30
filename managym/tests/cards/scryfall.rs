//! Every definition names the Scryfall printing of the set file it lives in.
//! `check_scryfall.py` beside these tests confirms names and sets online; this
//! offline check keeps the ids present, well formed and unique.

use std::collections::BTreeSet;

use managym::cardsets::catalog;

fn well_formed(id: &str) -> bool {
    let groups = id.split('-').collect::<Vec<_>>();
    groups.iter().map(|group| group.len()).eq([8, 4, 4, 4, 12])
        && groups.iter().all(|group| {
            group
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
}

#[test]
fn every_definition_has_a_well_formed_unique_scryfall_id() {
    let mut seen = BTreeSet::new();
    for definition in catalog() {
        assert!(
            well_formed(definition.scryfall_id),
            "{}: malformed Scryfall id {:?}",
            definition.key,
            definition.scryfall_id
        );
        assert!(
            seen.insert(definition.scryfall_id),
            "{}: duplicate Scryfall id",
            definition.key
        );
    }
}

#[test]
fn every_definition_key_names_its_set_and_is_unique() {
    let mut keys = BTreeSet::new();
    for definition in catalog() {
        let (set, card) = definition.key.split_once('.').expect("<set>.<card> key");
        assert!(!set.is_empty() && !card.is_empty(), "{}", definition.key);
        assert_eq!(set == "token", definition.is_token, "{}", definition.key);
        assert!(
            keys.insert(definition.key),
            "duplicate key {}",
            definition.key
        );
    }
}
