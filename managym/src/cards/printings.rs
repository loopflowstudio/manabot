//! Checked bibliographic set membership, separate from printed rules and tokens.
//!
//! The bundled nonbasic TLA inventory joins exact printings to authored keys.
//! `set_printings` validates the entire inventory before exposing any rows;
//! unknown sets fail rather than returning a partial supply. First-printing
//! ids on definitions and the world's rules digest remain unchanged.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{world_pool, ContentError, Eligibility};
use crate::state::card::CardDefId;

const TLA: &str = include_str!("printings/tla.json");

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Rarity {
    Common,
    Uncommon,
    Rare,
    Mythic,
}

impl Rarity {
    pub fn rotisserie_copies(self) -> u32 {
        match self {
            Self::Common => 4,
            Self::Uncommon => 2,
            Self::Rare | Self::Mythic => 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Printing {
    pub key: String,
    pub definition_scryfall_id: String,
    pub oracle_id: String,
    pub name: String,
    pub set: String,
    pub scryfall_id: String,
    pub rarity: Rarity,
}

#[derive(Deserialize)]
struct Inventory {
    printings: Vec<Printing>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SetPrinting {
    pub card_def_id: CardDefId,
    #[serde(flatten)]
    pub printing: Printing,
}

fn uuid(id: &str) -> bool {
    id.split('-').map(str::len).eq([8, 4, 4, 4, 12])
        && id.bytes().all(|b| b == b'-' || b.is_ascii_hexdigit())
}

fn validate(set: &str, text: &str, expected: usize) -> Result<Vec<SetPrinting>, ContentError> {
    let inventory: Inventory = serde_json::from_str(text)
        .map_err(|e| ContentError(format!("{set}: invalid printing metadata: {e}")))?;
    if inventory.printings.len() != expected {
        return Err(ContentError(format!(
            "{set}: expected {expected} nonbasic printing rows, got {}",
            inventory.printings.len()
        )));
    }
    let definitions = world_pool()
        .definitions
        .iter()
        .filter_map(|r| Some((r.key.as_deref()?, r)))
        .collect::<BTreeMap<_, _>>();
    let mut keys = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut oracles = BTreeSet::new();
    let mut rows = Vec::new();
    for printing in inventory.printings {
        let key = printing.key.clone();
        let fail = |why: &str| ContentError(format!("{set}/{key}: {why}"));
        if printing.set != set || !uuid(&printing.scryfall_id) || !uuid(&printing.oracle_id) {
            return Err(fail("invalid set or printing/oracle identity"));
        }
        if !keys.insert(key.clone())
            || !ids.insert(printing.scryfall_id.clone())
            || !oracles.insert(printing.oracle_id.clone())
        {
            return Err(fail("ambiguous duplicate key, printing or Oracle identity"));
        }
        let row = definitions
            .get(key.as_str())
            .ok_or_else(|| fail("definition is missing"))?;
        if row.eligibility != Eligibility::Eligible
            || row.scryfall_id.as_deref() != Some(printing.definition_scryfall_id.as_str())
        {
            return Err(fail("definition printing identity or eligibility differs"));
        }
        // Scryfall's combined face name identifies a transforming card; the
        // engine's registry names its front face for deck entry.
        if row.registry_name != printing.name.split(" // ").next().unwrap_or("") {
            return Err(fail("definition name differs"));
        }
        if crate::format::limited::basics()[row.card_def_id.0 as usize] {
            return Err(fail("basic land in nonbasic inventory"));
        }
        rows.push(SetPrinting {
            card_def_id: row.card_def_id,
            printing,
        });
    }
    let prefix = format!("{set}.");
    for (key, _) in definitions.iter().filter(|(key, row)| {
        key.starts_with(&prefix)
            && row.eligibility == Eligibility::Eligible
            && !crate::format::limited::basics()[row.card_def_id.0 as usize]
    }) {
        if !keys.contains(*key) {
            return Err(ContentError(format!(
                "{set}/{key}: missing printing metadata"
            )));
        }
    }
    rows.sort_by_key(|r| r.card_def_id);
    Ok(rows)
}

/// A complete checked nonbasic printing inventory. No network or prefix fallback.
pub fn set_printings(set: &str) -> Result<Vec<SetPrinting>, ContentError> {
    match set {
        "tla" => validate(set, TLA, 281),
        _ => Err(ContentError(format!(
            "{set}: no complete printing inventory"
        ))),
    }
}

/// Fingerprint of the bundled printing rows and their retrieval provenance.
pub fn printing_digest(set: &str) -> Result<String, ContentError> {
    set_printings(set)?;
    Ok(format!("{:x}", Sha256::digest(TLA.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printing_inventory_has_exact_tla_supply_and_cross_set_reprints() {
        let rows = set_printings("tla").unwrap();
        let mut rarities = [0; 4];
        for row in &rows {
            rarities[row.printing.rarity as usize] += 1;
            assert!(!row.printing.key.starts_with("tle."));
        }
        assert_eq!(rarities, [91, 110, 60, 20]);
        assert_eq!(
            rows.iter()
                .map(|r| r.printing.rarity.rotisserie_copies())
                .sum::<u32>(),
            664
        );
        for key in [
            "ptk.corrupt_court_official",
            "eld.epic_downfall",
            "ths.lightning_strike",
        ] {
            let row = rows.iter().find(|r| r.printing.key == key).unwrap();
            assert_ne!(
                row.printing.definition_scryfall_id,
                row.printing.scryfall_id
            );
        }
        assert!(set_printings("tle").is_err());
    }

    #[test]
    fn printing_metadata_rejects_missing_ambiguous_and_contradictory_rows() {
        let original: serde_json::Value = serde_json::from_str(TLA).unwrap();
        let mut missing = original.clone();
        missing["printings"].as_array_mut().unwrap().pop();
        assert!(validate("tla", &missing.to_string(), 281)
            .unwrap_err()
            .0
            .contains("expected 281"));
        let mut duplicate = original.clone();
        duplicate["printings"][1] = duplicate["printings"][0].clone();
        assert!(validate("tla", &duplicate.to_string(), 281)
            .unwrap_err()
            .0
            .contains("duplicate"));
        for (field, value) in [
            ("key", "tla.missing"),
            ("name", "Wrong"),
            ("set", "tle"),
            ("rarity", "unknown"),
            ("definition_scryfall_id", "bad"),
        ] {
            let mut bad = original.clone();
            bad["printings"][0][field] = value.into();
            assert!(validate("tla", &bad.to_string(), 281).is_err(), "{field}");
        }
    }
}
