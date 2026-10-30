//! A deck: main-deck and sideboard counts by card name.
//!
//! One concept whatever its source (an authored preset, an imported file, a
//! training supply). Identity derives from the counts alone; player names,
//! filenames and line order never identify a deck. Serialized keys are
//! `decklist`/`sideboard`, the names `PlayerConfig` receipts have always used.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, hash, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct Deck {
    #[serde(rename = "decklist")]
    pub main: BTreeMap<String, usize>,
    #[serde(default)]
    pub sideboard: BTreeMap<String, usize>,
}

impl Deck {
    pub fn new(main: BTreeMap<String, usize>, sideboard: BTreeMap<String, usize>) -> Self {
        Self { main, sideboard }
    }

    /// SHA-256 of `[main, sideboard]` as Python's `json.dumps(..., sort_keys=True)`
    /// writes it. Retained run evidence records this exact digest.
    pub fn identity(&self) -> String {
        let mut text = String::from("[");
        for (index, section) in [&self.main, &self.sideboard].into_iter().enumerate() {
            if index > 0 {
                text.push_str(", ");
            }
            text.push('{');
            for (position, (name, count)) in section.iter().enumerate() {
                if position > 0 {
                    text.push_str(", ");
                }
                push_python_json_string(&mut text, name);
                text.push_str(&format!(": {count}"));
            }
            text.push('}');
        }
        text.push(']');
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }
}

/// Python's default (`ensure_ascii=True`) JSON string encoding.
fn push_python_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matches_the_retained_python_digest() {
        // hashlib.sha256(json.dumps([{"Forest": 10, "é \"x\"": 1}, {"🂡": 2}],
        //     sort_keys=True).encode()).hexdigest()
        let deck = Deck::new(
            BTreeMap::from([("Forest".into(), 10), ("é \"x\"".into(), 1)]),
            BTreeMap::from([("🂡".into(), 2)]),
        );
        assert_eq!(
            deck.identity(),
            "109a643d1ecbd6ace50990b6ac987466eedd82baaa23805b36393a9281551c52"
        );
    }

    #[test]
    fn serialized_keys_are_the_player_config_keys() {
        let deck = Deck::new(BTreeMap::from([("Forest".into(), 1)]), BTreeMap::new());
        assert_eq!(
            serde_json::to_value(&deck).unwrap(),
            serde_json::json!({"decklist": {"Forest": 1}, "sideboard": {}})
        );
    }
}
