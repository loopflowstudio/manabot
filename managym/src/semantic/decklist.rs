//! Decklist files: source-preserving records, parsing, name resolution and import.
//!
//! A [`Decklist`] is one submitted document in the `counted-lines-v1` dialect:
//! its exact source text (BOM and newlines included), filename, parsed entries
//! and line-specific issues. Importing resolves each entry's name against a
//! content pool and admits the resulting [`Deck`] through the same native
//! construction admission game creation uses, so a decklist never carries a
//! second legality policy. [`ImportedMatch`] is the private setup record of a
//! two-file import, binding the pool identity it was admitted against.
//!
//! Entry points: [`parse_decklist`], [`read_decklist`], [`admit_decklists`],
//! [`import_match`] and [`ImportedMatch::validate_current`].

use std::{collections::BTreeMap, io::Read, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use super::{
    content_pool,
    deck_admission::{admit_in_pool, DeckIssue, DeckSection},
    pool_content, ContentPool, IrError, PoolDefinition, PoolIdentity,
};
use crate::{cards::content_pack::ContentPack, state::deck::Deck};

pub const DIALECT: &str = "counted-lines-v1";
/// Operational bound on one source document, not a construction rule.
pub const MAX_SOURCE_BYTES: usize = 65_536;

/// One counted line. Resolution replaces `name` with the pool's registry name
/// and fills the definition identity; parsing alone leaves them empty.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct DeckEntry {
    pub line: usize,
    pub section: DeckSection,
    pub count: usize,
    pub name: String,
    pub card_def_id: Option<u32>,
    pub semantic_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct Decklist {
    pub dialect: String,
    pub source: String,
    pub filename: String,
    pub entries: Vec<DeckEntry>,
    pub issues: Vec<DeckIssue>,
}

/// One document's admission: its resolved source record and the admitted deck.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct DeckImport {
    pub decklist: Decklist,
    pub deck: Deck,
}

/// Private two-seat setup record; never attach submitted lists to viewer frames.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct ImportedMatch {
    pub pool: PoolIdentity,
    pub ruleset: String,
    pub hero: DeckImport,
    pub villain: DeckImport,
}

/// Decklists that failed admission, keyed by the caller's label (a seat, or a
/// position in a supply), each carrying its line diagnostics.
#[derive(Clone, Debug)]
pub enum DecklistError {
    Pool(IrError),
    Rejected(BTreeMap<String, Decklist>),
}

impl std::fmt::Display for DecklistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pool(error) => write!(f, "{error}"),
            Self::Rejected(decklists) => {
                let mut lines = Vec::new();
                for (label, decklist) in decklists {
                    for issue in &decklist.issues {
                        let line = issue.line.map_or("-".to_owned(), |line| line.to_string());
                        lines.push(format!(
                            "{label} {}:{line}: {}",
                            decklist.filename, issue.message
                        ));
                    }
                }
                write!(f, "{}", lines.join("; "))
            }
        }
    }
}

impl From<IrError> for DecklistError {
    fn from(error: IrError) -> Self {
        Self::Pool(error)
    }
}

// ---------------------------------------------------------------------------
// Parsing

/// Parse counts/names, Main/Deck/Sideboard headings and full-line `#` comments.
pub fn parse_decklist(source: &str, filename: &str) -> Decklist {
    let mut decklist = Decklist {
        dialect: DIALECT.into(),
        source: source.into(),
        filename: filename.into(),
        entries: Vec::new(),
        issues: Vec::new(),
    };
    if source.len() > MAX_SOURCE_BYTES {
        decklist.issues.push(DeckIssue {
            code: "size".into(),
            ..DeckIssue::format("Decklist exceeds the 64 KiB import limit.")
        });
        return decklist;
    }
    let mut section = DeckSection::Main;
    for (index, raw) in source_lines(source).enumerate() {
        let number = index + 1;
        let line = if number == 1 {
            raw.trim().trim_start_matches('\u{feff}').trim()
        } else {
            raw.trim()
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let heading = line.to_lowercase();
        match heading.strip_suffix(':').unwrap_or(&heading) {
            "main" | "deck" => {
                section = DeckSection::Main;
                continue;
            }
            "sideboard" => {
                section = DeckSection::Sideboard;
                continue;
            }
            _ => {}
        }
        let issue = |code: &str, card_name: Option<&str>, message: &str| DeckIssue {
            line: Some(number),
            section: Some(section),
            card_name: card_name.map(str::to_owned),
            code: code.into(),
            message: message.into(),
        };
        let Some((digits, name)) = counted_line(line) else {
            decklist.issues.push(issue(
                "syntax",
                None,
                "Expected COUNT Card Name, or a Main/Sideboard heading.",
            ));
            continue;
        };
        // Bound the magnitude before conversion so arbitrary numerals stay actionable.
        let magnitude = digits
            .trim_start_matches(['+', '-'])
            .trim_start_matches('0');
        if digits.starts_with('-') || magnitude.is_empty() || magnitude.len() > 9 {
            decklist.issues.push(issue(
                "count",
                Some(name),
                "Quantity must be a positive supported integer.",
            ));
            continue;
        }
        decklist.entries.push(DeckEntry {
            line: number,
            section,
            count: magnitude.parse().expect("nine ASCII digits"),
            name: name.into(),
            card_def_id: None,
            semantic_key: None,
        });
    }
    decklist
}

/// Lines split at `\r\n`, `\n` or `\r`, so CRLF and classic-Mac files number alike.
fn source_lines(source: &str) -> impl Iterator<Item = &str> {
    let mut rest = Some(source);
    std::iter::from_fn(move || {
        let text = rest?;
        match text.find(['\r', '\n']) {
            Some(end) => {
                let skip = if text[end..].starts_with("\r\n") {
                    2
                } else {
                    1
                };
                rest = Some(&text[end + skip..]).filter(|tail| !tail.is_empty());
                Some(&text[..end])
            }
            None => {
                rest = None;
                Some(text)
            }
        }
    })
}

/// `[+-]?[0-9]+`, whitespace, then a nonempty name. Lines are already trimmed.
fn counted_line(line: &str) -> Option<(&str, &str)> {
    let unsigned = line.strip_prefix(['+', '-']).unwrap_or(line);
    let digits = unsigned.len()
        - unsigned
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .len();
    let numeral_end = line.len() - unsigned.len() + digits;
    let rest = &line[numeral_end..];
    let name = rest.trim_start();
    (digits > 0 && name.len() < rest.len() && !name.is_empty())
        .then(|| (&line[..numeral_end], name))
}

/// Bounded UTF-8 read that keeps BOM and newline bytes.
pub fn read_decklist(path: &Path, filename: Option<&str>) -> Result<Decklist, String> {
    let display = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut data = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| {
            file.take(MAX_SOURCE_BYTES as u64 + 1)
                .read_to_end(&mut data)
        })
        .map_err(|error| format!("{display}: {error}"))?;
    if data.len() > MAX_SOURCE_BYTES {
        return Err(format!(
            "{display}: decklist exceeds the 64 KiB import limit"
        ));
    }
    let source =
        String::from_utf8(data).map_err(|_| format!("{display}: decklist is not UTF-8"))?;
    Ok(parse_decklist(&source, filename.unwrap_or(&display)))
}

// ---------------------------------------------------------------------------
// Resolution and admission

fn name_key(name: &str) -> String {
    name.nfc()
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Resolve front or full double-faced names; reverse-only, ambiguous and
/// unknown names are rejected without substitution. Then admit natively,
/// attributing each admission issue to every source line that named the card.
fn admit_with_pool(
    decklist: &Decklist,
    pool: &ContentPool,
    content: &ContentPack,
    ruleset: &str,
) -> Result<DeckImport, Decklist> {
    let mut names: BTreeMap<String, Vec<&PoolDefinition>> = BTreeMap::new();
    let mut reverse_faces: BTreeMap<String, &str> = BTreeMap::new();
    for definition in &pool.definitions {
        names
            .entry(name_key(&definition.registry_name))
            .or_default()
            .push(definition);
        if let Some((front, reverse)) = definition.registry_name.split_once(" // ") {
            names.entry(name_key(front)).or_default().push(definition);
            reverse_faces.insert(name_key(reverse), &definition.registry_name);
        }
    }
    let mut issues = decklist.issues.clone();
    let mut entries = Vec::new();
    let mut source_lines: BTreeMap<(DeckSection, String), Vec<usize>> = BTreeMap::new();
    let mut deck = Deck::default();
    for entry in &decklist.entries {
        let key = name_key(&entry.name);
        let candidates = names.get(&key).map(Vec::as_slice).unwrap_or_default();
        let issue = |code: &str, message: String| DeckIssue {
            line: Some(entry.line),
            section: Some(entry.section),
            card_name: Some(entry.name.clone()),
            code: code.into(),
            message,
        };
        let definition = match candidates {
            [definition] => *definition,
            [] if reverse_faces.contains_key(&key) => {
                issues.push(issue(
                    "face",
                    format!(
                        "Reverse faces cannot be deck entries; use {}.",
                        reverse_faces[&key]
                    ),
                ));
                continue;
            }
            [] => {
                issues.push(issue(
                    "unknown",
                    format!("Unknown card name: {}.", entry.name),
                ));
                continue;
            }
            _ => {
                issues.push(issue(
                    "ambiguous",
                    format!("Ambiguous card name: {}.", entry.name),
                ));
                continue;
            }
        };
        let name = definition.registry_name.clone();
        source_lines
            .entry((entry.section, name.clone()))
            .or_default()
            .push(entry.line);
        let cards = match entry.section {
            DeckSection::Main => &mut deck.main,
            DeckSection::Sideboard => &mut deck.sideboard,
        };
        *cards.entry(name.clone()).or_default() += entry.count;
        entries.push(DeckEntry {
            name,
            card_def_id: definition.card_def_id.map(|id| id.0),
            semantic_key: definition.semantic_key.clone(),
            ..entry.clone()
        });
    }
    for issue in admit_in_pool(pool, content, &deck, ruleset) {
        let lines = match (issue.section, &issue.card_name) {
            (Some(section), Some(name)) => source_lines
                .get(&(section, name.clone()))
                .cloned()
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if lines.is_empty() {
            issues.push(issue);
        } else {
            issues.extend(lines.into_iter().map(|line| DeckIssue {
                line: Some(line),
                ..issue.clone()
            }));
        }
    }
    let decklist = Decklist {
        entries,
        issues,
        ..decklist.clone()
    };
    if decklist.issues.is_empty() {
        Ok(DeckImport { decklist, deck })
    } else {
        Err(decklist)
    }
}

/// Admit labelled decklists against one pool, rejecting the whole input when
/// any document has an issue; nothing is silently filtered.
pub fn admit_decklists(
    decklists: &[(String, Decklist)],
    pack_key: &str,
    ruleset: &str,
) -> Result<Vec<DeckImport>, DecklistError> {
    let pool = content_pool(pack_key)?;
    let content = pool_content(pack_key)?;
    let mut admitted = Vec::new();
    let mut rejected = BTreeMap::new();
    for (label, decklist) in decklists {
        match admit_with_pool(decklist, &pool, &content, ruleset) {
            Ok(imported) => admitted.push(imported),
            Err(decklist) => {
                rejected.insert(label.clone(), decklist);
            }
        }
    }
    if rejected.is_empty() {
        Ok(admitted)
    } else {
        Err(DecklistError::Rejected(rejected))
    }
}

pub fn import_match(
    hero: &Decklist,
    villain: &Decklist,
    pack_key: &str,
    ruleset: &str,
) -> Result<ImportedMatch, DecklistError> {
    let mut imports = admit_decklists(
        &[
            ("hero".into(), hero.clone()),
            ("villain".into(), villain.clone()),
        ],
        pack_key,
        ruleset,
    )?;
    let villain = imports.pop().expect("two admitted decklists");
    let hero = imports.pop().expect("two admitted decklists");
    Ok(ImportedMatch {
        pool: content_pool(pack_key)?.identity(),
        ruleset: ruleset.into(),
        hero,
        villain,
    })
}

impl ImportedMatch {
    /// Game identity: pool, ruleset and both decks. Filenames, line order and
    /// comments cannot change it.
    pub fn setup_id(&self) -> String {
        let value = serde_json::json!({
            "decks": [&self.hero.deck, &self.villain.deck],
            "pool": &self.pool,
            "ruleset": &self.ruleset,
        });
        format!("{:x}", Sha256::digest(canonical_json(&value).as_bytes()))
    }

    /// The same imports with seats exchanged.
    pub fn swapped(&self) -> Self {
        Self {
            hero: self.villain.clone(),
            villain: self.hero.clone(),
            ..self.clone()
        }
    }

    /// Re-admit both retained sources against the installed pool; the record
    /// must match exactly, so tampered sources, decks or stale pools reject.
    pub fn validate_current(&self) -> Result<(), String> {
        let pool = content_pool(&self.pool.pack_key).map_err(|error| error.to_string())?;
        if pool.identity() != self.pool {
            return Err("Imported deck content identity differs from the installed pool".into());
        }
        let content = pool_content(&self.pool.pack_key).map_err(|error| error.to_string())?;
        for retained in [&self.hero, &self.villain] {
            let source = parse_decklist(&retained.decklist.source, &retained.decklist.filename);
            if admit_with_pool(&source, &pool, &content, &self.ruleset).as_ref() != Ok(retained) {
                return Err("Recorded imported deck differs from its admitted source".into());
            }
        }
        Ok(())
    }

    /// Bind a consumer's playable setup to this record: re-admit the sources,
    /// then require the same pool and both seats' decks.
    pub fn validate_setup(
        &self,
        content_pack: Option<&str>,
        hero: &Deck,
        villain: &Deck,
    ) -> Result<(), String> {
        self.validate_current()?;
        if content_pack != Some(self.pool.pack_key.as_str())
            || hero != &self.hero.deck
            || villain != &self.villain.deck
        {
            return Err("Match differs from its admitted decklist files".into());
        }
        Ok(())
    }
}

/// Compact JSON with object keys sorted, independent of serde_json features.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            let fields: Vec<_> = keys
                .into_iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        serde_json::Value::String(key.clone()),
                        canonical_json(&map[key])
                    )
                })
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        serde_json::Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = "tla-pool-v1";
    const HERO: &str = "# Novel development deck\nMain\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island\n";
    const VILLAIN: &str = "20 Island\n";

    fn admit(decklist: &Decklist, pool: &ContentPool) -> Result<DeckImport, Decklist> {
        admit_with_pool(
            decklist,
            pool,
            &pool_content(PACK).unwrap(),
            super::super::deck_admission::DEVELOPMENT_RULESET,
        )
    }

    fn lines(decklist: &Decklist) -> Vec<(Option<usize>, &str)> {
        decklist
            .issues
            .iter()
            .map(|issue| (issue.line, issue.code.as_str()))
            .collect()
    }

    #[test]
    fn dialect_keeps_original_lines_and_reports_bad_counts() {
        let source = "\u{feff}Deck\r\n# comment\r\n2 Forest\r\nSideboard:\r\n0 Island\r\n-1 Mountain\r\n2x Forest\r\n99999999999999999999 Island\r\n+3 Plains";
        let decklist = parse_decklist(source, "input.txt");
        assert_eq!(decklist.source, source);
        assert_eq!(decklist.filename, "input.txt");
        let entries: Vec<_> = decklist
            .entries
            .iter()
            .map(|e| (e.line, e.section, e.name.as_str(), e.count))
            .collect();
        assert_eq!(
            entries,
            [
                (3, DeckSection::Main, "Forest", 2),
                (9, DeckSection::Sideboard, "Plains", 3)
            ]
        );
        assert_eq!(
            lines(&decklist),
            [
                (Some(5), "count"),
                (Some(6), "count"),
                (Some(7), "syntax"),
                (Some(8), "count")
            ]
        );
        let oversized = parse_decklist(&"1 Forest\n".repeat(8000), "big.txt");
        assert_eq!(lines(&oversized), [(None, "size")]);
    }

    #[test]
    fn names_resolve_by_normalized_key_and_sum_counts() {
        let pool = content_pool(PACK).unwrap();
        let reformatted = parse_decklist(
            "10 mountain\n5 Forest\n5  FOREST\n2 Avatar Enthusiasts\nSideboard\n1 Island",
            "x",
        );
        let imported = admit(&reformatted, &pool).unwrap();
        let original = admit(&parse_decklist(HERO, "hero.txt"), &pool).unwrap();
        assert_eq!(imported.deck, original.deck);
        assert_eq!(
            imported.deck.main,
            BTreeMap::from([
                ("Avatar Enthusiasts".into(), 2),
                ("Forest".into(), 10),
                ("Mountain".into(), 10)
            ])
        );
        assert!(imported
            .decklist
            .entries
            .iter()
            .all(|e| e.card_def_id.is_some()));
    }

    #[test]
    fn unknown_unsupported_and_ambiguous_entries_name_their_lines() {
        let mut pool = content_pool(PACK).unwrap();
        let source = "10 Forest\n1 Definitely Not A Card\nSideboard\n1 Accumulate Wisdom\n2 Accumulate Wisdom\nMain\n1 Accumulate Wisdom\n";
        let rejected = admit(&parse_decklist(source, "x"), &pool).unwrap_err();
        assert_eq!(rejected.source, source);
        for expected in [
            (Some(2), "unknown"),
            (Some(4), "unsupported"),
            (Some(5), "unsupported"),
            (Some(7), "unsupported"),
        ] {
            assert!(lines(&rejected).contains(&expected), "{expected:?}");
        }
        let forest = pool
            .definitions
            .iter()
            .find(|row| row.registry_name == "Forest")
            .unwrap()
            .clone();
        pool.definitions.push(PoolDefinition {
            registry_name: "FOREST".into(),
            ..forest
        });
        let ambiguous = admit(&parse_decklist("10 Forest", "x"), &pool).unwrap_err();
        assert!(lines(&ambiguous).contains(&(Some(1), "ambiguous")));
    }

    /// The pool with Forest renamed to a two-face registry name.
    fn double_faced_pool() -> ContentPool {
        let mut pool = content_pool(PACK).unwrap();
        for row in &mut pool.definitions {
            if row.registry_name == "Forest" {
                row.registry_name = "Front Face // Back Face".into();
            }
        }
        pool
    }

    #[test]
    fn double_faced_entries_resolve_by_front_or_full_name_only() {
        let pool = double_faced_pool();
        for name in ["Front Face", "Front Face // Back Face"] {
            let result = admit(&parse_decklist(&format!("10 Island\n1 {name}"), "x"), &pool);
            let decklist = match result {
                Ok(imported) => imported.decklist,
                Err(decklist) => decklist,
            };
            let names: Vec<_> = decklist.entries.iter().map(|e| e.name.as_str()).collect();
            assert_eq!(names, ["Island", "Front Face // Back Face"]);
            assert!(!decklist
                .issues
                .iter()
                .any(|issue| issue.code == "ambiguous" || issue.code == "face"));
        }
        let reverse = admit(&parse_decklist("10 Island\n1 Back Face", "x"), &pool).unwrap_err();
        assert!(lines(&reverse).contains(&(Some(2), "face")));
    }

    #[test]
    fn construction_errors_reject_the_whole_match() {
        for source in [
            "Sideboard\n1 Island",
            "1 Forest\nSideboard\n0 Island",
            "1001 Forest",
        ] {
            let error = import_match(
                &parse_decklist(source, "hero.txt"),
                &parse_decklist(VILLAIN, "villain.txt"),
                PACK,
                "development-v1",
            )
            .unwrap_err();
            let DecklistError::Rejected(rejected) = &error else {
                panic!("{error}");
            };
            assert_eq!(rejected.keys().collect::<Vec<_>>(), ["hero"]);
            assert!(error.to_string().starts_with("hero hero.txt:"));
        }
        let error = import_match(
            &parse_decklist(HERO, "hero.txt"),
            &parse_decklist(VILLAIN, "villain.txt"),
            PACK,
            "standard",
        )
        .unwrap_err();
        assert!(error.to_string().contains("ruleset"));
    }

    #[test]
    fn retained_imports_reject_tampering_and_ignore_presentation() {
        let imported = import_match(
            &parse_decklist(&HERO.replace('\n', "\r\n"), "hero.txt"),
            &parse_decklist(VILLAIN, "villain.txt"),
            PACK,
            "development-v1",
        )
        .unwrap();
        imported.validate_current().unwrap();
        let restored: ImportedMatch =
            serde_json::from_str(&serde_json::to_string(&imported).unwrap()).unwrap();
        assert_eq!(restored, imported);
        assert!(serde_json::to_value(&imported).unwrap()["pool"]
            .get("definitions")
            .is_none());
        let reformatted = import_match(
            &parse_decklist(
                "10 mountain\n5 Forest\n5 forest\n2 Avatar Enthusiasts\nSideboard\n1 Island",
                "other.txt",
            ),
            &parse_decklist(VILLAIN, "villain.txt"),
            PACK,
            "development-v1",
        )
        .unwrap();
        assert_eq!(reformatted.setup_id(), imported.setup_id());

        let mut tampered = imported.clone();
        tampered.hero.decklist.source = "10 Island".into();
        assert!(tampered.validate_current().unwrap_err().contains("source"));
        let mut stale = imported.clone();
        stale.pool.content_digest = "stale".into();
        assert!(stale.validate_current().unwrap_err().contains("identity"));
    }
}
